"""Joint training: shared elo-agnostic encoder + N per-band heads (routed CE) + shared value head.
Mirrors train_one_stage's mechanics (compile/fused-AdamW/cosine/val/resume) with per-band routing."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce, wdl_ce
from style_policy.legal_mask import u64_to_mask
from style_policy.model_spec import elo_to_bucket
from style_policy.history import horizon_dropout


def _init_wandb(spec, stage, device):
    """Start a W&B run if spec has a 'wandb' block, else None (mirrors training_loop._init_wandb)."""
    cfg = spec.get("wandb")
    if not cfg:
        return None
    import wandb
    return wandb.init(project=cfg.get("project", "style_policy"), entity=cfg.get("entity"),
                      name=spec["name"], mode=cfg.get("mode", "online"),
                      config={"device": device, "architecture": spec["architecture"], **stage})


def _routed_policy_loss(model, cls, squares, hidx, from_sq, to_sq, fmask, tmask, ls):
    B = squares.shape[0]
    fl = squares.new_zeros(()); tl = squares.new_zeros(())
    for g in range(model.n_bands):
        m = hidx == g
        k = int(m.sum())
        if k == 0:
            continue
        sq = squares[m]
        cl = cls[m]  # index cls to the band's row subset
        fl = fl + masked_square_ce(model.heads[g].from_logits(sq, cl), from_sq[m], fmask[m], label_smoothing=ls) * k
        tl = tl + masked_square_ce(model.heads[g].to_logits(sq, from_sq[m], cl), to_sq[m], tmask[m], label_smoothing=ls) * k
    return fl / B, tl / B


def _step(model, batch, device, n_elo, ls, vlw, last_move_dropout: float = 0.0):
    packed = batch["packed_pre"].to(device)
    elo = batch["elo_to_move"]
    hidx = model.head_index(elo).to(device)
    from_sq = batch["from_sq"].to(device); to_sq = batch["to_sq"].to(device)
    fmask = u64_to_mask(batch["from_legal_u64"].to(device))
    tmask = u64_to_mask(batch["to_legal_u64"].to(device))
    result = batch["result"].to(device)
    # Build hist tuple if batch carries history columns; else pass None (backward-compat).
    if "hist_from" in batch:
        hf = batch["hist_from"].to(device)
        ht = batch["hist_to"].to(device)
        hc = batch["hist_cap"].to(device)
        if last_move_dropout > 0.0:
            hf, ht, hc = horizon_dropout(hf, ht, hc, p=last_move_dropout)
        hist = (hf, ht, hc)
    else:
        hist = None
    cls, squares = model.encode(packed, hist=hist)
    fl, tl = _routed_policy_loss(model, cls, squares, hidx, from_sq, to_sq, fmask, tmask, ls)
    vl = wdl_ce(model.value_head(cls, elo_idx=elo_to_bucket(elo, n_elo).to(device)), result)
    return fl + tl + vlw * vl, {"from_ce": float(fl), "to_ce": float(tl), "wdl_ce": float(vl)}


@torch.no_grad()
def _validate(model, val_dl, device, n_elo, use_amp, amp_dtype, vlw):
    was = model.training; model.eval(); tot = 0.0; nb = 0
    for batch in val_dl:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
            loss, m = _step(model, batch, device, n_elo, 0.0, vlw, last_move_dropout=0.0)
        tot += m["from_ce"] + m["to_ce"]; nb += 1
    if was:
        model.train()
    return tot / max(nb, 1)


def _export(model, arch, ckpt_dir, name, do_compile):
    sd = model.state_dict()
    if do_compile:
        sd = {k.replace("encoder._orig_mod.", "encoder.", 1): v for k, v in sd.items()}
    enc_sd = {k: v for k, v in sd.items() if k.startswith("encoder.") or k.startswith("value_head.")}
    enc_path = ckpt_dir / f"{name}_encoder.pt"
    torch.save({"architecture": arch, "model": enc_sd}, enc_path)
    torch.save({"architecture": arch, "bands": model.bands, "model": sd}, ckpt_dir / f"{name}.pt")
    d = int(arch["d_model"]); h = int(arch["head_hidden"])
    hd = ckpt_dir / "band_heads"; hd.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(model.bands):
        pre = f"heads.{i}."
        hsd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        torch.save({"band_head": hsd, "d_model": d, "hidden": h,
                    "source_checkpoint": str(enc_path), "band": int(b)}, hd / f"{name}_band_{b}.pt")


def train_multiband(spec: dict, device: str, *, resume: bool = False) -> dict:
    stage = spec["stages"][0]; arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    name = spec["name"]; ckpt_dir = Path(spec["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(stage["use_amp"]); amp_dtype = torch.bfloat16
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = MultiBandPolicy.from_config(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=stage["train"]["learning_rate"],
                            weight_decay=stage["weight_decay"], fused=(device == "cuda"))
    presorted = bool(spec.get("presorted", False))
    ds = PackedMoveDataset(spec["train_h5"], sample_n=stage["sample"]["n"],
                           seed=stage["sample"]["seed"], sequential=presorted)
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=not presorted,
                    num_workers=stage["dataloader_num_workers"], collate_fn=PackedMoveDataset.collate)
    val_dl = None
    if spec.get("val_h5") and spec.get("val_sample"):
        vds = PackedMoveDataset(spec["val_h5"], sample_n=spec["val_sample"]["n"], seed=spec["val_sample"]["seed"])
        val_dl = DataLoader(vds, batch_size=stage["batch_size"], shuffle=False,
                            num_workers=stage["dataloader_num_workers"], collate_fn=PackedMoveDataset.collate)
    total_steps = math.ceil(len(ds) / stage["batch_size"]) * int(stage["train"]["epochs"])
    warmup = int(stage.get("warmup_steps", 0)); lr_min = float(stage.get("lr_min_frac", 0.0))
    sched = None
    if str(stage.get("lr_schedule", "constant")) == "cosine":
        def _lam(s):
            if warmup > 0 and s < warmup:
                return (s + 1) / warmup
            p = min(1.0, (s - warmup) / max(1, total_steps - warmup))
            return lr_min + (1.0 - lr_min) * 0.5 * (1.0 + math.cos(math.pi * p))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lam)

    do_compile = bool(stage.get("compile", True)) and device == "cuda"
    if do_compile:
        model.encoder = torch.compile(model.encoder)

    resume_path = ckpt_dir / f"{name}.resume.pt"; step = 0; best = float("inf")
    if resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); step = int(st["step"]); best = float(st["best"])
        if sched is not None and st.get("scheduler"):
            sched.load_state_dict(st["scheduler"])

    ls = stage.get("label_smoothing", 0.0); vlw = stage.get("value_loss_weight", 1.0)
    lmd = float(stage.get("last_move_dropout", 0.0))
    val_interval = int(stage.get("val_interval", 0)); ckpt_interval = int(stage.get("checkpoint_interval", 0))
    print(f"multiband train: {name} steps={total_steps} compile={'on' if do_compile else 'off'}")
    run = _init_wandb(spec, stage, device)
    model.train(); model.encoder  # noqa
    while step < total_steps:
        for batch in dl:
            if step >= total_steps:
                break
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                loss, m = _step(model, batch, device, n_elo, ls, vlw, last_move_dropout=lmd)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
            opt.step()
            if sched is not None:
                sched.step()
            if step % stage["log_interval"] == 0:
                print(f"step={step}/{total_steps} from_ce={m['from_ce']:.3f} to_ce={m['to_ce']:.3f} wdl={m['wdl_ce']:.3f}", flush=True)
                if run is not None:
                    run.log({"train/from_ce": m["from_ce"], "train/to_ce": m["to_ce"],
                             "train/wdl_ce": m["wdl_ce"], "lr": opt.param_groups[0]["lr"]}, step=step)
            if val_dl is not None and val_interval and step > 0 and step % val_interval == 0:
                v = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype, vlw)
                print(f"  [val step={step}] from+to_ce={v:.4f}", flush=True)
                best = min(best, v)
                if run is not None:
                    run.log({"val/from_to_ce": v}, step=step)
            if ckpt_interval and step > 0 and step % ckpt_interval == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                            "best": best, "scheduler": sched.state_dict() if sched else None}, resume_path)
            step += 1
    final_val = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype, vlw) if val_dl else float("nan")
    _export(model, arch, ckpt_dir, name, do_compile)
    print(f"saved {ckpt_dir}/{name}.pt + encoder + per-band heads; final_val={final_val:.4f}")
    if run is not None:
        run.finish()
    return {"steps": step, "final_val": final_val}
