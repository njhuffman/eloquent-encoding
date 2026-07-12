"""Pass-2 training: FlatMultiTaskPolicy on 128M human moves + Stockfish depth-8 best-move + eval + WDL.
Four losses: human-move CE (flat, elo-cond) + SF-best-move CE (flat) + WDL CE + eval MSE, all over the
exact 1792 legal mask supplied by the dataloader. Mirrors multiband_train's scaffolding."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from style_policy.flat_policy import FlatMultiTaskPolicy, masked_move_ce
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import wdl_ce
from style_policy.model_spec import elo_to_bucket
from style_policy.multiband_train import _init_wandb


def _step(model, batch, device, n_elo, w, log_extra=False):
    packed = batch["packed_pre"].to(device)
    elo_b = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
    legal = batch["legal_mask"].to(device)                       # (B,1792) bool
    h_tgt = batch["human_move_idx"].to(device)
    result = batch["result"].to(device)
    s_tgt = batch["sf_move_idx"].to(device); s_valid = batch["sf_move_valid"].to(device)
    nv = batch["nnue_value"].to(device); n_valid = batch["nnue_valid"].to(device)

    cls, squares = model.encode(packed, hist=None)
    h_logits = model.human_logits(cls, squares, elo_b)
    s_logits = model.sf_move_logits(cls, squares)
    h_ce = masked_move_ce(h_logits, h_tgt, legal)
    s_ce = masked_move_ce(s_logits, s_tgt, legal, valid=s_valid)
    w_ce = wdl_ce(model.value_head(cls, elo_idx=elo_b), result)
    pred_v = model.eval_value(cls)
    e_mse = (((pred_v - nv) ** 2) * n_valid.float()).sum() / n_valid.float().sum().clamp(min=1.0)
    total = w["human"] * h_ce + w["sf_move"] * s_ce + w["wdl"] * w_ce + w["eval"] * e_mse
    md = {"human_ce": float(h_ce), "sf_move_ce": float(s_ce), "wdl_ce": float(w_ce), "eval_mse": float(e_mse)}
    if log_extra:
        with torch.no_grad():
            hm = (h_logits.masked_fill(~legal, -1e9).argmax(-1) == h_tgt).float().mean()
            sm = ((s_logits.masked_fill(~legal, -1e9).argmax(-1) == s_tgt).float() * s_valid.float()).sum() \
                / s_valid.float().sum().clamp(min=1.0)
            md["human_match"] = float(hm) * 100.0; md["sf_move_match"] = float(sm) * 100.0
    return total, md


@torch.no_grad()
def _validate(model, val_dl, device, n_elo, use_amp, amp_dtype):
    was = model.training; model.eval(); ce = 0.0; mm = 0.0; nb = 0
    for batch in val_dl:
        packed = batch["packed_pre"].to(device)
        elo_b = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
        legal = batch["legal_mask"].to(device); h_tgt = batch["human_move_idx"].to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
            cls, squares = model.encode(packed, hist=None)
            hl = model.human_logits(cls, squares, elo_b)
            ce += float(masked_move_ce(hl, h_tgt, legal))
            mm += float((hl.masked_fill(~legal, -1e9).argmax(-1) == h_tgt).float().mean()) * 100.0
        nb += 1
    if was:
        model.train()
    return ce / max(nb, 1), mm / max(nb, 1)


def _export(model, arch, ckpt_dir, name, do_compile):
    sd = model.state_dict()
    if do_compile:
        sd = {k.replace("encoder._orig_mod.", "encoder.", 1): v for k, v in sd.items()}
    torch.save({"architecture": arch, "model": sd}, ckpt_dir / f"{name}.pt")
    enc_sd = {k: v for k, v in sd.items() if k.startswith("encoder.") or k.startswith("value_head.")}
    torch.save({"architecture": arch, "model": enc_sd}, ckpt_dir / f"{name}_encoder.pt")


def train_flat(spec: dict, device: str, *, resume: bool = False) -> dict:
    stage = spec["stages"][0]; arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    name = spec["name"]; ckpt_dir = Path(spec["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(stage["use_amp"]); amp_dtype = torch.bfloat16
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = FlatMultiTaskPolicy.from_config(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=stage["train"]["learning_rate"],
                            weight_decay=stage["weight_decay"], fused=(device == "cuda"))
    w = {"human": float(stage.get("human_weight", 1.0)), "sf_move": float(stage.get("sf_move_weight", 1.0)),
         "wdl": float(stage.get("value_loss_weight", 1.0)), "eval": float(stage.get("nnue_weight", 1.0))}
    preshuffled = bool(spec.get("preshuffled", False))
    ds = PackedMoveDataset(spec["train_h5"], sample_n=stage["sample"]["n"], seed=stage["sample"]["seed"],
                           sequential=preshuffled, sf_labels_path=spec["sf_labels"], flat_mask=True)
    nw = int(stage["dataloader_num_workers"])
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=not preshuffled, num_workers=nw,
                    collate_fn=PackedMoveDataset.collate, pin_memory=(device == "cuda"),
                    persistent_workers=(nw > 0), prefetch_factor=(6 if nw > 0 else None))
    val_dl = None
    if spec.get("val_h5") and spec.get("val_sample"):
        vds = PackedMoveDataset(spec["val_h5"], sample_n=spec["val_sample"]["n"], seed=spec["val_sample"]["seed"],
                                flat_mask=True)
        val_dl = DataLoader(vds, batch_size=stage["batch_size"], shuffle=False, num_workers=nw,
                            collate_fn=PackedMoveDataset.collate)
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
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        step = int(st["step"]); best = float(st["best"])
        if sched is not None and st.get("scheduler"):
            sched.load_state_dict(st["scheduler"])

    li = int(stage["log_interval"]); vi = int(stage.get("val_interval", 0)); ci = int(stage.get("checkpoint_interval", 0))
    print(f"flat train: {name} steps={total_steps} compile={'on' if do_compile else 'off'} weights={w}", flush=True)
    run = _init_wandb(spec, stage, device); model.train()
    while step < total_steps:
        for batch in dl:
            if step >= total_steps:
                break
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                loss, m = _step(model, batch, device, n_elo, w, log_extra=(step % li == 0))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
            opt.step()
            if sched is not None:
                sched.step()
            if step % li == 0:
                print(f"step={step}/{total_steps} human_ce={m['human_ce']:.3f} sf_move_ce={m['sf_move_ce']:.3f} "
                      f"wdl={m['wdl_ce']:.3f} eval_mse={m['eval_mse']:.4f} "
                      f"human_match={m['human_match']:.2f}% sf_match={m['sf_move_match']:.2f}%", flush=True)
                if run is not None:
                    run.log({f"train/{k}": v for k, v in m.items()} | {"lr": opt.param_groups[0]["lr"]}, step=step)
            if val_dl is not None and vi and step > 0 and step % vi == 0:
                v_ce, v_mm = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype)
                print(f"  [val step={step}] human_ce={v_ce:.4f} human_match={v_mm:.2f}%", flush=True)
                best = min(best, v_ce)
                if run is not None:
                    run.log({"val/human_ce": v_ce, "val/human_match": v_mm}, step=step)
            if ci and step > 0 and step % ci == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                            "best": best, "scheduler": sched.state_dict() if sched else None}, resume_path)
            step += 1
    _export(model, arch, ckpt_dir, name, do_compile)
    fv_ce, fv_mm = (_validate(model, val_dl, device, n_elo, use_amp, amp_dtype) if val_dl else (float("nan"), float("nan")))
    print(f"saved {ckpt_dir}/{name}.pt + encoder; final human_ce={fv_ce:.4f} human_match={fv_mm:.2f}%", flush=True)
    if run is not None:
        run.finish()
    return {"steps": step, "final_human_ce": fv_ce, "final_human_match": fv_mm}
