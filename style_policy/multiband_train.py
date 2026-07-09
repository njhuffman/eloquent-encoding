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
from style_policy.history import horizon_dropout, binary_history_dropout


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


def masked_square_softce(logits, target, mask):
    """Soft cross-entropy: -sum_sq target(sq) * log_softmax(masked logits)(sq), mean over batch.
    `target` is a probability vector over the 64 squares (0 on illegal squares)."""
    logits = logits.masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1).float()
    target = target.float()
    contrib = target * logp
    contrib = torch.where(target > 0, contrib, torch.zeros_like(contrib))  # avoid 0*-inf = nan
    return -contrib.sum(-1).mean()


def _routed_distill_loss(model, cls, squares, hidx, from_sq, maia_from, maia_to, fmask, tmask):
    """Factored KL-to-Maia-3: match Maia-3's P(from) with the from-head, and its P(to|true-from)
    with the to-head conditioned on the true human from (same conditioning as CE training)."""
    B = squares.shape[0]
    fl = squares.new_zeros(()); tl = squares.new_zeros(())
    for g in range(model.n_bands):
        m = hidx == g; k = int(m.sum())
        if k == 0:
            continue
        sq = squares[m]; cl = cls[m]
        fl = fl + masked_square_softce(model.heads[g].from_logits(sq, cl), maia_from[m], fmask[m]) * k
        tl = tl + masked_square_softce(model.heads[g].to_logits(sq, from_sq[m], cl), maia_to[m], tmask[m]) * k
    return fl / B, tl / B


@torch.no_grad()
def _routed_human_metrics(model, cls, squares, hidx, from_sq, to_sq, fmask, tmask):
    """Hard human-move CE (from+to) + joint top-1 move-match vs the TRUE human move, for ANY
    training objective. Lets the distill and human runs log a directly comparable curve
    (train/from_ce differs between them since distill's is soft-CE-to-Maia-3)."""
    B = squares.shape[0]; ce = 0.0; match = 0; NEG = float("-inf")
    for g in range(model.n_bands):
        mm = hidx == g; k = int(mm.sum())
        if k == 0:
            continue
        sq = squares[mm]; cl = cls[mm]; fsq = from_sq[mm]; tsq = to_sq[mm]
        fl = model.heads[g].from_logits(sq, cl).masked_fill(~fmask[mm], NEG)
        tl = model.heads[g].to_logits(sq, fsq, cl).masked_fill(~tmask[mm], NEG)
        ce += (torch.nn.functional.cross_entropy(fl, fsq, reduction="sum")
               + torch.nn.functional.cross_entropy(tl, tsq, reduction="sum")).item()
        match += ((fl.argmax(-1) == fsq) & (tl.argmax(-1) == tsq)).sum().item()
    return ce / B, 100.0 * match / B


def _step(model, batch, device, n_elo, ls, vlw, last_move_dropout: float = 0.0,
          dropout_mode: str = "horizon", distill: bool = False, log_human: bool = False):
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
            drop_fn = binary_history_dropout if dropout_mode == "binary" else horizon_dropout
            hf, ht, hc = drop_fn(hf, ht, hc, p=last_move_dropout)
        hist = (hf, ht, hc)
    else:
        hist = None
    cls, squares = model.encode(packed, hist=hist)
    if distill:
        maia_from = batch["maia_from"].to(device); maia_to = batch["maia_to"].to(device)
        fl, tl = _routed_distill_loss(model, cls, squares, hidx, from_sq, maia_from, maia_to, fmask, tmask)
    else:
        fl, tl = _routed_policy_loss(model, cls, squares, hidx, from_sq, to_sq, fmask, tmask, ls)
    vl = wdl_ce(model.value_head(cls, elo_idx=elo_to_bucket(elo, n_elo).to(device)), result)
    md = {"from_ce": float(fl), "to_ce": float(tl), "wdl_ce": float(vl)}
    if log_human:
        h_ce, h_match = _routed_human_metrics(model, cls, squares, hidx, from_sq, to_sq, fmask, tmask)
        md["human_ce"] = h_ce; md["human_match"] = h_match
    return fl + tl + vlw * vl, md


@torch.no_grad()
def _validate(model, val_dl, device, n_elo, use_amp, amp_dtype):
    """Held-out HUMAN move CE (from+to) + joint move-match, computed from the TRUE human move
    regardless of training objective -> distill and CE runs overlay on the same comparable,
    low-noise curve. Returns (human_ce, human_match%)."""
    was = model.training; model.eval(); ce = 0.0; mm = 0.0; nb = 0
    for batch in val_dl:
        packed = batch["packed_pre"].to(device); elo = batch["elo_to_move"]
        hidx = model.head_index(elo).to(device)
        from_sq = batch["from_sq"].to(device); to_sq = batch["to_sq"].to(device)
        fmask = u64_to_mask(batch["from_legal_u64"].to(device)); tmask = u64_to_mask(batch["to_legal_u64"].to(device))
        hist = None
        if "hist_from" in batch:
            hist = (batch["hist_from"].to(device), batch["hist_to"].to(device), batch["hist_cap"].to(device))
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
            cls, squares = model.encode(packed, hist=hist)
            h_ce, h_mm = _routed_human_metrics(model, cls, squares, hidx, from_sq, to_sq, fmask, tmask)
        ce += h_ce; mm += h_mm; nb += 1
    if was:
        model.train()
    return ce / max(nb, 1), mm / max(nb, 1)


def _snapshot(model, arch, ckpt_dir, name, step, do_compile):
    """Write a step-tagged joint checkpoint ({name}.step{step}.pt), loadable directly by
    MultiBandPolicy.from_config + history_ksweep (carries architecture + bands; compile prefix
    stripped). Used to capture scale milestones during a single long run for a scaling sweep."""
    sd = model.state_dict()
    if do_compile:
        sd = {k.replace("encoder._orig_mod.", "encoder.", 1): v for k, v in sd.items()}
    torch.save({"architecture": arch, "bands": model.bands, "model": sd},
               ckpt_dir / f"{name}.step{step}.pt")


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


def _load_encoder_weights(model, ckpt_path, device) -> int:
    """Warm-start: copy ONLY encoder.* tensors from a checkpoint into `model`, leaving heads +
    value head at their fresh init. Strips a possible torch.compile prefix so keys match the
    (uncompiled) fresh model. Returns the number of encoder tensors loaded."""
    sd = torch.load(ckpt_path, map_location=device)["model"]
    enc_sd = {k.replace("encoder._orig_mod.", "encoder.", 1): v
              for k, v in sd.items() if k.startswith("encoder.")}
    model.load_state_dict(enc_sd, strict=False)  # strict=False: heads/value_head keys are absent
    return len(enc_sd)


def _freeze_encoder(model) -> int:
    """Freeze the encoder in place (requires_grad=False on every encoder param).
    Returns the number of param tensors frozen."""
    n = 0
    for p in model.encoder.parameters():
        p.requires_grad_(False)
        n += 1
    return n


def _trainable_params(model):
    """Params to hand the optimizer: only those with requires_grad. With nothing frozen this is
    exactly model.parameters() in the same order (backward-compatible)."""
    return [p for p in model.parameters() if p.requires_grad]


def train_multiband(spec: dict, device: str, *, resume: bool = False) -> dict:
    stage = spec["stages"][0]; arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    name = spec["name"]; ckpt_dir = Path(spec["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(stage["use_amp"]); amp_dtype = torch.bfloat16
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = MultiBandPolicy.from_config(arch).to(device)
    # Warm-start the encoder from an existing checkpoint (generic-encoder experiment). Only on a
    # fresh run: when resuming, encoder weights come from resume.pt instead.
    init_encoder_from = spec.get("init_encoder_from")
    if init_encoder_from and not resume:
        n_enc = _load_encoder_weights(model, init_encoder_from, device)
        print(f"init_encoder_from: loaded {n_enc} encoder tensors from {init_encoder_from}", flush=True)
    # Freeze the encoder so only heads + value head train (applies on fresh and resumed runs so a
    # resumed frozen run stays frozen). The optimizer is then built over trainable params only.
    freeze_encoder = bool(spec.get("freeze_encoder", False))
    if freeze_encoder:
        n_frozen = _freeze_encoder(model)
        print(f"freeze_encoder: froze {n_frozen} encoder param tensors", flush=True)
    opt = torch.optim.AdamW(_trainable_params(model), lr=stage["train"]["learning_rate"],
                            weight_decay=stage["weight_decay"], fused=(device == "cuda"))
    preshuffled = bool(spec.get("preshuffled", False))
    preload = bool(spec.get("preload_ram", False))
    ds = PackedMoveDataset(spec["train_h5"], sample_n=stage["sample"]["n"],
                           seed=stage["sample"]["seed"], sequential=preshuffled, preload=preload)
    _nw = 0 if preload else int(stage["dataloader_num_workers"])   # preload -> single process (RAM held once)
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=not preshuffled,
                    num_workers=_nw, collate_fn=PackedMoveDataset.collate,
                    pin_memory=(device == "cuda"),
                    persistent_workers=(_nw > 0), prefetch_factor=(6 if _nw > 0 else None))
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
    distill = bool(spec.get("distill", False))
    if distill:
        print("DISTILL mode: factored KL to Maia-3 soft targets (maia_from / maia_to)", flush=True)
    lmd = float(stage.get("last_move_dropout", 0.0))
    dropout_mode = stage.get("history_dropout", "horizon")  # "horizon" (graded) | "binary"
    val_interval = int(stage.get("val_interval", 0)); ckpt_interval = int(stage.get("checkpoint_interval", 0))
    snapshot_steps = {int(s) for s in stage.get("snapshot_steps", [])}  # step-tagged scale milestones
    print(f"multiband train: {name} steps={total_steps} compile={'on' if do_compile else 'off'}")
    run = _init_wandb(spec, stage, device)
    model.train(); model.encoder  # noqa
    while step < total_steps:
        for batch in dl:
            if step >= total_steps:
                break
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                loss, m = _step(model, batch, device, n_elo, ls, vlw, last_move_dropout=lmd,
                                dropout_mode=dropout_mode, distill=distill,
                                log_human=(step % stage["log_interval"] == 0))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
            opt.step()
            if sched is not None:
                sched.step()
            if step % stage["log_interval"] == 0:
                hstr = f" human_ce={m['human_ce']:.3f} human_match={m['human_match']:.2f}%" if "human_ce" in m else ""
                print(f"step={step}/{total_steps} from_ce={m['from_ce']:.3f} to_ce={m['to_ce']:.3f} wdl={m['wdl_ce']:.3f}{hstr}", flush=True)
                if run is not None:
                    logd = {"train/from_ce": m["from_ce"], "train/to_ce": m["to_ce"],
                            "train/wdl_ce": m["wdl_ce"], "lr": opt.param_groups[0]["lr"]}
                    if "human_ce" in m:
                        logd["train/human_ce"] = m["human_ce"]; logd["train/human_match"] = m["human_match"]
                    run.log(logd, step=step)
            if val_dl is not None and val_interval and step > 0 and step % val_interval == 0:
                v_ce, v_mm = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype)
                print(f"  [val step={step}] human_ce={v_ce:.4f} human_match={v_mm:.2f}%", flush=True)
                best = min(best, v_ce)
                if run is not None:
                    run.log({"val/human_ce": v_ce, "val/human_match": v_mm}, step=step)
            if ckpt_interval and step > 0 and step % ckpt_interval == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                            "best": best, "scheduler": sched.state_dict() if sched else None}, resume_path)
            if step in snapshot_steps:
                _snapshot(model, arch, ckpt_dir, name, step, do_compile)
                print(f"  [snapshot step={step}] wrote {name}.step{step}.pt", flush=True)
            step += 1
    if val_dl:
        fv_ce, fv_mm = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype)
    else:
        fv_ce, fv_mm = float("nan"), float("nan")
    _export(model, arch, ckpt_dir, name, do_compile)
    print(f"saved {ckpt_dir}/{name}.pt + encoder + per-band heads; final human_ce={fv_ce:.4f} human_match={fv_mm:.2f}%")
    if run is not None:
        run.finish()
    return {"steps": step, "final_human_ce": fv_ce, "final_human_match": fv_mm}
