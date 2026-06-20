"""Step-driven training: from_ce + to_ce with AMP (bf16 by default), grad-accum, periodic
validation, resumable checkpoints, and optional W&B logging.

The loop is driven by a global micro-batch step counter (not a plain epoch loop) so a
run can resume after interruption: a `.resume.pt` checkpoint carries model + optimizer
+ scaler + step + best_val. Resuming restores all of that and continues until the same
total step budget is reached. Data ordering is NOT byte-exact across a resume (the
dataloader re-shuffles) — only the total amount of training and the optimizer state are
preserved, which is what matters for SGD.

AMP dtype defaults to **bf16**: fp16 autocast overflows the forward pass (max 65504) once
activations grow, producing a NaN loss that GradScaler cannot recover from (it just skips
every step -> the model silently freezes). bf16 has fp32's exponent range, so it does not
overflow and needs no GradScaler. A fail-fast guard also aborts on any non-finite loss.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce, top1_legal
from style_policy.model_spec import elo_to_bucket

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _resolve_amp_dtype(name: str) -> torch.dtype:
    if name not in _AMP_DTYPES:
        raise ValueError(f"amp_dtype must be one of {sorted(_AMP_DTYPES)} (got {name!r})")
    return _AMP_DTYPES[name]


def _init_wandb(spec: dict, stage_idx: int, stage: dict, device: str):
    """Start a W&B run if spec has a 'wandb' block, else return None (logging disabled).

    The 'wandb' block is optional and keyed at the top level of the spec:
        wandb: {project: style_policy, mode: online|offline|disabled, entity: <optional>}
    Absent block -> no run (so smoke/CI configs train without a W&B login). When online,
    W&B samples GPU/CPU/VRAM utilization automatically — no extra code needed.
    """
    cfg = spec.get("wandb")
    if not cfg:
        return None
    import wandb
    return wandb.init(
        project=cfg.get("project", "style_policy"),
        entity=cfg.get("entity"),
        name=f"{spec['name']}_stage_{stage_idx}",
        mode=cfg.get("mode", "online"),
        config={"stage": stage_idx, "device": device, "architecture": spec["architecture"], **stage},
    )


def _step_loss(model, batch, device, n_elo, label_smoothing):
    packed = batch["packed_pre"].to(device)
    elo_idx = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
    from_logits, from_mask, to_logits, to_mask = model.forward_policy(
        packed, batch["from_sq"].to(device),
        batch["from_legal_u64"].to(device), batch["to_legal_u64"].to(device),
        elo_idx=elo_idx)
    fl = masked_square_ce(from_logits, batch["from_sq"].to(device), from_mask, label_smoothing=label_smoothing)
    tl = masked_square_ce(to_logits, batch["to_sq"].to(device), to_mask, label_smoothing=label_smoothing)
    metrics = {"from_ce": fl.item(), "to_ce": tl.item(),
               "from_top1": top1_legal(from_logits, batch["from_sq"].to(device), from_mask),
               "to_top1": top1_legal(to_logits, batch["to_sq"].to(device), to_mask)}
    return fl + tl, metrics


def _make_loader(h5: str, stage: dict, sample_n: int, seed: int, *, shuffle: bool):
    ds = PackedMoveDataset(h5, sample_n=sample_n, seed=seed)
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=shuffle,
                    num_workers=stage["dataloader_num_workers"], collate_fn=PackedMoveDataset.collate)
    return ds, dl


@torch.no_grad()
def _validate(model, val_dl, device, n_elo, use_amp, amp_dtype) -> dict:
    """Mean unsmoothed val metrics over a fixed subset. Restores train mode on exit."""
    was_training = model.training
    model.eval()
    tot = {"from_ce": 0.0, "to_ce": 0.0, "from_top1": 0.0, "to_top1": 0.0}
    nb = 0
    for batch in val_dl:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
            _, m = _step_loss(model, batch, device, n_elo, 0.0)
        for k in tot:
            tot[k] += m[k]
        nb += 1
    if was_training:
        model.train()
    nb = max(nb, 1)
    return {f"val/{k}": tot[k] / nb for k in tot}


def train_one_stage(spec: dict, stage_idx: int, device: str, *, resume: bool = False) -> dict:
    stage = spec["stages"][stage_idx - 1]
    arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"]); (ckpt_dir / "metrics").mkdir(parents=True, exist_ok=True)
    use_amp = bool(stage["use_amp"])
    amp_dtype = _resolve_amp_dtype(str(stage.get("amp_dtype", "bf16")))

    model = BasePolicy.from_config(arch).to(device)
    if stage_idx > 1:
        prev = ckpt_dir / f"{name}_stage_{stage_idx-1}.pt"
        model.load_state_dict(torch.load(prev, map_location=device)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=stage["train"]["learning_rate"], weight_decay=stage["weight_decay"])
    # GradScaler is only needed/valid for fp16; bf16 has fp32 range and must NOT be scaled.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda" and amp_dtype == torch.float16)
    accum = int(stage.get("gradient_accumulation_steps", 1))

    train_ds, train_dl = _make_loader(spec["train_h5"], stage, stage["sample"]["n"], stage["sample"]["seed"], shuffle=True)
    val_dl = None
    if spec.get("val_h5") and spec.get("val_sample"):
        _, val_dl = _make_loader(spec["val_h5"], stage, spec["val_sample"]["n"], spec["val_sample"]["seed"], shuffle=False)
    val_interval = int(stage.get("val_interval", 0))
    ckpt_interval = int(stage.get("checkpoint_interval", 0))

    steps_per_epoch = math.ceil(len(train_ds) / stage["batch_size"])
    total_steps = steps_per_epoch * int(stage["train"]["epochs"])

    resume_path = ckpt_dir / f"{name}_stage_{stage_idx}.resume.pt"
    best_val = float("inf")
    global_step = 0
    if resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"]); global_step = int(st["global_step"]); best_val = float(st["best_val"])
        print(f"Resumed from {resume_path} at step {global_step}/{total_steps} (best_val={best_val:.4f})")

    def _save_resume():
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "global_step": global_step, "best_val": best_val, "architecture": arch}, resume_path)

    print(f"amp={'on' if use_amp else 'off'} dtype={amp_dtype} grad_scaler={'on' if scaler.is_enabled() else 'off'}")
    run = _init_wandb(spec, stage_idx, stage, device)
    model.train(); opt.zero_grad()
    pending = False          # grads accumulated since the last optimizer step
    m = {"from_ce": float("nan"), "to_ce": float("nan"), "from_top1": 0.0, "to_top1": 0.0}
    final_val: dict = {}

    while global_step < total_steps:
        for batch in train_dl:
            if global_step >= total_steps:
                break
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                loss, m = _step_loss(model, batch, device, n_elo, stage.get("label_smoothing", 0.0))
            # Fail fast: a non-finite loss means the run has diverged. Don't silently freeze
            # for thousands of steps (the fp16-overflow failure mode). Save a diagnostic and abort.
            if not torch.isfinite(loss):
                _save_resume()
                if run is not None:
                    run.finish(exit_code=1)
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) at step {global_step}; training diverged. "
                    f"State saved to {resume_path}. (amp_dtype={amp_dtype}; if fp16, switch to bf16.)")
            scaler.scale(loss / accum).backward()
            pending = True
            if (global_step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
                scaler.step(opt); scaler.update(); opt.zero_grad()
                pending = False

            if global_step % stage["log_interval"] == 0:
                print(f"step={global_step}/{total_steps} loss={loss.item():.4f} "
                      f"from_top1={m['from_top1']*100:.1f}% to_top1={m['to_top1']*100:.1f}%")
                if run is not None:
                    run.log({"train/loss": loss.item(), "train/from_ce": m["from_ce"], "train/to_ce": m["to_ce"],
                             "train/from_top1": m["from_top1"], "train/to_top1": m["to_top1"],
                             "lr": opt.param_groups[0]["lr"]}, step=global_step)

            if val_dl is not None and val_interval > 0 and global_step > 0 and global_step % val_interval == 0:
                vm = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype)
                vloss = vm["val/from_ce"] + vm["val/to_ce"]
                print(f"  [val step={global_step}] from_ce={vm['val/from_ce']:.4f} to_ce={vm['val/to_ce']:.4f} "
                      f"from_top1={vm['val/from_top1']*100:.1f}% to_top1={vm['val/to_top1']*100:.1f}%")
                if run is not None:
                    run.log({**vm, "val/loss": vloss}, step=global_step)
                if vloss < best_val:
                    best_val = vloss
                    torch.save({"model": model.state_dict(), "architecture": arch,
                                "best_val": best_val, "global_step": global_step},
                               ckpt_dir / f"{name}_stage_{stage_idx}.best.pt")

            if ckpt_interval > 0 and global_step > 0 and global_step % ckpt_interval == 0:
                _save_resume()

            global_step += 1

    # flush any partial accumulation group left at the step budget / epoch boundary
    if pending:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
        scaler.step(opt); scaler.update(); opt.zero_grad()

    if val_dl is not None:
        final_val = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype)
        vloss = final_val["val/from_ce"] + final_val["val/to_ce"]
        print(f"[final val] from_ce={final_val['val/from_ce']:.4f} to_ce={final_val['val/to_ce']:.4f} "
              f"from_top1={final_val['val/from_top1']*100:.1f}% to_top1={final_val['val/to_top1']*100:.1f}%")
        if run is not None:
            run.log({**final_val, "val/loss": vloss}, step=global_step)
        if vloss < best_val:
            best_val = vloss
            torch.save({"model": model.state_dict(), "architecture": arch,
                        "best_val": best_val, "global_step": global_step},
                       ckpt_dir / f"{name}_stage_{stage_idx}.best.pt")

    out = ckpt_dir / f"{name}_stage_{stage_idx}.pt"
    torch.save({"model": model.state_dict(), "architecture": arch}, out)
    _save_resume()
    rec = {"stage": stage_idx, "steps": global_step, "best_val": best_val,
           "last_train_metrics": m, "final_val_metrics": final_val}
    (ckpt_dir / "metrics" / f"{name}_stage_{stage_idx}.json").write_text(json.dumps(rec, indent=2))
    print(f"Saved {out}")
    if run is not None:
        run.finish()
    return rec
