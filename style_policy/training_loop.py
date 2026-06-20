"""One-stage training: from_ce + to_ce (+ promo) with AMP, grad-accum, checkpoint + metrics JSON."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce, top1_legal
from style_policy.model_spec import elo_to_bucket
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def _step_loss(model, batch, device, n_elo, label_smoothing):
    packed = batch["packed_pre"].to(device)
    elo_idx = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
    from_logits, from_mask = model.forward_from(packed, batch["from_legal_u64"].to(device), elo_idx=elo_idx)
    to_logits, to_mask = model.forward_to(packed, batch["from_sq"].to(device), batch["to_legal_u64"].to(device), elo_idx=elo_idx)
    fl = masked_square_ce(from_logits, batch["from_sq"].to(device), from_mask, label_smoothing=label_smoothing)
    tl = masked_square_ce(to_logits, batch["to_sq"].to(device), to_mask, label_smoothing=label_smoothing)
    metrics = {"from_ce": fl.item(), "to_ce": tl.item(),
               "from_top1": top1_legal(from_logits, batch["from_sq"].to(device), from_mask),
               "to_top1": top1_legal(to_logits, batch["to_sq"].to(device), to_mask)}
    return fl + tl, metrics


def train_one_stage(spec: dict, stage_idx: int, device: str) -> dict:
    stage = spec["stages"][stage_idx - 1]
    arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    ckpt_dir = Path(spec["checkpoint_dir"]); (ckpt_dir / "metrics").mkdir(parents=True, exist_ok=True)
    model = BasePolicy.from_config(arch).to(device)
    if stage_idx > 1:
        prev = ckpt_dir / f"{spec['name']}_stage_{stage_idx-1}.pt"
        model.load_state_dict(torch.load(prev, map_location=device)["model"])
    ds = PackedMoveDataset(spec["train_h5"], sample_n=stage["sample"]["n"], seed=stage["sample"]["seed"])
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=True,
                    num_workers=stage["dataloader_num_workers"], collate_fn=PackedMoveDataset.collate)
    opt = torch.optim.AdamW(model.parameters(), lr=stage["train"]["learning_rate"], weight_decay=stage["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=stage["use_amp"] and device == "cuda")
    accum = int(stage.get("gradient_accumulation_steps", 1))
    model.train()
    for epoch in range(int(stage["train"]["epochs"])):
        opt.zero_grad()
        for i, batch in enumerate(dl):
            with torch.amp.autocast("cuda", enabled=stage["use_amp"] and device == "cuda"):
                loss, m = _step_loss(model, batch, device, n_elo, stage.get("label_smoothing", 0.0))
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if i % stage["log_interval"] == 0:
                print(f"epoch={epoch} step={i} loss={loss.item():.4f} "
                      f"from_top1={m['from_top1']*100:.1f}% to_top1={m['to_top1']*100:.1f}%")
    out = ckpt_dir / f"{spec['name']}_stage_{stage_idx}.pt"
    torch.save({"model": model.state_dict(), "architecture": arch}, out)
    rec = {"stage": stage_idx, "last_batch_metrics": m}
    (ckpt_dir / "metrics" / f"{spec['name']}_stage_{stage_idx}.json").write_text(json.dumps(rec, indent=2))
    print(f"Saved {out}")
    return rec
