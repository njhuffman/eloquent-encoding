"""CLI: python -m style_policy.train --model NAME --stage K  (0=init, K>=1 trains stages[K-1])."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from style_policy.model import BasePolicy
from style_policy.model_spec import load_spec
from style_policy.training_loop import train_one_stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Resume this stage from its {name}_stage_{K}.resume.pt if present")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_spec(args.model)
    ckpt_dir = Path(spec["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 0:
        model = BasePolicy.from_config(spec["architecture"])
        out = ckpt_dir / f"{spec['name']}_stage_0.pt"
        torch.save({"model": model.state_dict(), "architecture": spec["architecture"]}, out)
        print(f"Saved {out}"); return 0
    if not (1 <= args.stage <= len(spec["stages"])):
        print(f"--stage out of range (1..{len(spec['stages'])})", file=sys.stderr); return 1
    train_one_stage(spec, args.stage, device, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
