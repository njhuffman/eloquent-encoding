"""Launch Pass-2 flat multi-task training.
  Sequential (masked, after labeling done): python scripts/train_flat.py --model flat_multitask_128M
  Overlap (unmasked, while labeling runs):  python scripts/train_flat.py --model flat_multitask_128M --follow
"""
import argparse, torch
from style_policy.model_spec import load_spec
from style_policy.flat_train import train_flat, train_flat_follow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flat_multitask_128M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--follow", action="store_true", help="overlap labeling: unmasked, track the labeled frontier")
    a = ap.parse_args()
    fn = train_flat_follow if a.follow else train_flat
    rec = fn(load_spec(a.model), a.device, resume=a.resume)
    print(rec)


if __name__ == "__main__":
    main()
