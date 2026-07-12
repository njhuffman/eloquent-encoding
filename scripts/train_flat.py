"""Launch Pass-2 flat multi-task training: python scripts/train_flat.py --model flat_multitask_128M"""
import argparse, torch
from style_policy.model_spec import load_spec
from style_policy.flat_train import train_flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flat_multitask_128M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    rec = train_flat(load_spec(a.model), a.device, resume=a.resume)
    print(rec)


if __name__ == "__main__":
    main()
