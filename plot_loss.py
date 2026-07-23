"""Plot the training loss curve stored in a checkpoint.

    python plot_loss.py --checkpoint ckpt.pt --out loss_curve.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def smooth(xs, k=25):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k)
        out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--out", default="loss_curve.png")
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    losses = ckpt["train_loss_curve"]
    steps = range(1, len(losses) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, color="#c7c7d8", lw=0.8, label="loss")
    plt.plot(steps, smooth(losses), color="#2b5fff", lw=1.8, label="smoothed (25)")
    plt.xlabel("optimizer step")
    plt.ylabel("training loss (cross-entropy per token)")
    plt.title(f"Final run — {ckpt.get('steps')} steps, final loss {sum(losses[-50:])/50:.3f}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"saved {args.out}  ({len(losses)} points)")


if __name__ == "__main__":
    main()
