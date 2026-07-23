"""Trainer. Every knob is a flag so one file runs every experiment and the
checkpoint records the full config, which is what evaluate.py rebuilds from.

    python train.py --data data/train_corpus.txt --steps 2000 --out ckpt.pt

Hard caps are asserted below: a run over 2000 steps or 2M params aborts before
it can write a checkpoint.
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, gen):
    ix = torch.randint(len(ids) - block - 1, (batch,), generator=gen)
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x, y


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    if args.schedule == "constant":
        return args.lr
    if args.schedule == "wsd":
        hold = int(args.steps * (1 - args.decay_frac))
        if step < hold:
            return args.lr
        prog = (step - hold) / max(1, args.steps - hold)
        return args.min_lr + (args.lr - args.min_lr) * (1 - prog)
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))


def make_optimizer(model, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 and "emb" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": args.wd},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


@torch.no_grad()
def dev_bpb(model, cfg, tok, text):
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_bytes = len(text.encode("utf-8"))
    block, stride = cfg.block_size, max(1, cfg.block_size // 2)
    total_nll, scored = 0.0, 1
    model.eval()
    while scored < len(ids):
        start = max(0, scored - stride)
        end = min(len(ids), start + block)
        logits, _ = model(ids[start:end][None, :])
        logp = torch.log_softmax(logits[0], dim=-1)
        targets = ids[start + 1:end]
        nll = -logp[torch.arange(len(targets)), targets]
        total_nll += nll[scored - (start + 1):].sum().item()
        scored = end
    model.train()
    return total_nll / math.log(2) / n_bytes


def save(path, model, cfg, steps, losses):
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": steps,
                "train_loss_curve": losses}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dev", default="data/dev_eval.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--schedule", default="constant", choices=["constant", "cosine", "wsd"])
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie_weights", type=int, default=0)
    ap.add_argument("--pos_type", default="learned", choices=["learned", "rope"])
    ap.add_argument("--norm_type", default="layernorm", choices=["layernorm", "rmsnorm"])
    ap.add_argument("--mlp", default="gelu", choices=["gelu", "swiglu"])
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--init_std", type=float, default=0.05)
    ap.add_argument("--scale_residual", type=int, default=0)
    ap.add_argument("--bias", type=int, default=1)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    for k in ["block_size", "n_layer", "n_head", "n_embd", "dropout", "pos_type",
              "norm_type", "mlp", "mlp_ratio", "init_std"]:
        setattr(cfg, k, getattr(args, k))
    cfg.tie_weights = bool(args.tie_weights)
    cfg.scale_residual = bool(args.scale_residual)
    cfg.bias = bool(args.bias)

    model = GPT(cfg)
    n = model.n_params()
    print(f"model: {n:,} params  (vocab {cfg.vocab_size}, d {cfg.n_embd}, "
          f"L {cfg.n_layer}, block {cfg.block_size}, tie {cfg.tie_weights})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params (got {n:,})"

    opt = make_optimizer(model, args)
    dev_text = None
    if args.eval_every:
        try:
            dev_text = open(args.dev, encoding="utf-8").read()
        except OSError:
            dev_text = None

    model.train()
    t0 = time.time()
    losses = []
    best = None
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, args)
        x, y = get_batch(ids, cfg.block_size, args.batch, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr_at(step-1, args):.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")
        if dev_text is not None and step % args.eval_every == 0:
            b = dev_bpb(model, cfg, tok, dev_text)
            print(f"           dev bpb {b:.4f} @ step {step}")
            if best is None or b < best[0]:
                best = (b, {k: v.clone() for k, v in model.state_dict().items()}, step)

    if best is not None:
        final_b = dev_bpb(model, cfg, tok, dev_text)
        if final_b <= best[0]:
            save(args.out, model, cfg, args.steps, losses)
            print(f"saved {args.out}  dev bpb {final_b:.4f} (final, step {args.steps})")
        else:
            model.load_state_dict(best[1])
            save(args.out, model, cfg, best[2], losses)
            print(f"saved {args.out}  dev bpb {best[0]:.4f} (best, step {best[2]})")
    else:
        save(args.out, model, cfg, args.steps, losses)
        print(f"saved {args.out}  final_loss {sum(losses[-50:])/50:.4f}  "
              f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
