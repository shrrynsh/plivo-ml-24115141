# NOTES — best configuration

1. The final model is a Devanagari-aware byte-level BPE tokenizer (vocab 2048, ~3.36 bytes/token, lossless with a raw-byte fallback) feeding a 5-layer, d=160, 4-head GPT with RoPE, RMSNorm, SwiGLU, bias-free linears, scaled residual init and a tied input/output embedding — 1,863,840 params, under the 2M cap, dev bpb 1.6722 (2000 steps).
2. The tokenizer is the single biggest lever: per-byte scoring rewards fewer tokens/byte, and a raw-byte tokenizer otherwise spends three tokens on every Devanagari character.
3. The optimizer is second — AdamW with a linear warmup → cosine decay, weight decay on matmuls only, and gradient clipping let a ~10× higher peak LR (3e-3) converge inside the 2000-step cap.
4. Batch size is third: steps are capped but tokens/step are not, so batch 48 turns ~0.8 epochs into ~3–4 and cuts gradient variance for free.
5. Peak LR and batch size are coupled — 3e-3 was the *worst* setting at batch 8 and the *best* at batch 48 — so they must be tuned together, not independently.
6. Weight tying pays for the upgrades: it frees vocab·d parameters, which is what lets vocab 2048 coexist with a 5th layer and the modern block under the cap.
7. The LLaMA-era block (RoPE/RMSNorm/SwiGLU/bias-free) is only worth about −0.024 bpb here; it helps mainly because the batch is now large enough to train it in 2000 steps.
8. I avoided changes that just add parameters without training faster, since with a fixed 2000-step budget what matters is how fast the model converges, not how big it is.
9. Progression: 2.3718 (baseline) → 2.1368 (BPE) → 2.0630 (AdamW/cosine) → 1.7355 (batch 32) → 1.6959 (batch 48, lr 3e-3) → 1.6722 (LLaMA-lite), a 29.5% reduction over baseline.
10. Reproducible: seed 1337, seeded batch sampler, committed tokenizer.json and checkpoint, lossless round-trip verified by the scorer.
