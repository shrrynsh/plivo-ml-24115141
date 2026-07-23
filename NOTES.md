# NOTES — best configuration

1. The best completed configuration pairs a Devanagari-aware byte-level BPE tokenizer (vocab 1024, ~2.81 bytes/token, lossless with a raw-byte fallback) with the starter GPT (4 layers, d=160, 4 heads, learned positions, LayerNorm, GELU) — 1,585,600 params, under the 2M cap.
2. The tokenizer is the single biggest lever: per-byte scoring rewards fewer tokens/byte, and a raw-byte tokenizer otherwise spends three tokens on every Devanagari character.
3. The optimizer is the second lever — AdamW with a linear warmup → cosine decay, weight decay on matmuls only, and gradient clipping let a ~10× higher peak LR (3e-3) converge inside the 2000-step cap.
4. Batch size is the third: steps are capped but tokens/step are not, so batch 48 turns ~0.8 epochs into ~5 and cuts gradient variance for free.
5. Peak LR and batch size are coupled — 3e-3 was the *worst* setting at batch 8 and the *best* at batch 48 — so they must be tuned together.
6. Final dev bpb is 1.6959 versus the 2.3718 byte-tokenizer baseline (28.5% lower), at exactly 2000 steps.
7. Architecture experiments — weight tying + vocab 2048 + a 5th layer (R4), and a RoPE/RMSNorm/SwiGLU/bias-free block (R5) — were still running at the deadline; see RUNLOG for their hypotheses and status.
8. The batch-tuned baseline above is the completed, verified, reproducible configuration submitted here.
9. Reproducible: seed 1337, seeded batch sampler, committed tokenizer.json and checkpoint, lossless round-trip verified by the scorer.
