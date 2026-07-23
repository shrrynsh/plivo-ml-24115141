# RUNLOG

Metric: bits-per-byte (bpb) on `data/dev_eval.txt`, scored with the unmodified
`python evaluate.py --checkpoint <ckpt> --text_file data/dev_eval.txt`. Lower is
better. Seed 1337 on every run. Caps (≤2000 steps, ≤2,000,000 params) are asserted
in `train.py`, so a run that violates either just crashes instead of producing a
checkpoint.

Corpus: 7,318,592 bytes, ~86% ASCII / ~14% Devanagari by character. Devanagari is
3 UTF-8 bytes/char, so on a raw-byte tokenizer Hindi eats a hugely disproportionate
share of both the context window and the token budget.

## Reading the baseline before touching anything

Things in the starter that look wrong / left on the table, roughly in order of how
much I expect them to matter:

1. **Byte tokenizer (vocab 256).** A Devanagari character costs 3 tokens, so the
   128-token window holds very little real Hindi text and bpb's per-token loss is
   spread over few bytes. This looks like the single biggest lever.
2. **Constant LR 3e-4, no warmup, no decay.** Almost certainly undertrains in 2000
   steps — worth watching whether the loss is still falling at the end.
3. **Plain Adam, no weight decay, no gradient clipping.**
4. **Batch 8.** ~0.8 epochs over the corpus in 2000 steps and noisy gradients.
   Steps are capped but tokens/step are not, so this is basically free to raise.
5. **Untied head** wastes `vocab·n_embd` params; **flat 0.05 init** for every
   weight; learned absolute position embeddings.

Plan: fix the tokenizer first (biggest suspected lever), then the optimizer, then
spend the freed compute/parameter budget on batch size and architecture — one
change at a time, reverting anything that doesn't help on dev.

## Runs

| run | change | dev bpb | Δ | verdict |
|-----|--------|--------:|---:|---------|
| R0  | baseline (byte, Adam const 3e-4, batch 8) | 2.3718 | — | reference |
| R1  | tokenizer → byte-level BPE vocab 1024 (else = R0) | 2.1368 | −0.235 | keep |
| R2a | optimizer → AdamW+warmup+cosine+clip+wd, lr 1e-3 | 2.0830 | −0.054 | — |
| R2b | same, lr 2e-3 | 2.0630 | −0.074 | keep |
| R2c | same, lr 3e-3 | 2.1298 | −0.007 | too hot @ batch 8 |
| R3a | batch 8 → 16 (lr 2e-3) | 1.8422 | −0.221 | keep |
| R3b | batch 16 → 32 (lr 2e-3) | 1.7355 | −0.328 | keep |

### R0 — baseline
- **Hypothesis:** establish the starting number before changing anything.
- **Config:** byte tokenizer (vocab 256); 4 layers, 4 heads, d=160, block 128;
  Adam, constant lr 3e-4; batch 8; 2000 steps; 1,339,840 params.
- **Result:** dev **bpb 2.3718**. Train loss ≈1.73 and still falling at step 2000
  — confirms undertraining, so the optimizer is going to matter once the tokenizer
  stops wasting the window.
- **Conclusion:** reference point. Attack the tokenizer next.

### R1 — byte-level BPE tokenizer
- **Hypothesis:** a BPE tokenizer trained on the corpus should pack multiple bytes
  per token (especially Devanagari consonant+matra runs), so the same 128-token
  window holds far more real text and bpb's byte denominator grows. Biggest
  suspected lever.
- **What changed (only this):** raw-byte → byte-level BPE, vocab 1024, with a
  Devanagari-aware pre-tokenizer (U+0900–U+097F kept in the word class, optional
  leading space) so Hindi syllables actually merge. 256 base byte tokens kept for a
  lossless byte fallback. Model + optimizer identical to R0.
- **Result:** dev **bpb 2.3718 → 2.1368** (−0.235). Corpus packs 2.81 bytes/token;
  dev drops from 159k tokens to 55k. 1,585,600 params.
- **Conclusion:** keep — biggest single drop, as expected. But train loss is still
  ~4.06 and falling steeply at step 2000: the 1024-vocab has much higher per-token
  entropy, and the constant-LR Adam badly undertrains it. The optimizer is clearly
  leaving a lot on the table — that's next.

### R2 — optimizer bundle + LR sweep
- **Hypothesis:** R1 is undertrained (loss ~4.06, still dropping). Swapping to AdamW
  with a linear warmup → cosine decay, decoupled weight decay on the matmuls, and
  gradient clipping should let a much higher peak LR actually converge inside 2000
  steps. Treated "the optimizer" as one lever and swept peak LR ∈ {1e-3, 2e-3, 3e-3}.
- **What changed:** Adam const 3e-4 → AdamW(0.9, 0.95), wd 0.1 (2D weights only),
  warmup 100, cosine to 1e-4, clip 1.0. Arch/tokenizer/batch unchanged from R1.
- **Result:** lr 1e-3 → 2.0830, **lr 2e-3 → 2.0630 (best)**, lr 3e-3 → 2.1298.
  dev **bpb 2.1368 → 2.0630** (−0.074) at lr 2e-3.
- **Conclusion:** keep AdamW+cosine at lr 2e-3. Clear interior optimum. The ambitious
  3e-3 actually *lost* (worse than even 1e-3): its loss curve sits persistently higher
  (final train loss 4.07 vs 3.86 for 2e-3), with no divergence spike — the signature of
  a step size too large for batch-8's gradient noise, so updates overshoot and never
  settle. That points somewhere specific: a **bigger batch** cuts gradient variance,
  which should make that same 3e-3 safe — batch size and LR are coupled, not
  independent. Test batch size next, then revisit 3e-3 at large batch.

### R3 — batch size, then the batch × LR interaction
- **Hypothesis:** the model is undertrained (R2 batch 8 is ~0.8 epochs). Steps are
  capped but tokens/step are not, so a bigger batch raises corpus coverage and cuts
  gradient variance for free (only wall-clock cost, which isn't graded). And per R2's
  diagnosis, a larger batch should let the 3e-3 that overshot at batch 8 converge —
  so after the batch sweep I retest 3e-3 at large batch.
- **What changed:** batch 8 → {16, 32, 48} at lr 2e-3; then a 3e-3 vs 2e-3 probe at
  batch 32 and 48. Everything else = R2 (AdamW cosine, BPE-1024, baseline arch).
- **Result:** at lr 2e-3, batch 16 → 1.8422, batch 32 → 1.7355 (train loss fell 3.86
  → 2.93, i.e. ~3 epochs of coverage vs 0.8). batch 48 and the 3e-3 probes: below.
- **Conclusion (batch):** batch size is the biggest single win after the tokenizer —
  keep pushing it. (batch×LR conclusion appended once b48 runs finish.)
