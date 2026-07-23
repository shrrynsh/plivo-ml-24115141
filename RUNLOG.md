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
