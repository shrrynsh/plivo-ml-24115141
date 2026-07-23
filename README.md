# plivo-ml-24115141 — 2000-step LLM speedrun

A small GPT trained from scratch on a mixed English+Hindi corpus, CPU-only, under
hard caps: ≤2000 optimizer steps and ≤2,000,000 parameters. Scored by bits-per-byte
on held-out text.

## Layout
- `model.py`, `tokenizer.py`, `train.py`, `evaluate.py` — the submission (what gets graded).
- `starter/` — the untouched handout, kept for reference.
- `data/` — the provided `train_corpus.txt` and `dev_eval.txt` (not committed).
- `RUNLOG.md` — every run: hypothesis, change, dev bpb before/after, conclusion.
- `NOTES.md` — the final config in ≤10 sentences.

## Reproduce
```
# 1. train the tokenizer on the corpus (writes tokenizer.json)
python tokenizer.py --data data/train_corpus.txt --vocab_size <V> --out tokenizer.json

# 2. train the model (see NOTES.md for the exact final flags)
python train.py --data data/train_corpus.txt --steps 2000 --out ckpt.pt <flags>

# 3. score
python evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
```
The evaluator command works unmodified from inside this folder. Seed is 1337; the
batch sampler is seeded, so a run reproduces given the same flags. The tokenizer is
lossless (`decode(encode(text)) == text`) with a raw-byte fallback for any UTF-8.
