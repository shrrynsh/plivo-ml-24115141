"""Byte-level BPE trained on the corpus. Keeps the starter interface:
load() -> object with .encode(str)->list[int], .decode(list[int])->str, .vocab_size.

The 256 raw bytes are always base tokens, so encoding is lossless for any
UTF-8 input and every merge is just a concatenation. The pre-tokenizer keeps
the Devanagari block in the "word" class so a consonant and its matra can
merge into one token instead of the three bytes a raw-byte tokenizer spends.
"""
import json
import os
import re

# word | non-word-non-space | whitespace, each with an optional leading space.
# The Devanagari range U+0900-U+097F is folded into the word class.
_PAT = re.compile(r" ?[0-9A-Za-z_ऀ-ॿ]+| ?[^\s0-9A-Za-z_ऀ-ॿ]+|\s+")
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(_HERE, "tokenizer.json")


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.new_id = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.id_to_bytes.append(self.id_to_bytes[a] + self.id_to_bytes[b])
        self.vocab_size = 256 + len(self.merges)
        self._cache = {}

    def _merge_word(self, bs):
        cached = self._cache.get(bs)
        if cached is not None:
            return cached
        word = list(bs)
        while len(word) >= 2:
            best, best_rank = -1, None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best < 0:
                break
            word[best:best + 2] = [self.new_id[(word[best], word[best + 1])]]
        self._cache[bs] = word
        return word

    def encode(self, text):
        out = []
        for piece in _PAT.findall(text):
            out.extend(self._merge_word(piece.encode("utf-8")))
        return out

    def decode(self, ids):
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8", "replace")

    def save(self, path=_DEFAULT):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": [list(m) for m in self.merges]}, f)


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", "replace")


def load(path=None):
    path = path or os.environ.get("TOKENIZER_JSON") or _DEFAULT
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d.get("type") == "bpe":
            return BPETokenizer(d["merges"])
    return ByteTokenizer()


def train_bpe(text, vocab_size):
    from collections import Counter, defaultdict

    freq = Counter(_PAT.findall(text))
    assert "".join(_PAT.findall(text)) == text, "pre-tokenization dropped characters"
    words = [list(w.encode("utf-8")) for w in freq]
    counts = list(freq.values())

    pairs = defaultdict(int)
    where = defaultdict(set)
    for i, w in enumerate(words):
        for p in zip(w, w[1:]):
            pairs[p] += counts[i]
            where[p].add(i)

    merges = []
    while 256 + len(merges) < vocab_size and pairs:
        best = max(pairs, key=pairs.get)
        if pairs[best] <= 0:
            break
        new_id = 256 + len(merges)
        a, b = best
        merges.append(best)
        for i in list(where[best]):
            w, c = words[i], counts[i]
            for p in zip(w, w[1:]):
                pairs[p] -= c
            nw, j = [], 0
            while j < len(w):
                if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                    nw.append(new_id)
                    j += 2
                else:
                    nw.append(w[j])
                    j += 1
            words[i] = nw
            for p in zip(nw, nw[1:]):
                pairs[p] += c
                where[p].add(i)
        pairs.pop(best, None)
        where.pop(best, None)
    return BPETokenizer(merges)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--out", default=_DEFAULT)
    args = ap.parse_args()
    text = open(args.data, encoding="utf-8").read()
    tok = train_bpe(text, args.vocab_size)
    tok.save(args.out)
    sample = text[:100000]
    assert tok.decode(tok.encode(sample)) == sample, "round-trip failed"
    nb, nt = len(text.encode("utf-8")), len(tok.encode(text))
    print(f"saved {args.out}  vocab {tok.vocab_size}  {nb / nt:.3f} bytes/token")
