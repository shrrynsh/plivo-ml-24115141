"""Small GPT, config-driven so evaluate.py can rebuild any variant from the
fields saved in the checkpoint. New knobs default to the starter's behaviour,
so flipping one at a time isolates its effect.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = False
    pos_type = "learned"     # learned | rope
    norm_type = "layernorm"  # layernorm | rmsnorm
    mlp = "gelu"             # gelu | swiglu
    mlp_ratio = 4.0
    init_std = 0.05
    scale_residual = False
    bias = True


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if cfg.norm_type == "rmsnorm" else nn.LayerNorm(cfg.n_embd)


def rope_cache(T, hd, base, dtype):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(T).float(), inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    T = x.shape[-2]
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.mlp
        h = int(cfg.mlp_ratio * cfg.n_embd)
        if cfg.mlp == "swiglu":
            self.w1 = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
            self.w3 = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
            self.w2 = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        else:
            self.fc = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
            self.proj = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_rope = cfg.pos_type == "rope"
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        if cfg.scale_residual:
            s = 1.0 / math.sqrt(2 * cfg.n_layer)
            for name, p in self.named_parameters():
                if name.endswith(("proj.weight", "w2.weight")):
                    with torch.no_grad():
                        p.mul_(s)
        self._rope = None

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        rope = None
        if self.use_rope:
            if self._rope is None or self._rope[0].shape[0] < T:
                self._rope = rope_cache(max(T, self.cfg.block_size),
                                        self.cfg.n_embd // self.cfg.n_head, 10000.0, x.dtype)
            rope = self._rope
        else:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))[None]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, rope)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
