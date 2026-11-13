"""A decoder-only Transformer with Llama-style blocks."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_384
    width: int = 768
    layers: int = 12
    query_heads: int = 12
    kv_heads: int = 4
    hidden_width: int = 3_328
    context: int = 256

    def __post_init__(self):
        if self.width % self.query_heads or self.query_heads % self.kv_heads:
            raise ValueError("Width and attention head counts must divide evenly.")
        if (self.width // self.query_heads) % 2:
            raise ValueError("RoPE requires an even head dimension.")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)
        return (values * self.weight).to(x.dtype)


def rotate(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_heads = config.query_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.width // config.query_heads
        kv_width = self.kv_heads * self.head_dim
        self.q = nn.Linear(config.width, config.width, bias=False)
        self.k = nn.Linear(config.width, kv_width, bias=False)
        self.v = nn.Linear(config.width, kv_width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, cos, sin):
        batch, length, _ = x.shape
        q = self.q(x).view(batch, length, self.query_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        y = F.scaled_dot_product_attention(
            rotate(q, cos, sin), rotate(k, cos, sin), v,
            is_causal=True, enable_gqa=self.query_heads != self.kv_heads,
        )
        return self.out(y.transpose(1, 2).reshape(batch, length, -1))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.gate = nn.Linear(config.width, config.hidden_width, bias=False)
        self.up = nn.Linear(config.width, config.hidden_width, bias=False)
        self.down = nn.Linear(config.hidden_width, config.width, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attn_norm(x), cos, sin)
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        head_dim = config.width // config.query_heads
        inv_freq = 10_000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        phase = torch.outer(torch.arange(config.context), inv_freq)
        self.register_buffer("rope_cos", phase.cos(), persistent=False)
        self.register_buffer("rope_sin", phase.sin(), persistent=False)
        self.apply(self._initialize)
        residual_std = 0.02 / (2 * config.layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attention.out.weight, std=residual_std)
            nn.init.normal_(block.down.weight, std=residual_std)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def features(self, tokens):
        length = tokens.size(1)
        if not 0 < length <= self.config.context:
            raise ValueError(f"Sequence length must be 1–{self.config.context}.")
        x = self.embedding(tokens)
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, tokens):
        return F.linear(self.features(tokens), self.embedding.weight)
