import torch
import torch.nn as nn
from typing import Callable, Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU implementation: (Swish(W1x) * W3x) * W2
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class LlamaMIRASLayer(nn.Module):
    def __init__(self, dim: int, sequence_block: nn.Module, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.sequence_block = sequence_block
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # 1) Sequence modeling (MIRAS block like MONETA)
        # We pass the RoPE frequencies (freqs_cis) into the block
        h = x + self.sequence_block(self.attention_norm(x), freqs_cis=freqs_cis)
        # 2) Feed-forward network (SwiGLU)
        out = h + self.mlp(self.ffn_norm(h))
        return out

class ModularLlama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        dim: int,
        block_factory: Callable[[int], nn.Module],
        max_seq_len: int = 4096,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Precompute RoPE frequencies (freqs_cis)
        freqs_cis = self._precompute_freqs_cis(dim, max_seq_len)
        # Buffer so it moves with `.to(device)` and can be excluded from state_dict if desired.
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.layers = nn.ModuleList(
            [LlamaMIRASLayer(dim, block_factory(dim), hidden_dim or 4 * dim) 
             for _ in range(n_layers)]
        )

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        # Mathematical RoPE precomputation
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs) # complex64

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Slice precomputed RoPE to current sequence length
        freqs_cis = self.freqs_cis[:seq_len]
        
        for layer in self.layers:
            h = layer(h, freqs_cis=freqs_cis)
            
        return self.output(self.norm(h))