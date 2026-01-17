import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
from torch.utils.checkpoint import checkpoint

from .moneta import MonetaState

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
        # Fused SwiGLU input projection: one GEMM instead of two.
        # Equivalent to separate w1/w3 with:
        #   w13.weight = cat([w1.weight, w3.weight], dim=0)
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU implementation: (Swish(W1x) * W3x) * W2
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(nn.functional.silu(x1) * x3)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Backwards-compatible loading from older checkpoints that had `w1` and `w3`.
        """
        w13_key = prefix + "w13.weight"
        w1_key = prefix + "w1.weight"
        w3_key = prefix + "w3.weight"

        if w13_key not in state_dict and w1_key in state_dict and w3_key in state_dict:
            # Build fused weight and remove legacy keys to avoid "unexpected key" warnings.
            state_dict[w13_key] = torch.cat([state_dict[w1_key], state_dict[w3_key]], dim=0)
            state_dict.pop(w1_key)
            state_dict.pop(w3_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

    def forward_step(
        self,
        x_t: torch.Tensor,  # (b, d)
        *,
        freqs_cis_t: torch.Tensor,  # (1, d/2) complex
        state: MonetaState,
    ) -> Tuple[torch.Tensor, MonetaState]:
        """
        Incremental (1-token) forward for fast generation.
        """
        # 1) Sequence modeling
        seq_in = self.attention_norm(x_t)
        seq_out, new_state = self.sequence_block.forward_step(seq_in, freqs_cis_t=freqs_cis_t, state=state)
        h = x_t + seq_out
        # 2) MLP
        out = h + self.mlp(self.ffn_norm(h))
        return out, new_state


@dataclass
class ModularLlamaState:
    pos: int
    layer_states: list[MonetaState]

class ModularLlama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        dim: int,
        block_factory: Callable[[int], nn.Module],
        max_seq_len: int = 4096,
        hidden_dim: Optional[int] = None,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")

        self.grad_checkpoint = bool(grad_checkpoint)
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
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                # Optional *outer* checkpointing across layers.
                #
                # Note: MIRAS/MONETA-style blocks may also checkpoint their inner-loop
                # memory update per chunk. Keeping this layer-level checkpoint as a
                # separate knob lets you trade extra recompute for lower activation
                # memory across the full transformer stack.
                h = checkpoint(lambda _h: layer(_h, freqs_cis=freqs_cis), h, use_reentrant=False)
            else:
                h = layer(h, freqs_cis=freqs_cis)
            
        return self.output(self.norm(h))

    def init_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        pos: int = 0,
    ) -> ModularLlamaState:
        layer_states: list[MonetaState] = []
        for layer in self.layers:
            # The sequence_block must expose init_state for fast generation.
            layer_states.append(layer.sequence_block.init_state(batch_size=batch_size, device=device, dtype=dtype))
        return ModularLlamaState(pos=int(pos), layer_states=layer_states)

    def forward_step(
        self,
        token_t: torch.Tensor,  # (b,)
        *,
        state: ModularLlamaState,
    ) -> Tuple[torch.Tensor, ModularLlamaState]:
        """
        Incremental (1-token) forward: returns logits for this token position.
        """
        if token_t.dim() != 1:
            raise ValueError(f"token_t must have shape (b,), got {tuple(token_t.shape)}")

        b = int(token_t.shape[0])
        h = self.tok_embeddings(token_t)  # (b, d)
        pos = int(state.pos)
        freqs_cis_t = self.freqs_cis[pos : pos + 1]  # (1, d/2) complex

        new_layer_states: list[MonetaState] = []
        for layer, layer_state in zip(self.layers, state.layer_states):
            h, new_s = layer.forward_step(h, freqs_cis_t=freqs_cis_t, state=layer_state)
            new_layer_states.append(new_s)

        logits = self.output(self.norm(h))  # (b, vocab)
        return logits, ModularLlamaState(pos=pos + 1, layer_states=new_layer_states)