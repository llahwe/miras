import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint


class MonetaBlock(nn.Module):
    """
    Minimal / research-sketch MONETA-like block.

    Key invariant for this project: the recurrence state lives in (dim, dim)
    (per batch element), so the block input/output stays in feature space `dim`.
    """

    def __init__(
        self,
        dim: int,
        p: int = 3,
        q: int = 4,
        expansion_factor: int = 4,  # kept for future work; not used in recurrence
        eps: float = 1e-6,
        # Backwards-compat: historically used for token-level detaching. In the refactor
        # we interpret this as a *default chunk size* when `chunk_size` is not provided,
        # matching typical configs where this was set to 256.
        detach_state_every: int = 256,
        *,
        chunk_size: Optional[int] = None,
        tbptt_horizon_chunks: int = 4,
        grad_checkpoint_inner: bool = True,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")

        self.dim = dim
        self.p = p
        self.q = q
        self.eps = eps
        self.expansion_factor = expansion_factor

        # Chunking + TBPTT knobs:
        # - Chunking is a forward-pass engineering trick: process the inner-loop update
        #   in chunks (e.g. 256 tokens) and optionally checkpoint the chunk function.
        # - TBPTT applies to the *outer loop*: detach the recurrence state every H chunks,
        #   giving gradient horizon of H * chunk_size tokens while still carrying the
        #   long-term memory state forward through the whole sequence.
        if chunk_size is None:
            chunk_size = int(detach_state_every)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got chunk_size={self.chunk_size}")

        self.tbptt_horizon_chunks = int(tbptt_horizon_chunks)
        self.grad_checkpoint_inner = bool(grad_checkpoint_inner)

        # 1) Linear projections (Llama-style bias=False)
        # Fuse Q/K/V projection into one GEMM (same math, fewer FLOPs / kernel launches).
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)

        # 2) Depthwise convs (kernel=4) over sequence length
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)

        # 3) Data-dependent recurrence parameters (scalar per token)
        self.param_gen = nn.Linear(dim, 2, bias=True)  # -> (eta, alpha) logits

        # Initial recurrence matrix (shared across batch), used at t=0.
        self.W0 = nn.Parameter(torch.empty(dim, dim))
        nn.init.eye_(self.W0)

        # 4) Output gating
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

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
        Backwards-compatible loading from older checkpoints that had `q_proj/k_proj/v_proj`.
        """
        qkv_key = prefix + "qkv_proj.weight"
        q_key = prefix + "q_proj.weight"
        k_key = prefix + "k_proj.weight"
        v_key = prefix + "v_proj.weight"

        if qkv_key not in state_dict and q_key in state_dict and k_key in state_dict and v_key in state_dict:
            state_dict[qkv_key] = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)
            state_dict.pop(q_key)
            state_dict.pop(k_key)
            state_dict.pop(v_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        x: (b, n, dim) real
        freqs_cis: (n, dim/2) complex (as produced by `ModularLlama`)
        """
        # `view_as_complex` requires the last dimension to be contiguous (stride 1).
        # q/k/v here often come from transpose/conv slices and are not contiguous.
        x_ri = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
        x_complex = torch.view_as_complex(x_ri)
        freqs_cis = freqs_cis.view(1, x.shape[1], -1)
        x_rotated = x_complex * freqs_cis
        return torch.view_as_real(x_rotated).flatten(2).type_as(x)

    def _run_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        A: torch.Tensor,
        W: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the MONETA inner-loop update over a chunk.

        Args:
            q/k/v: (b, c, d)
            eta/alpha: (b, c, 1, 1)
            A/W: (b, d, d) recurrence state entering the chunk

        Returns:
            y: (b, c, d) outputs for the chunk
            A/W: recurrence state after processing the chunk
        """
        b, c, d = q.shape
        y = q.new_empty((b, c, d))

        for t in range(c):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]  # (b, d)
            etat, alphat = eta[:, t], alpha[:, t]  # (b, 1, 1)

            # Predict v_t from k_t using current W: (b,d) @ (b,d,d) -> (b,d)
            pred = torch.bmm(kt.unsqueeze(1), W).squeeze(1)
            diff = pred - vt

            # Smooth Lp gradient wrt pred (elementwise), shape (b, d)
            # Fast path for default p=3: abs(diff)^(p-1) == diff^2 (same math, fewer ops).
            if self.p == 3:
                grad_pred = 3 * (torch.tanh(10 * diff) * (diff * diff))
            else:
                grad_pred = self.p * (torch.tanh(10 * diff) * torch.abs(diff).pow(self.p - 1))

            # Convert to gradient wrt W: outer(k_t, grad_pred) -> (b, d, d)
            grad_W = kt.unsqueeze(2) * grad_pred.unsqueeze(1)

            # A_t = alpha * A_{t-1} - eta * grad_W
            A = alphat * A - etat * grad_W

            # W_t = A_t / ||A_t||_q^{q-2}
            # Fast path for default q=4:
            #   ||A||_4^{(4-2)} == ||A||_4^2 == sqrt(sum_ij |A_ij|^4)
            if self.q == 4:
                a4 = A.square().square()
                denom = torch.sqrt(a4.sum(dim=(-2, -1), keepdim=True)) + self.eps
                W = A / denom
            else:
                norm_q = torch.linalg.vector_norm(A, ord=self.q, dim=(-2, -1), keepdim=True)
                W = A / (norm_q.pow(self.q - 2) + self.eps)

            # Output y_t = q_t W_t  -> (b, d)
            y[:, t] = torch.bmm(qt.unsqueeze(1), W).squeeze(1)

        return y, A, W

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        assert d == self.dim

        # Projections -> depthwise convs (slice back to length n)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_conv(q.transpose(1, 2))[:, :, :n].transpose(1, 2)
        k = self.k_conv(k.transpose(1, 2))[:, :, :n].transpose(1, 2)
        v = self.v_conv(v.transpose(1, 2))[:, :, :n].transpose(1, 2)

        # RoPE + L2 norm (as in your earlier sketch)
        q = F.normalize(self.apply_rope(q, freqs_cis), p=2, dim=-1)
        k = F.normalize(self.apply_rope(k, freqs_cis), p=2, dim=-1)

        # Data-dependent eta/alpha (scalar per token)
        params = self.param_gen(x)  # (b, n, 2)
        eta, alpha = torch.chunk(params, 2, dim=-1)
        eta = torch.sigmoid(eta).view(b, n, 1, 1)
        alpha = torch.sigmoid(alpha).view(b, n, 1, 1)

        # Recurrence state: (b, dim, dim)
        A = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        W = self.W0.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(b, -1, -1)

        # Outputs for entire sequence
        y = x.new_empty((b, n, d))

        # Process inner-loop update in forward chunks.
        # TBPTT applies at chunk boundaries (outer loop): detach state every H chunks.
        chunk_size = min(self.chunk_size, n) if n > 0 else self.chunk_size
        do_ckpt = self.grad_checkpoint_inner and self.training and torch.is_grad_enabled()

        chunk_idx = 0
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)

            q_c = q[:, start:end]
            k_c = k[:, start:end]
            v_c = v[:, start:end]
            eta_c = eta[:, start:end]
            alpha_c = alpha[:, start:end]

            if do_ckpt:
                y_c, A, W = checkpoint(
                    lambda _q, _k, _v, _eta, _alpha, _A, _W: self._run_chunk(_q, _k, _v, _eta, _alpha, _A, _W),
                    q_c,
                    k_c,
                    v_c,
                    eta_c,
                    alpha_c,
                    A,
                    W,
                    use_reentrant=False,
                )
            else:
                y_c, A, W = self._run_chunk(q_c, k_c, v_c, eta_c, alpha_c, A, W)

            y[:, start:end] = y_c

            chunk_idx += 1
            if self.tbptt_horizon_chunks > 0 and (chunk_idx % self.tbptt_horizon_chunks == 0) and end < n:
                # Truncated BPTT: cut gradient through the recurrence state every H chunks.
                A = A.detach()
                W = W.detach()
        return self.out_proj(y) * torch.sigmoid(self.gate_proj(x))


def moneta_factory(dim: int, **kwargs) -> MonetaBlock:
    """Factory matching `ModularLlama(block_factory=...)` signature."""
    return MonetaBlock(dim=dim, **kwargs)


def build_moneta_llama(
    vocab_size: int,
    n_layers: int,
    dim: int,
    *,
    max_seq_len: int = 4096,
    hidden_dim: Optional[int] = None,
    **moneta_kwargs,
):
    """Convenience constructor kept to match package exports."""
    from .llama_macro import ModularLlama

    return ModularLlama(
        vocab_size=vocab_size,
        n_layers=n_layers,
        dim=dim,
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        block_factory=lambda d: MonetaBlock(dim=d, **moneta_kwargs),
    )

