import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
        detach_state_every: int = 256,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")

        self.dim = dim
        self.p = p
        self.q = q
        self.eps = eps
        self.expansion_factor = expansion_factor
        # Truncated BPTT: detach recurrence state every N steps to cap VRAM for long seq_len.
        # Set <= 0 to disable (full BPTT; can OOM for seq_len=4096).
        self.detach_state_every = int(detach_state_every)

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

        # Pre-allocate outputs to avoid Python list growth + stack allocation.
        y = x.new_empty((b, n, d))
        for t in range(n):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]  # (b, d)
            etat, alphat = eta[:, t], alpha[:, t]  # (b,1,1)

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
            yt = torch.bmm(qt.unsqueeze(1), W).squeeze(1)
            y[:, t] = yt

            # Truncated BPTT: prevent the autograd graph from growing with n.
            if self.detach_state_every > 0 and ((t + 1) % self.detach_state_every == 0) and (t + 1) < n:
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

