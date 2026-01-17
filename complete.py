from __future__ import annotations

import argparse
import random
from typing import Optional

import torch
from transformers import AutoTokenizer

from miras import ModularLlama, moneta_factory


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _resolve_dtype(device: torch.device, dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype != "auto":
        raise ValueError(f"Unknown dtype: {dtype}")

    # auto
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


@torch.inference_mode()
def generate(
    *,
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    top_k: int,
) -> str:
    if not isinstance(model, ModularLlama):
        raise TypeError("This generator expects `ModularLlama` (for fast incremental decoding).")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) == 0:
        raise ValueError("Prompt tokenized to empty sequence.")

    # Incremental decode: prefill state with the prompt, then sample one token at a time.
    ids: list[int] = list(map(int, prompt_ids))
    state = model.init_state(batch_size=1, device=device, dtype=next(model.parameters()).dtype, pos=0)

    logits = None
    for tid in ids:
        logits, state = model.forward_step(torch.tensor([tid], device=device, dtype=torch.long), state=state)

    assert logits is not None

    for _ in range(max_new_tokens):
        next_logits = logits[0]  # (V,)

        if temperature <= 0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                v, ix = torch.topk(next_logits, k=min(top_k, next_logits.numel()))
                probs = torch.softmax(v, dim=-1)
                next_id = int(ix[torch.multinomial(probs, 1)].item())
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

        ids.append(next_id)
        logits, state = model.forward_step(torch.tensor([next_id], device=device, dtype=torch.long), state=state)

    return tokenizer.decode(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text with a MONETA checkpoint.")
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint file (e.g. runs/<run>/checkpoints/latest.pt).",
    )
    ap.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="<= 0 for greedy decoding; > 0 for sampling.",
    )
    ap.add_argument("--top-k", type=int, default=50, help="Top-k sampling; 0 disables.")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmuls on CUDA.")
    ap.add_argument("--seed", type=int, default=1337, help="Sampling seed.")
    args = ap.parse_args()

    device = pick_device(args.device)
    seed_all(int(args.seed))

    if device.type == "cuda":
        tf32 = not bool(args.no_tf32)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})

    tokenizer_name = cfg.get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = 10**9
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = int(cfg.get("n_layers", 2))
    dim = int(cfg.get("dim", 128))
    max_seq_len = int(cfg.get("max_seq_len", 4096))

    prompt_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + int(args.max_new_tokens) > max_seq_len:
        raise ValueError(
            f"prompt_len({len(prompt_ids)}) + max_new_tokens({args.max_new_tokens}) "
            f"> max_seq_len({max_seq_len}). Re-train with a larger --max-seq-len or "
            f"shorten prompt / generation."
        )

    model = ModularLlama(
        vocab_size=int(tokenizer.vocab_size),
        n_layers=n_layers,
        dim=dim,
        max_seq_len=max_seq_len,
        block_factory=moneta_factory,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.grad_checkpoint = False  # inference: no recompute

    dtype = _resolve_dtype(device, str(args.dtype))
    model = model.to(dtype=dtype)

    out = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        device=device,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
    )
    print(out)


if __name__ == "__main__":
    main()

