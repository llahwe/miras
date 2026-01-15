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
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    for _ in range(max_new_tokens):
        logits = model(ids)  # (1, T, V)
        next_logits = logits[0, -1]  # (V,)

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

        ids = torch.cat(
            [ids, torch.tensor([[next_id]], device=device, dtype=ids.dtype)],
            dim=1,
        )

    return tokenizer.decode(ids[0].tolist())


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
    ap.add_argument("--seed", type=int, default=1337, help="Sampling seed.")
    args = ap.parse_args()

    device = pick_device(args.device)
    seed_all(int(args.seed))

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

