from __future__ import annotations

import itertools

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from miras import ModularLlama, moneta_factory


def get_tiny_batch(
    *,
    batch_size: int = 2,
    seq_len: int = 64,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    split: str = "train",
    tokenizer_name: str = "gpt2",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, AutoTokenizer]:
    """
    Streams a tiny patch from FineWeb-Edu and tokenizes it into (batch_size, seq_len).

    Returns:
      tokens: int64 tensor of shape (B, T)
      tokenizer: the tokenizer used (so caller can set vocab_size correctly)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Avoid padding issues for tokenizers without an explicit pad token.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    it = iter(ds)

    needed = batch_size * seq_len
    pieces: list[torch.Tensor] = []
    total = 0

    for ex in itertools.islice(it, 256):
        text = ex.get("text")
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if ids.numel() == 0:
            continue
        pieces.append(ids)
        total += ids.numel()
        if total >= needed:
            break

    if total < needed:
        raise RuntimeError(
            f"Could not collect enough tokens from {dataset_name}/{dataset_config} "
            f"to form a batch of {needed} tokens (only got {total})."
        )

    flat = torch.cat(pieces, dim=0)[:needed].to(dtype=torch.long)
    tokens = flat.view(batch_size, seq_len).to(device)
    return tokens, tokenizer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens, tokenizer = get_tiny_batch(device=device)

    model = ModularLlama(
        vocab_size=int(tokenizer.vocab_size),
        n_layers=2,
        dim=256,
        block_factory=moneta_factory,
        max_seq_len=max(tokens.shape[1], 64),
    ).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tokens.reshape(-1),
    )

    loss.backward()
    optimizer.step()
    print(f"Dry run successful. Loss: {loss.item()}")


if __name__ == "__main__":
    main()