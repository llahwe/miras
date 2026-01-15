from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import itertools
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from miras import ModularLlama, moneta_factory


@dataclass(frozen=True)
class RunConfig:
    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    tokenizer_name: str = "gpt2"
    buffer_tokens: int = 2_000_000  # ~2M tokens fits easily on laptop (~8MB as int32)

    # Model
    dim: int = 128
    n_layers: int = 2
    # This must be >= seq_len. The MIRAS paper uses 4096.
    max_seq_len: int = 256

    # Training
    microbatch_size: int = 4
    seq_len: int = 128
    grad_accum_steps: int = 8  # effective batch = microbatch_size * grad_accum_steps
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 1337

    # Runtime
    device: str = "auto"  # auto|cpu|mps|cuda
    max_time_seconds: int = 3600
    log_every_steps: int = 20
    ckpt_every_seconds: int = 600  # ~10 minutes

    # Output
    runs_dir: str = "runs"
    run_name: str | None = None  # if None, auto timestamped


def _now_tag() -> str:
    # Example: 2026-01-15_13-04-59
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    # Prefer MPS on Mac, else CUDA, else CPU.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_or_build_token_buffer(
    *,
    cfg: RunConfig,
    run_dir: Path,
) -> tuple[torch.Tensor, Any]:
    """
    Best-practice choice for laptop runs: build a token buffer ONCE, save it to disk,
    then sample training batches from it.

    Why this is preferable to "true streaming" for your use case:
    - It decouples network/tokenization overhead from the training step time
    - It makes runs restartable and comparable (same fixed token reservoir if you keep the buffer)
    - It makes throughput measurements meaningful (model compute, not data I/O)
    """
    buffer_path = run_dir / "token_buffer.pt"
    meta_path = run_dir / "token_buffer_meta.json"

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    # We're not using a HF "model" here; we just tokenize text into ids.
    # Some tokenizers (e.g. GPT-2) default to model_max_length=1024 and warn on longer texts.
    tokenizer.model_max_length = 10**9
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if buffer_path.exists() and meta_path.exists():
        flat = torch.load(buffer_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("buffer_tokens", -1)) != int(flat.numel()):
            raise RuntimeError(
                f"Token buffer mismatch: meta says {meta.get('buffer_tokens')} "
                f"but tensor has {flat.numel()} tokens"
            )
        return flat, tokenizer

    ds = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
        streaming=True,
    )

    needed = int(cfg.buffer_tokens)
    pieces: list[torch.Tensor] = []
    total = 0
    started = time.time()

    # We stop once we've collected `buffer_tokens` ids.
    for ex in ds:
        text = ex.get("text")
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if ids.numel() == 0:
            continue
        pieces.append(ids)
        total += int(ids.numel())
        if total >= needed:
            break

    if total < needed:
        raise RuntimeError(f"Could only collect {total} tokens, need {needed}.")

    flat = torch.cat(pieces, dim=0)[:needed].to(dtype=torch.int32).contiguous()
    torch.save(flat, buffer_path)
    _write_json(
        meta_path,
        {
            "dataset_name": cfg.dataset_name,
            "dataset_config": cfg.dataset_config,
            "dataset_split": cfg.dataset_split,
            "tokenizer_name": cfg.tokenizer_name,
            "buffer_tokens": int(flat.numel()),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "build_seconds": time.time() - started,
        },
    )
    return flat, tokenizer


def _sample_batch(
    flat: torch.Tensor, *, microbatch_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples contiguous spans from the flat token buffer.

    Returns:
      x: (B, T) input ids
      y: (B, T) next-token targets aligned with x (shifted by 1 in the flat stream)
    """
    # Need T+1 tokens for next-token prediction.
    span = seq_len + 1
    max_start = flat.numel() - span - 1
    if max_start <= 0:
        raise ValueError("Token buffer is too small for requested seq_len.")

    starts = torch.randint(0, max_start, (microbatch_size,), generator=None)
    xs = []
    ys = []
    for s in starts.tolist():
        chunk = flat[s : s + span].to(dtype=torch.long)
        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    x = torch.stack(xs, dim=0).to(device, non_blocking=True)
    y = torch.stack(ys, dim=0).to(device, non_blocking=True)
    return x, y


def _checkpoint(
    *,
    ckpt_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: RunConfig,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"

    payload: dict[str, Any] = {
        "step": step,
        "config": dataclasses.asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    torch.save(payload, path)

    # Also update a stable pointer for "resume the latest".
    latest = ckpt_dir / "latest.pt"
    torch.save(payload, latest)
    return path


def _try_resume(
    resume_path: Path,
    *,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # RNG restoration makes "resume" closer to deterministic.
    if "rng" in ckpt:
        rng = ckpt["rng"]
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    model.to(device)
    return int(ckpt.get("step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop-sized MIRAS/MONETA training run.")
    parser.add_argument("--run-name", type=str, default=None, help="Run folder name under runs/.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt file.")
    parser.add_argument("--max-time-seconds", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None, help="Training context length (paper: 4096).")
    parser.add_argument("--max-seq-len", type=int, default=None, help="RoPE/cache precompute length (must be >= seq-len).")
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--buffer-tokens", type=int, default=None, help="How many tokens to cache locally.")
    args = parser.parse_args()

    cfg = RunConfig(run_name=args.run_name)
    if args.max_time_seconds is not None:
        cfg = dataclasses.replace(cfg, max_time_seconds=int(args.max_time_seconds))
    if args.seq_len is not None:
        cfg = dataclasses.replace(cfg, seq_len=int(args.seq_len))
    if args.max_seq_len is not None:
        cfg = dataclasses.replace(cfg, max_seq_len=int(args.max_seq_len))
    if args.microbatch_size is not None:
        cfg = dataclasses.replace(cfg, microbatch_size=int(args.microbatch_size))
    if args.grad_accum_steps is not None:
        cfg = dataclasses.replace(cfg, grad_accum_steps=int(args.grad_accum_steps))
    if args.buffer_tokens is not None:
        cfg = dataclasses.replace(cfg, buffer_tokens=int(args.buffer_tokens))

    if cfg.max_seq_len < cfg.seq_len:
        raise ValueError(f"max_seq_len must be >= seq_len, got {cfg.max_seq_len} < {cfg.seq_len}")

    device = _select_device(cfg.device)
    _seed_all(cfg.seed)

    run_name = cfg.run_name or _now_tag()
    run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Store the config at the start (best practice: immutable record of intent).
    _write_json(run_dir / "config.json", dataclasses.asdict(cfg) | {"selected_device": str(device)})

    # Build/load the token buffer once.
    flat, tokenizer = _load_or_build_token_buffer(cfg=cfg, run_dir=run_dir)

    model = ModularLlama(
        vocab_size=int(tokenizer.vocab_size),
        n_layers=cfg.n_layers,
        dim=cfg.dim,
        max_seq_len=max(cfg.max_seq_len, cfg.seq_len),
        block_factory=moneta_factory,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = _try_resume(Path(args.resume), device=device, model=model, optimizer=optimizer)

    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    model.train()
    t_start = time.time()
    t_last_ckpt = time.time()
    t_last_log = time.time()
    tokens_since_log = 0

    step = start_step
    while True:
        elapsed = time.time() - t_start
        if elapsed >= cfg.max_time_seconds:
            break

        # Gradient accumulation: do N micro-steps, then one optimizer step.
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = _sample_batch(
                flat, microbatch_size=cfg.microbatch_size, seq_len=cfg.seq_len, device=device
            )
            logits = model(x)  # (B, T, V)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            (loss / cfg.grad_accum_steps).backward()
            total_loss += float(loss.item())
            tokens_since_log += int(cfg.microbatch_size * cfg.seq_len)

        # Clip gradients (stability best-practice for longer runs).
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        step += 1

        # Periodic logging.
        if step % cfg.log_every_steps == 0:
            dt_log = max(1e-9, time.time() - t_last_log)
            toks_per_s = tokens_since_log / dt_log

            record = {
                "step": step,
                "elapsed_s": time.time() - t_start,
                "loss": total_loss / cfg.grad_accum_steps,
                "tokens_per_s": toks_per_s,
                "device": str(device),
                "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            print(
                f"[step {step}] loss={record['loss']:.4f} "
                f"tok/s={record['tokens_per_s']:.0f} grad_norm={record['grad_norm']:.3f}"
            )

            t_last_log = time.time()
            tokens_since_log = 0

        # Periodic checkpointing by wall-clock time (restart-friendly).
        if (time.time() - t_last_ckpt) >= cfg.ckpt_every_seconds:
            path = _checkpoint(
                ckpt_dir=ckpt_dir, model=model, optimizer=optimizer, step=step, cfg=cfg
            )
            print(f"Saved checkpoint: {path}")
            t_last_ckpt = time.time()

    # Final checkpoint at end.
    path = _checkpoint(ckpt_dir=ckpt_dir, model=model, optimizer=optimizer, step=step, cfg=cfg)
    print(f"Finished. Final checkpoint: {path}")


if __name__ == "__main__":
    main()

