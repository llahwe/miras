from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from miras import ModularLlama, moneta_factory


ModelSize = Literal["s", "m", "l"]
DeviceChoice = Literal["auto", "cpu", "mps", "cuda"]
PrecisionChoice = Literal["auto", "fp32", "bf16", "fp16"]
DataSource = Literal["random", "hf_buffer"]


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(choice: DeviceChoice) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    # Prefer MPS on Mac, else CUDA, else CPU.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _model_hparams_from_size(model_size: ModelSize) -> tuple[int, int, int]:
    """
    Returns (dim, n_layers, n_heads) per MIRAS paper table.
    Note: n_heads is recorded for bookkeeping; the current ModularLlama/MonetaBlock
    implementation is not multi-head attention.
    """
    if model_size == "s":
        return 768, 12, 16
    if model_size == "m":
        return 1024, 24, 16
    if model_size == "l":
        return 1536, 24, 16
    raise ValueError(f"Unknown model_size: {model_size}")


def _resolve_precision(device: torch.device, precision: PrecisionChoice) -> tuple[torch.dtype, bool]:
    """
    Returns (amp_dtype, use_grad_scaler).
    - For bf16: no grad scaler.
    - For fp16: use grad scaler on CUDA.
    """
    if precision == "fp32":
        return torch.float32, False

    if device.type == "cuda":
        if precision == "auto":
            # Prefer bf16 when available; else fp16.
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, False
            return torch.float16, True
        if precision == "bf16":
            return torch.bfloat16, False
        if precision == "fp16":
            return torch.float16, True

    # MPS/CPU: keep it simple and stable.
    if precision in ("auto", "bf16", "fp16"):
        return torch.float32, False

    raise ValueError(f"Unknown precision: {precision}")


@dataclass(frozen=True)
class TrainConfig:
    # Run
    run_name: str | None = None
    runs_dir: str = "runs"
    seed: int = 1337

    # Model (paper sizes)
    model_size: ModelSize = "s"
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 16  # recorded for paper parity; not used by current model
    hidden_dim: int | None = None  # defaults to 4*dim in ModularLlama if None

    # MONETA memory control
    # Detach recurrence state every N steps (truncated BPTT). <=0 disables (can OOM for long seq_len).
    moneta_detach_state_every: int = 256

    # Sequence
    seq_len: int = 4096
    max_seq_len: int = 4096  # RoPE precompute length; must be >= seq_len

    # Data
    data_source: DataSource = "hf_buffer"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    tokenizer_name: str = "gpt2"
    buffer_tokens: int = 2_000_000

    # Training
    microbatch_size: int = 1
    grad_accum_steps: int = 16  # effective batch = microbatch_size * grad_accum_steps
    global_batch_size: int | None = None  # if set, overrides grad_accum_steps
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    # Runtime
    device: DeviceChoice = "auto"
    precision: PrecisionChoice = "auto"
    compile: bool = False
    grad_checkpoint: bool = True
    tf32: bool = True  # CUDA only
    max_steps: int = 0  # 0 means "no step limit"
    max_time_seconds: int = 3600
    log_every_steps: int = 20
    # Vast-friendly defaults:
    ckpt_every_seconds: int = 300
    ckpt_milestone_every_seconds: int = 3600
    ckpt_keep_last: int = 12
    ckpt_keep_milestones: int = 24
    ckpt_prune: bool = True

    # Logging
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: str | None = None  # comma-separated
    wandb_mode: str | None = None  # online|offline|disabled

    # Sync (Google Drive via rclone)
    gdrive_sync: bool = False
    gdrive_remote: str = "gdrive"
    # Remote base directory under Google Drive (rclone remote path).
    # If you create nested folders like research/papers/miras in Drive, set this accordingly.
    gdrive_dir: str = "research/papers/miras/runs"

    # Dry run
    dry_run: bool = False
    dry_run_vocab_size: int = 50257  # GPT-2 vocab size
    dry_run_steps: int = 2
    dry_run_seq_len: int = 128
    dry_run_microbatch_size: int = 2


def _build_or_load_token_buffer(*, cfg: TrainConfig, run_dir: Path) -> tuple[torch.Tensor, Any]:
    """
    Builds a flat token buffer ONCE, saves it, and samples training batches from it.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    buffer_path = run_dir / "token_buffer.pt"
    meta_path = run_dir / "token_buffer_meta.json"

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
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


def _sample_batch_from_flat(
    flat: torch.Tensor, *, microbatch_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized sampling of contiguous spans from a flat token buffer.

    Returns:
      x: (B, T) input ids
      y: (B, T) next-token targets
    """
    span = seq_len + 1
    max_start = flat.numel() - span - 1
    if max_start <= 0:
        raise ValueError("Token buffer is too small for requested seq_len.")

    starts = torch.randint(0, max_start, (microbatch_size,), device="cpu")
    offsets = torch.arange(span, device="cpu").view(1, span)
    idx = starts.view(microbatch_size, 1) + offsets  # (B, span)
    chunk = flat[idx].to(dtype=torch.long)  # (B, span)

    x = chunk[:, :-1].to(device, non_blocking=True)
    y = chunk[:, 1:].to(device, non_blocking=True)
    return x, y


def _make_random_batch(
    *, microbatch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (microbatch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (microbatch_size, seq_len), device=device, dtype=torch.long)
    return x, y


def _checkpoint(
    *,
    ckpt_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    cfg: TrainConfig,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"

    payload: dict[str, Any] = {
        "step": step,
        "config": dataclasses.asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    torch.save(payload, path)
    torch.save(payload, ckpt_dir / "latest.pt")
    return path


def _try_get_git_metadata() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parent
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root).decode()
        dirty = bool(status.strip())
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root).decode().strip()
        return {"git_commit": commit, "git_dirty": dirty, "git_branch": branch}
    except Exception:
        return {}


def _maybe_write_milestone_marker(*, ckpt_dir: Path, step: int, ckpt_path: Path) -> Path:
    """
    Write a small marker file identifying a checkpoint to keep as a "milestone".
    This avoids duplicating multi-GB .pt files.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    marker = ckpt_dir / f"milestone_step_{step:08d}.json"
    payload = {
        "step": int(step),
        "checkpoint": ckpt_path.name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker


def _prune_checkpoints(*, ckpt_dir: Path, keep_last: int, keep_milestones: int) -> None:
    """
    Prune old step checkpoints while preserving:
    - latest.pt
    - last `keep_last` step_*.pt
    - any step_*.pt referenced by the newest `keep_milestones` milestone markers
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step_re = re.compile(r"^step_(\d{8})\.pt$")
    steps: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = step_re.match(p.name)
        if m:
            steps.append((int(m.group(1)), p))
    steps.sort(key=lambda t: t[0])

    marker_re = re.compile(r"^milestone_step_(\d{8})\.json$")
    markers: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("milestone_step_*.json"):
        m = marker_re.match(p.name)
        if m:
            markers.append((int(m.group(1)), p))
    markers.sort(key=lambda t: t[0], reverse=True)

    # Keep only newest N markers; delete older markers.
    keep_marker_paths = {p for _s, p in markers[:keep_milestones]}
    for _s, p in markers[keep_milestones:]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    preserve_steps: set[int] = set()
    for s, _p in steps[-keep_last:]:
        preserve_steps.add(s)

    for p in keep_marker_paths:
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            preserve_steps.add(int(meta["step"]))
        except Exception:
            continue

    for s, p in steps:
        if s in preserve_steps:
            continue
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def _rclone_copyto(src: Path, dst: str) -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone not found on PATH; install it or disable --gdrive-sync.")
    subprocess.run(["rclone", "copyto", str(src), dst], check=True)


def _maybe_gdrive_sync_run(*, cfg: TrainConfig, run_dir: Path, ckpt_path: Path | None) -> None:
    if not cfg.gdrive_sync:
        return
    remote = cfg.gdrive_remote.rstrip(":")
    base = cfg.gdrive_dir.strip("/").rstrip("/")
    remote_run = f"{remote}:{base}/{run_dir.name}"

    # Small metadata files.
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if config_path.exists():
        _rclone_copyto(config_path, f"{remote_run}/config.json")
    if metrics_path.exists():
        _rclone_copyto(metrics_path, f"{remote_run}/metrics.jsonl")

    if ckpt_path is not None and ckpt_path.exists():
        _rclone_copyto(ckpt_path, f"{remote_run}/checkpoints/{ckpt_path.name}")
        latest = ckpt_path.parent / "latest.pt"
        if latest.exists():
            _rclone_copyto(latest, f"{remote_run}/checkpoints/latest.pt")


def _try_resume(
    resume_path: Path,
    *,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> int:
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified MIRAS/MONETA trainer (laptop + Vast.ai).")

    # Common
    p.add_argument("--dry-run", action="store_true", help="Run a short offline sanity pass and exit.")
    p.add_argument("--model-size", type=str, default="s", choices=["s", "m", "l"])
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (PyTorch 2.x).")
    p.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable activation checkpointing per layer (higher VRAM; slightly faster).",
    )
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmuls on CUDA.")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags.")
    p.add_argument("--wandb-mode", type=str, default=None, help="online|offline|disabled")

    # Sequence/training
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help=(
            "ELIF: this is the *effective* batch size (how many sequences contribute to one optimizer step). "
            "We achieve it via gradient accumulation: global_batch_size = microbatch_size * grad_accum_steps. "
            "If set, this overrides --grad-accum-steps."
        ),
    )
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)

    # MONETA memory control
    p.add_argument(
        "--moneta-detach-state-every",
        type=int,
        default=256,
        help="Detach MONETA recurrence state every N steps (truncated BPTT). <=0 disables (higher VRAM).",
    )

    # Runtime control
    p.add_argument("--max-steps", type=int, default=0, help="0 means no step limit.")
    p.add_argument("--max-time-seconds", type=int, default=3600)
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--ckpt-every-seconds", type=int, default=300)
    p.add_argument("--ckpt-milestone-every-seconds", type=int, default=3600)
    p.add_argument("--ckpt-keep-last", type=int, default=12)
    p.add_argument("--ckpt-keep-milestones", type=int, default=24)
    p.add_argument("--no-ckpt-prune", action="store_true", help="Disable pruning old checkpoints.")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt file.")

    # Data
    p.add_argument("--data-source", type=str, default="hf_buffer", choices=["random", "hf_buffer"])
    p.add_argument("--dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", type=str, default="sample-10BT")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--tokenizer-name", type=str, default="gpt2")
    p.add_argument("--buffer-tokens", type=int, default=2_000_000)

    # Sync (Google Drive via rclone)
    p.add_argument("--gdrive-sync", action="store_true", help="Sync checkpoints + metadata to Google Drive via rclone.")
    p.add_argument("--gdrive-remote", type=str, default="gdrive", help="rclone remote name (default: gdrive).")
    p.add_argument(
        "--gdrive-dir",
        type=str,
        default="research/papers/miras/runs",
        help="Remote base directory under the drive (default: research/papers/miras/runs).",
    )

    # Dry-run knobs (offline)
    p.add_argument(
        "--dry-run-full-model",
        action="store_true",
        help="Use the selected --model-size config during dry-run (can OOM on laptops).",
    )
    p.add_argument("--dry-run-dim", type=int, default=128, help="Dry-run model dimension (default: 128).")
    p.add_argument("--dry-run-n-layers", type=int, default=2, help="Dry-run number of layers (default: 2).")
    p.add_argument("--dry-run-steps", type=int, default=2)
    p.add_argument("--dry-run-seq-len", type=int, default=128)
    p.add_argument("--dry-run-microbatch-size", type=int, default=2)
    p.add_argument("--dry-run-vocab-size", type=int, default=50257)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dim, n_layers, n_heads = _model_hparams_from_size(args.model_size)
    cfg = TrainConfig(
        dry_run=bool(args.dry_run),
        model_size=args.model_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        moneta_detach_state_every=int(args.moneta_detach_state_every),
        run_name=args.run_name,
        runs_dir=args.runs_dir,
        seed=int(args.seed),
        device=args.device,
        precision=args.precision,
        compile=bool(args.compile),
        grad_checkpoint=not bool(args.no_grad_checkpoint),
        tf32=not bool(args.no_tf32),
        seq_len=int(args.seq_len),
        max_seq_len=int(args.max_seq_len),
        microbatch_size=int(args.microbatch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        global_batch_size=(int(args.global_batch_size) if args.global_batch_size is not None else None),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        max_steps=int(args.max_steps),
        max_time_seconds=int(args.max_time_seconds),
        log_every_steps=int(args.log_every_steps),
        ckpt_every_seconds=int(args.ckpt_every_seconds),
        ckpt_milestone_every_seconds=int(args.ckpt_milestone_every_seconds),
        ckpt_keep_last=int(args.ckpt_keep_last),
        ckpt_keep_milestones=int(args.ckpt_keep_milestones),
        ckpt_prune=not bool(args.no_ckpt_prune),
        wandb=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
        gdrive_sync=bool(args.gdrive_sync),
        gdrive_remote=args.gdrive_remote,
        gdrive_dir=args.gdrive_dir,
        data_source=args.data_source,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        tokenizer_name=args.tokenizer_name,
        buffer_tokens=int(args.buffer_tokens),
        dry_run_steps=int(args.dry_run_steps),
        dry_run_seq_len=int(args.dry_run_seq_len),
        dry_run_microbatch_size=int(args.dry_run_microbatch_size),
        dry_run_vocab_size=int(args.dry_run_vocab_size),
    )

    if cfg.max_seq_len < cfg.seq_len and not cfg.dry_run:
        raise ValueError(f"max_seq_len must be >= seq_len, got {cfg.max_seq_len} < {cfg.seq_len}")

    device = _select_device(cfg.device)
    _seed_all(cfg.seed)

    # Dry-run should be a fast, low-memory sanity check by default.
    # Use --dry-run-full-model if you explicitly want the full S/M/L config.
    if cfg.dry_run and not bool(args.dry_run_full_model):
        object.__setattr__(cfg, "dim", int(args.dry_run_dim))  # type: ignore[misc]
        object.__setattr__(cfg, "n_layers", int(args.dry_run_n_layers))  # type: ignore[misc]
        object.__setattr__(cfg, "seq_len", int(cfg.dry_run_seq_len))  # type: ignore[misc]
        object.__setattr__(cfg, "max_seq_len", int(cfg.dry_run_seq_len))  # type: ignore[misc]
        # If user didn't explicitly set accumulation/global batch, keep dry-run simple.
        if args.global_batch_size is None and int(args.grad_accum_steps) == 16:
            object.__setattr__(cfg, "grad_accum_steps", 1)  # type: ignore[misc]

    # If user specifies global_batch_size, compute grad_accum_steps from it.
    if cfg.global_batch_size is not None:
        if cfg.global_batch_size <= 0:
            raise ValueError(f"global_batch_size must be > 0, got {cfg.global_batch_size}")
        if cfg.dry_run:
            mb = int(cfg.dry_run_microbatch_size)
        else:
            mb = int(cfg.microbatch_size)
        if mb <= 0:
            raise ValueError(f"microbatch_size must be > 0, got {mb}")
        if cfg.global_batch_size % mb != 0:
            raise ValueError(
                f"global_batch_size must be divisible by microbatch_size for exact accumulation "
                f"(got global_batch_size={cfg.global_batch_size}, microbatch_size={mb})"
            )
        object.__setattr__(cfg, "grad_accum_steps", cfg.global_batch_size // mb)  # type: ignore[misc]

    # Performance knobs (safe defaults).
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        # Helps matmul heuristics on some versions.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_dtype, use_scaler = _resolve_precision(device, cfg.precision)
    # Avoid deprecated torch.cuda.amp.GradScaler; only relevant on CUDA anyway.
    scaler = (
        torch.amp.GradScaler("cuda", enabled=bool(use_scaler and device.type == "cuda"))
        if hasattr(torch, "amp")
        else torch.cuda.amp.GradScaler(enabled=bool(use_scaler and device.type == "cuda"))
    )


    run_name = cfg.run_name or _now_tag()
    run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data + vocab size.
    tokenizer = None
    flat = None
    if cfg.dry_run or cfg.data_source == "random":
        vocab_size = int(cfg.dry_run_vocab_size)
    else:
        flat, tokenizer = _build_or_load_token_buffer(cfg=cfg, run_dir=run_dir)
        vocab_size = int(tokenizer.vocab_size)

    # Record config.
    eff_mb = int(cfg.dry_run_microbatch_size if cfg.dry_run else cfg.microbatch_size)
    eff_global_bs = int(eff_mb * cfg.grad_accum_steps)
    _write_json(
        run_dir / "config.json",
        dataclasses.asdict(cfg)
        | {
            "selected_device": str(device),
            "vocab_size": vocab_size,
            "effective_microbatch_size": eff_mb,
            "effective_global_batch_size": eff_global_bs,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available() and torch.backends.mps.is_built()),
            **_try_get_git_metadata(),
        },
    )

    model = ModularLlama(
        vocab_size=vocab_size,
        n_layers=cfg.n_layers,
        dim=cfg.dim,
        max_seq_len=max(cfg.max_seq_len, cfg.seq_len),
        hidden_dim=cfg.hidden_dim,
        block_factory=lambda d: moneta_factory(d, detach_state_every=int(cfg.moneta_detach_state_every)),
        grad_checkpoint=bool(cfg.grad_checkpoint),
    ).to(device)

    if cfg.compile:
        # Reduce Python overhead in recurrent MONETA; compilation cost amortizes over longer runs.
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Optional W&B init.
    wandb_run = None
    if cfg.wandb and not cfg.dry_run:
        try:
            import wandb  # type: ignore

            tags = [t.strip() for t in (cfg.wandb_tags or "").split(",") if t.strip()] or None
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name or run_dir.name,
                tags=tags,
                mode=cfg.wandb_mode,
                config=dataclasses.asdict(cfg),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to init wandb (disable --wandb to proceed): {e}") from e

    start_step = 0
    if args.resume:
        start_step = _try_resume(Path(args.resume), device=device, model=model, optimizer=optimizer, scaler=scaler)

    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    model.train()
    t_start = time.time()
    t_last_ckpt = time.time()
    t_last_milestone = time.time()
    t_last_log = time.time()
    tokens_since_log = 0

    step = start_step
    target_steps = cfg.dry_run_steps if cfg.dry_run else cfg.max_steps

    while True:
        elapsed = time.time() - t_start
        if cfg.dry_run and step >= target_steps:
            break
        if not cfg.dry_run:
            if cfg.max_time_seconds > 0 and elapsed >= cfg.max_time_seconds:
                break
            if cfg.max_steps > 0 and step >= cfg.max_steps:
                break

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            if cfg.dry_run or cfg.data_source == "random":
                x, y = _make_random_batch(
                    microbatch_size=cfg.dry_run_microbatch_size if cfg.dry_run else cfg.microbatch_size,
                    seq_len=cfg.dry_run_seq_len if cfg.dry_run else cfg.seq_len,
                    vocab_size=vocab_size,
                    device=device,
                )
            else:
                assert flat is not None
                x, y = _sample_batch_from_flat(
                    flat,
                    microbatch_size=cfg.microbatch_size,
                    seq_len=cfg.seq_len,
                    device=device,
                )

            if amp_dtype == torch.float32:
                logits = model(x)  # (B, T, V)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
            else:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(x)  # (B, T, V)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )

            loss_to_backprop = loss / cfg.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            total_loss += float(loss.item())
            tokens_since_log += int(x.shape[0] * x.shape[1])

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1

        if step % cfg.log_every_steps == 0:
            dt_log = max(1e-9, time.time() - t_last_log)
            toks_per_s = tokens_since_log / dt_log
            record = {
                "step": step,
                "elapsed_s": time.time() - t_start,
                "loss": total_loss / cfg.grad_accum_steps,
                "tokens_per_s": toks_per_s,
                "device": str(device),
                "precision": str(amp_dtype),
                "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            _append_jsonl(metrics_path, record)
            if wandb_run is not None:
                try:
                    wandb_run.log(record, step=step)
                except Exception:
                    pass
            print(
                f"[step {step}] loss={record['loss']:.4f} tok/s={record['tokens_per_s']:.0f} grad_norm={record['grad_norm']:.3f}"
            )
            t_last_log = time.time()
            tokens_since_log = 0

        if not cfg.dry_run and (time.time() - t_last_ckpt) >= cfg.ckpt_every_seconds:
            path = _checkpoint(
                ckpt_dir=ckpt_dir, model=model, optimizer=optimizer, scaler=(scaler if scaler.is_enabled() else None), step=step, cfg=cfg
            )
            if (time.time() - t_last_milestone) >= cfg.ckpt_milestone_every_seconds:
                _maybe_write_milestone_marker(ckpt_dir=ckpt_dir, step=step, ckpt_path=path)
                t_last_milestone = time.time()

            try:
                _maybe_gdrive_sync_run(cfg=cfg, run_dir=run_dir, ckpt_path=path)
            except Exception as e:
                print(f"[warn] gdrive sync failed: {e}")

            if cfg.ckpt_prune:
                _prune_checkpoints(
                    ckpt_dir=ckpt_dir,
                    keep_last=int(cfg.ckpt_keep_last),
                    keep_milestones=int(cfg.ckpt_keep_milestones),
                )
            print(f"Saved checkpoint: {path}")
            t_last_ckpt = time.time()

    # Final checkpoint for non-dry runs.
    if not cfg.dry_run:
        path = _checkpoint(
            ckpt_dir=ckpt_dir, model=model, optimizer=optimizer, scaler=(scaler if scaler.is_enabled() else None), step=step, cfg=cfg
        )
        _maybe_write_milestone_marker(ckpt_dir=ckpt_dir, step=step, ckpt_path=path)
        try:
            _maybe_gdrive_sync_run(cfg=cfg, run_dir=run_dir, ckpt_path=path)
        except Exception as e:
            print(f"[warn] gdrive sync failed: {e}")
        if cfg.ckpt_prune:
            _prune_checkpoints(
                ckpt_dir=ckpt_dir,
                keep_last=int(cfg.ckpt_keep_last),
                keep_milestones=int(cfg.ckpt_keep_milestones),
            )
        print(f"Finished. Final checkpoint: {path}")
    else:
        print("Dry run finished successfully.")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

