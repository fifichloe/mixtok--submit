"""Learning-rate schedule and atomic checkpointing for Halton training."""
import math
import shutil
from pathlib import Path

import torch

from gene.support.paths import atomic_json, sha256


def set_budget_cosine_lr(optimizer, step: int, peak_lr: float, warmup: int,
                         hold_until: int, decay_until: int, min_lr: float) -> float:
    if not (1 <= step <= decay_until):
        raise ValueError("step must be within the budget-cosine horizon")
    if not (0 <= warmup <= hold_until < decay_until):
        raise ValueError("expected 0 <= warmup <= hold_until < decay_until")
    if not (0.0 <= min_lr <= peak_lr):
        raise ValueError("expected 0 <= min_lr <= peak_lr")
    if warmup and step <= warmup:
        lr = peak_lr * step / warmup
    elif step <= hold_until:
        lr = peak_lr
    else:
        progress = (step - hold_until) / (decay_until - hold_until)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_checkpoint(out: Path, core, optimizer, step: int, config: dict,
                    cursor: dict, rng_states: list):
    if shutil.disk_usage(out).free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free for checkpoint")
    state = dict(step=step, config=config, cursor=cursor, rng_states=rng_states,
                 model={k: v.detach().cpu() for k, v in core.state_dict().items()},
                 optimizer=optimizer.state_dict())
    temporary = out / "latest.pt.tmp"
    torch.save(state, temporary)
    check = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
    if check["step"] != step or set(check["model"]) != set(state["model"]):
        raise RuntimeError("checkpoint round-trip header mismatch")
    for key in state["model"]:
        if not torch.equal(check["model"][key], state["model"][key]):
            raise RuntimeError(f"checkpoint round-trip mismatch: {key}")
    digest = sha256(temporary)
    temporary.replace(out / "latest.pt")
    atomic_json(out / "latest.json", dict(step=step, sha256=digest,
                bytes=(out / "latest.pt").stat().st_size, round_trip_exact=True,
                config=config))
