"""Portable asset paths and small file helpers for generation training."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("MOTAR_ASSETS", REPO / "assets/generation"))
RESULT_ROOT = Path(os.environ.get("MOTAR_RESULTS", REPO / "results/generation"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def output_path(path):
    path = Path(path).resolve()
    root = RESULT_ROOT.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("output must be a named child of MOTAR_RESULTS")
    if path.is_relative_to(REPO):
        raise ValueError("large outputs must be outside the code repository")
    return path
