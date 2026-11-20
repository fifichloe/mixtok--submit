"""Locate the external tokenizer implementations used by the training code."""
import os
import sys
from pathlib import Path


def load_vendor():
    for key, suffix in (
        ("MIXTOK_TITOK_ROOT", Path(".")),
        ("MIXTOK_LLAMAGEN_ROOT", Path("tokenizer/tokenizer_image")),
    ):
        root = os.environ.get(key)
        if not root:
            raise RuntimeError(f"Set {key} to the external source checkout")
        source = (Path(root).expanduser().resolve() / suffix).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
