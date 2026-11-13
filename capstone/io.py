"""File integrity and serialization helpers."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path, **kwargs):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip", **kwargs)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib", "threadpoolctl")
    }}


def code_hashes():
    root = Path(__file__).parent
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}
