from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cairo.paths import ensure_dir


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
