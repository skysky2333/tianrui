from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_timestamped_run_dir(base_out_dir: str) -> tuple[Path, Path, str]:
    """
    Treats `base_out_dir` as a *container* directory and creates a unique
    timestamped subdirectory for this run.

    Returns: (base_dir, run_dir, run_name)
      - base_dir: Path(base_out_dir)
      - run_dir: base_dir / run_name
      - run_name: timestamp string like "20260228_141530" (with optional _vN suffix)
    """
    base_dir = Path(base_out_dir).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp_now()
    run_name = ts
    run_dir = base_dir / run_name
    v = 1
    while run_dir.exists():
        run_name = f"{ts}_v{v}"
        run_dir = base_dir / run_name
        v += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return base_dir, run_dir, run_name


def write_latest_run_marker(*, base_dir: Path, run_name: str, run_dir: Path, argv: Sequence[str]) -> None:
    marker_path = base_dir / "latest_run.json"
    payload = {
        "run_name": str(run_name),
        "run_dir": str(run_dir),
        "argv": [str(x) for x in argv],
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    marker_path.write_text(json.dumps(payload, indent=2) + "\n")


def copy_top_level_files_to_base(*, run_dir: Path, base_dir: Path) -> None:
    """
    Copy all top-level files from the run directory into the base directory.

    - Files only (directories are skipped).
    - Overwrites existing files in base_dir (intended to keep "latest" artifacts).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dir.iterdir():
        if p.is_file() and not p.name.startswith("."):
            shutil.copy2(p, base_dir / p.name)


def finalize_out_dir(*, base_dir: Path, run_dir: Path, run_name: str, argv: Sequence[str]) -> None:
    write_latest_run_marker(base_dir=base_dir, run_name=run_name, run_dir=run_dir, argv=argv)
    copy_top_level_files_to_base(run_dir=run_dir, base_dir=base_dir)
