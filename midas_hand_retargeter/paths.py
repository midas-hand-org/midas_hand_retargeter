"""Path discovery helpers for keeping MIDAS repos separate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _candidate_roots(start: Path | None = None) -> Iterable[Path]:
    seen: set[Path] = set()
    starts = [
        Path.cwd(),
        Path(__file__).resolve(),
    ]
    if start is not None:
        starts.append(Path(start).resolve())

    for item in starts:
        for parent in (item, *item.parents):
            if parent not in seen:
                seen.add(parent)
                yield parent


def find_mujoco_repo(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the MIDAS MuJoCo repo path.

    Resolution order:
    1. explicit ``path`` argument
    2. ``MIDAS_HAND_MUJOCO_DIR`` environment variable
    3. sibling directory named ``midas_hand_mujoco`` near cwd or this package
    """

    explicit = path or os.environ.get("MIDAS_HAND_MUJOCO_DIR")
    if explicit:
        repo = Path(explicit).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"MIDAS MuJoCo repo does not exist: {repo}")
        return repo

    for root in _candidate_roots():
        candidate = root / "midas_hand_mujoco"
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find midas_hand_mujoco. Pass mujoco_repo=... or set "
        "MIDAS_HAND_MUJOCO_DIR."
    )


def midas_description_dir(
    mujoco_repo: str | os.PathLike[str] | None = None,
) -> Path:
    return find_mujoco_repo(mujoco_repo) / "assets" / "midas_description"


def default_urdf_path(
    mujoco_repo: str | os.PathLike[str] | None = None,
) -> Path:
    path = midas_description_dir(mujoco_repo) / "midas_hand_urdf.urdf"
    if not path.exists():
        raise FileNotFoundError(f"MIDAS hand URDF does not exist: {path}")
    return path


def default_mjcf_path(
    mujoco_repo: str | os.PathLike[str] | None = None,
) -> Path:
    path = midas_description_dir(mujoco_repo) / "midas_whole_hand.xml"
    if not path.exists():
        raise FileNotFoundError(f"MIDAS whole-hand MJCF does not exist: {path}")
    return path
