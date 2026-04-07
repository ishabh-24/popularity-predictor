from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path

    @staticmethod
    def default() -> "Paths":
        root = Path(__file__).resolve().parents[1]
        return Paths(
            project_root=root,
            data_dir=root / "data",
            artifacts_dir=root / "artifacts",
        )

