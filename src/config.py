from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    project_root: str
    data_dir: str
    artifacts_dir: str

    @staticmethod
    def default() -> "Paths":
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return Paths(
            project_root=root,
            data_dir=os.path.join(root, "data"),
            artifacts_dir=os.path.join(root, "artifacts"),
        )

