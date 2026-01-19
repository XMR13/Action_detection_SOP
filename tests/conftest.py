from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    # Some pytest import modes (and some Windows invocations) may not include the repo
    # root on sys.path, causing imports like `import yolo_kit` to fail.
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_syspath()

