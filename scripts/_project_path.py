"""让脚本能够导入仓库 ``src`` 目录中的模块。

Make the repository's ``src`` directory importable by a script.
"""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_src_to_path() -> None:
    """Add ``<repository>/src`` to Python's module search path."""

    src_root = Path(__file__).resolve().parents[1] / "src"
    if not src_root.is_dir():
        raise RuntimeError(f"project source directory does not exist: {src_root}")

    source_path = str(src_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
