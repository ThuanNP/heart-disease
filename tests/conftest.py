"""Cấu hình chung cho bộ test.

Cung cấp fixture đường dẫn gốc dự án và dữ liệu dùng chung. Cảnh báo hạ tầng của
zmq/Jupyter trên Windows (không phải từ code dự án) được lọc trong pytest.ini.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def dataset_path(project_root: Path) -> Path:
    path = project_root / "dataset" / "heart.csv"
    assert path.exists(), f"Không tìm thấy tập dữ liệu: {path}"
    return path
