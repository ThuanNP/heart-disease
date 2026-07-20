"""Test thực thi notebook: chạy end-to-end, khẳng định KHÔNG lỗi và KHÔNG warning.

Mỗi notebook được nạp, loại bỏ ô ``!pip install`` (không phụ thuộc mạng khi test),
rồi thực thi bằng nbclient. Sau đó quét toàn bộ output:
- không có ô nào sinh output kiểu ``error`` (traceback);
- không có stream ``stderr`` (cảnh báo deprecated/runtime in ra stderr sẽ bị bắt).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = [
    "dataset_description.ipynb",
    "descriptive_statistics.ipynb",
    "hypothesis_testing.ipynb",
]


def _strip_pip_installs(nb: nbformat.NotebookNode) -> None:
    """Thay ô cài đặt gói bằng ô rỗng để test không phụ thuộc mạng."""
    for cell in nb.cells:
        if cell.cell_type == "code" and "pip install" in "".join(cell.source):
            cell.source = "# (bỏ qua 'pip install' khi chạy test)"


def _collect_problems(nb: nbformat.NotebookNode) -> list[str]:
    problems: list[str] = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            if otype == "error":
                tb = f"{out.get('ename')}: {out.get('evalue')}"
                problems.append(f"[ô {idx}] LỖI: {tb}")
            elif otype == "stream" and out.get("name") == "stderr":
                text = out.get("text", "").strip()
                if text:
                    problems.append(f"[ô {idx}] STDERR/cảnh báo: {text[:400]}")
    return problems


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_chay_khong_loi_khong_warning(nb_name, project_root: Path):
    nb_path = project_root / nb_name
    assert nb_path.exists(), f"Không tìm thấy notebook: {nb_path}"

    nb = nbformat.read(nb_path, as_version=4)
    _strip_pip_installs(nb)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        # Chạy với thư mục làm việc = gốc dự án để 'dataset/heart.csv' resolve đúng.
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute()

    problems = _collect_problems(nb)
    assert not problems, f"{nb_name} có vấn đề khi chạy:\n" + "\n".join(problems)
