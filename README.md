# Phân tích thống kê tập dữ liệu Bệnh tim (Heart Disease)

Đồ án môn **Thống kê trong Khoa học Máy tính**: mô tả dữ liệu, thống kê mô tả và
**kiểm định giả thuyết** (có tham số và phi tham số) trên tập dữ liệu bệnh tim.

Dự án gồm một thư viện tính toán kiểm định giả thuyết ([`hypothesis_testing.py`](hypothesis_testing.py))
cùng 3 notebook minh họa, kèm bộ **unit test** đảm bảo hàm và notebook chạy đúng,
không lỗi, không cảnh báo.

---

## 1. Nguồn dữ liệu

| | |
| --- | --- |
| **Tên** | Heart Disease Dataset |
| **Nguồn** | UCI Machine Learning Repository / Kaggle (johnsmith88) |
| **URL** | <https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset> |
| **File cục bộ** | [`dataset/heart.csv`](dataset/heart.csv) |
| **Kích thước** | 1025 dòng × 14 cột |
| **Biến mục tiêu** | `target` — 1 = có bệnh, 0 = không bệnh |

Dữ liệu gốc từ năm 1988, tổng hợp từ 4 cơ sở (Cleveland, Hungary, Switzerland,
Long Beach V). Bản `heart.csv` này là phiên bản nhân bản từ dữ liệu Cleveland gốc
(303 mẫu) — không ảnh hưởng tới quy trình kiểm định minh họa.

**14 thuộc tính:** `age`, `sex`, `cp` (loại đau ngực), `trestbps` (huyết áp lúc nghỉ),
`chol` (cholesterol), `fbs` (đường huyết đói), `restecg`, `thalach` (nhịp tim tối đa),
`exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`. Mô tả chi tiết từng thuộc tính
xem trong notebook [`dataset_description.ipynb`](dataset_description.ipynb).

---

## 2. Cấu trúc dự án

```text
heart-disease/
├── dataset/heart.csv              # Tập dữ liệu
├── dataset_description.ipynb      # 1.1 — Mô tả tập dữ liệu
├── descriptive_statistics.ipynb   # 1.2 — Thống kê mô tả + biểu đồ
├── hypothesis_testing.ipynb       # Kiểm định giả thuyết trên dữ liệu bệnh tim
├── hypothesis_testing.py          # Thư viện tính số liệu kiểm định giả thuyết
├── export_pipeline.py             # Xuất notebook sang .docx (xem mục 7)
├── export/                        # Kết xuất tự động (markdown/, docx/) — không sửa tay
├── tests/                         # Unit test
│   ├── conftest.py
│   ├── test_hypothesis_testing.py #   Test các hàm
│   └── test_notebooks.py          #   Test thực thi 3 notebook
├── pytest.ini                     # Cấu hình pytest
├── requirements.txt               # Phụ thuộc trực tiếp
└── README.md
```

---

## 3. Cài đặt môi trường

Yêu cầu **Python 3.11+**. Cài các gói cần thiết từ [`requirements.txt`](requirements.txt):

```powershell
pip install -r requirements.txt        # hoặc: uv pip install -r requirements.txt
```

Ngoài `numpy`/`scipy`/`pandas`/`matplotlib`/`seaborn`, file này còn khai báo
`nbformat`, `nbclient`, `ipykernel` — bắt buộc phải có thì `tests/test_notebooks.py`
mới thực thi được notebook.

---

## 4. Thư viện `hypothesis_testing.py`

Hàm chính là `hypothesis_test(test_type, alpha=0.05, alternative="two-sided", **kwargs)`,
trả về `dict` gồm `statistic`, `p_value`, `critical_value`, `reject_h0`, ... Thư viện
**chỉ tính số liệu**, không đặt sẵn giả thuyết nghiên cứu.

```python
from hypothesis_testing import hypothesis_test, list_test_types

# Kiểm định t một mẫu, giả thuyết một phía
res = hypothesis_test(
    "t_mean_sigma_unknown",
    x_bar=2020, mu0=2000, s=92, n=36,
    alpha=0.10, alternative="greater",
)
print(res["statistic"], res["p_value"], res["reject_h0"])

# Xem toàn bộ loại kiểm định và tham số cần truyền
list_test_types()
```

### Các loại kiểm định hỗ trợ

Các công thức được cài đặt theo giáo trình môn học. Tài liệu giảng dạy không được
phát hành kèm repo này.

**Có tham số**: `z_mean_sigma_known`, `t_mean_sigma_unknown`,
`z_proportion`, `z_two_means_independent`, `t_two_means_independent`, `t_two_means_welch`,
`t_paired`, `z_two_proportions`, `chi2_variance`, `f_two_variances`.

**Phi tham số**: `kruskal_wallis`, `sign_test`,
`wilcoxon_signed_rank`, `wilcoxon_rank_sum`, `chi2_independence`, `chi2_goodness_of_fit`.

Gọi `list_test_types()` để biết tham số bắt buộc của từng loại.

---

## 5. Chạy notebook

Mở bằng Jupyter / VS Code và chạy tuần tự **từ thư mục `heart-disease`** (đường dẫn
dữ liệu `dataset/heart.csv` là tương đối):

```powershell
cd heart-disease
jupyter lab        # hoặc: jupyter notebook
```

Thứ tự đọc gợi ý:

1. [`dataset_description.ipynb`](dataset_description.ipynb) — tổng quan, ý nghĩa thuộc tính.
2. [`descriptive_statistics.ipynb`](descriptive_statistics.ipynb) — thống kê mô tả, phân phối, tương quan.
3. [`hypothesis_testing.ipynb`](hypothesis_testing.ipynb) — các kiểm định giả thuyết.

---

## 6. Chạy test

Bộ test kiểm tra: (a) các hàm trong `hypothesis_testing.py` (đối chiếu số liệu với
`scipy`), và (b) cả 3 notebook chạy **end-to-end không lỗi, không cảnh báo**.

Cấu hình `pytest.ini` đặt `filterwarnings = error` → **mọi warning (kể cả
`DeprecationWarning`) đều làm test thất bại**, đảm bảo code luôn sạch.

Mở terminal trong thư mục `heart-disease` rồi chạy:

```powershell
python -m pytest                                     # chạy tất cả
python -m pytest -v                                  # chi tiết từng test
python -m pytest tests/test_hypothesis_testing.py    # chỉ test các hàm
python -m pytest tests/test_notebooks.py             # chỉ test 3 notebook
python -m pytest -k wilcoxon                         # chỉ test có tên chứa "wilcoxon"
```

---

## 7. Xuất notebook sang Word (.docx)

Pipeline xuất Word chuẩn cho notebook (giữ công thức LaTeX, biểu đồ và ép bảng DataFrame về bảng Markdown/Word):

```powershell
.venv/Scripts/python.exe export_pipeline.py --python .venv/Scripts/python.exe
```

Mặc định pipeline sẽ:

1. Chuyển `ipynb -> markdown` vào `export/markdown/`.
2. Chuẩn hóa markdown: chuyển mọi bảng HTML do `display(DataFrame)` sinh ra thành pipe table Markdown.
3. Chuyển `markdown -> docx` vào `export/docx/`.
