"""Unit test cho hypothesis_testing.py.

Chiến lược: đối chiếu số liệu do module tính với công thức độc lập / scipy,
kiểm tra cấu trúc kết quả, quy tắc bác bỏ H0, và xử lý lỗi tham số.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

import hypothesis_testing as ht
from hypothesis_testing import hypothesis_test, list_test_types

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Danh mục & xác thực tham số
# ---------------------------------------------------------------------------


def test_list_test_types_khop_voi_handlers():
    types = list_test_types()
    assert set(types) == set(ht._HANDLERS)
    for name, info in types.items():
        assert "mo_ta" in info and isinstance(info["mo_ta"], str)
        assert "tham_so" in info and isinstance(info["tham_so"], tuple)
        assert len(info["tham_so"]) > 0, name


def test_list_test_types_tra_ve_ban_sao():
    types = list_test_types()
    types["loai_gia"] = {}  # sửa bản trả về
    assert "loai_gia" not in list_test_types()  # không ảnh hưởng nội bộ


def test_loai_kiem_dinh_khong_hop_le():
    with pytest.raises(ValueError, match="không hợp lệ"):
        hypothesis_test("khong_ton_tai", x_bar=1, mu0=0, sigma=1, n=10)


def test_thieu_tham_so():
    with pytest.raises(ValueError, match="thiếu tham số"):
        hypothesis_test("z_mean_sigma_known", x_bar=1, mu0=0)  # thiếu sigma, n


def test_ten_kiem_dinh_duoc_strip_khoang_trang():
    r = hypothesis_test("  z_proportion  ", f=30, n=100, p0=0.5)
    assert r["test_type"] == "z_proportion"


# ---------------------------------------------------------------------------
# Kiểm định có tham số — đối chiếu công thức độc lập
# ---------------------------------------------------------------------------


def test_z_mean_sigma_known():
    x_bar, mu0, sigma, n = 2020, 2000, 100, 36
    r = hypothesis_test(
        "z_mean_sigma_known",
        x_bar=x_bar, mu0=mu0, sigma=sigma, n=n,
        alpha=0.10, alternative="greater",
    )
    z_exp = (x_bar - mu0) / (sigma / math.sqrt(n))
    assert r["statistic"] == pytest.approx(z_exp)
    assert r["statistic_name"] == "Z"
    assert r["p_value"] == pytest.approx(stats.norm.sf(z_exp))
    assert r["critical_value"] == pytest.approx(stats.norm.ppf(1 - 0.10))
    assert r["reject_h0"] is (z_exp > r["critical_value"])


def test_t_mean_sigma_unknown_khop_scipy_ttest():
    # Dựng mẫu để so với scipy.stats.ttest_1samp
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=40)
    x_bar = data.mean()
    s = data.std(ddof=1)
    n = len(data)
    mu0 = 4.5
    r = hypothesis_test(
        "t_mean_sigma_unknown",
        x_bar=x_bar, mu0=mu0, s=s, n=n,
        alpha=ALPHA, alternative="two-sided",
    )
    t_ref, p_ref = stats.ttest_1samp(data, popmean=mu0)
    assert r["statistic"] == pytest.approx(t_ref)
    assert r["p_value"] == pytest.approx(p_ref)
    assert r["df"] == n - 1


def test_z_proportion():
    f, n, p0 = 30, 100, 0.5
    r = hypothesis_test("z_proportion", f=f, n=n, p0=p0, alternative="two-sided")
    se = math.sqrt(p0 * (1 - p0) / n)
    z_exp = (f / n - p0) / se
    assert r["p_hat"] == pytest.approx(0.30)
    assert r["statistic"] == pytest.approx(z_exp)
    assert r["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z_exp)))


def test_z_two_means_independent():
    x1, x2, s1, s2, n1, n2 = 105.0, 100.0, 10.0, 12.0, 50, 60
    r = hypothesis_test(
        "z_two_means_independent",
        x1=x1, x2=x2, sigma1=s1, sigma2=s2, n1=n1, n2=n2,
        alternative="two-sided",
    )
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    z_exp = (x1 - x2) / se
    assert r["statistic"] == pytest.approx(z_exp)
    assert r["se"] == pytest.approx(se)


def test_t_two_means_independent_khop_scipy():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 3, size=30)
    b = rng.normal(8, 3, size=35)
    r = hypothesis_test(
        "t_two_means_independent",
        x1=a.mean(), x2=b.mean(),
        s1=a.std(ddof=1), s2=b.std(ddof=1),
        n1=len(a), n2=len(b),
        alternative="two-sided",
    )
    t_ref, p_ref = stats.ttest_ind(a, b, equal_var=True)
    assert r["statistic"] == pytest.approx(t_ref)
    assert r["p_value"] == pytest.approx(p_ref)
    assert r["df"] == len(a) + len(b) - 2


def test_t_two_means_welch_khop_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(10, 2, size=25)
    b = rng.normal(9, 5, size=40)
    r = hypothesis_test(
        "t_two_means_welch",
        x1=a.mean(), x2=b.mean(),
        s1=a.std(ddof=1), s2=b.std(ddof=1),
        n1=len(a), n2=len(b),
        alternative="two-sided",
    )
    t_ref, p_ref = stats.ttest_ind(a, b, equal_var=False)
    assert r["statistic"] == pytest.approx(t_ref)
    assert r["p_value"] == pytest.approx(p_ref)
    # df Welch (Welch–Satterthwaite) tính độc lập
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    df_exp = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    assert r["df"] == pytest.approx(df_exp)


def test_t_paired_khop_scipy():
    rng = np.random.default_rng(3)
    before = rng.normal(50, 5, size=20)
    after = before + rng.normal(-2, 3, size=20)
    d = before - after
    r = hypothesis_test(
        "t_paired",
        d_bar=d.mean(), sd=d.std(ddof=1), n=len(d),
        alternative="two-sided",
    )
    t_ref, p_ref = stats.ttest_rel(before, after)
    assert r["statistic"] == pytest.approx(t_ref)
    assert r["p_value"] == pytest.approx(p_ref)


def test_z_two_proportions():
    f1, n1, f2, n2 = 40, 100, 30, 120
    r = hypothesis_test(
        "z_two_proportions",
        f1=f1, n1=n1, f2=f2, n2=n2, alternative="two-sided",
    )
    p1, p2 = f1 / n1, f2 / n2
    p_bar = (f1 + f2) / (n1 + n2)
    se = math.sqrt(p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
    z_exp = (p1 - p2) / se
    assert r["statistic"] == pytest.approx(z_exp)
    assert r["p_bar"] == pytest.approx(p_bar)


def test_chi2_variance():
    s2, sigma0_2, n = 25.0, 20.0, 30
    r = hypothesis_test(
        "chi2_variance", s2=s2, sigma0_2=sigma0_2, n=n, alternative="greater",
    )
    chi2_exp = (n - 1) * s2 / sigma0_2
    assert r["statistic"] == pytest.approx(chi2_exp)
    assert r["df"] == n - 1
    assert r["p_value"] == pytest.approx(stats.chi2.sf(chi2_exp, n - 1))


def test_f_two_variances():
    s1_2, s2_2, n1, n2 = 30.0, 15.0, 25, 20
    r = hypothesis_test(
        "f_two_variances", s1_2=s1_2, s2_2=s2_2, n1=n1, n2=n2, alternative="greater",
    )
    f_exp = s1_2 / s2_2
    assert r["statistic"] == pytest.approx(f_exp)
    assert r["p_value"] == pytest.approx(stats.f.sf(f_exp, n1 - 1, n2 - 1))


# ---------------------------------------------------------------------------
# Kiểm định phi tham số — đối chiếu scipy
# ---------------------------------------------------------------------------


def test_kruskal_wallis_khop_scipy():
    groups = [[45.5, 44.5, 46.0], [45.8, 45.1, 47.2], [43.7, 44.0, 42.9]]
    r = hypothesis_test("kruskal_wallis", groups=groups)
    h_ref, p_ref = stats.kruskal(*groups)
    assert r["statistic"] == pytest.approx(h_ref)
    assert r["p_value"] == pytest.approx(p_ref)
    assert r["k"] == 3 and r["df"] == 2
    assert r["alternative"] == "two-sided"


def test_sign_test_hai_phia():
    diffs = [1, 2, -1, 3, 0, -2, 4, 5]  # bỏ số 0 -> n=7, n_pos=5
    r = hypothesis_test("sign_test", differences=diffs, alternative="two-sided")
    assert r["n"] == 7
    assert r["n_positive"] == 5
    assert r["n_negative"] == 2
    p_exp = 2 * min(stats.binom.cdf(5, 7, 0.5), stats.binom.sf(4, 7, 0.5))
    assert r["p_value"] == pytest.approx(p_exp)


def test_wilcoxon_signed_rank_khop_scipy():
    x = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
    y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
    r = hypothesis_test("wilcoxon_signed_rank", x=x, y=y, alternative="two-sided")
    stat_ref, p_ref = stats.wilcoxon(x, y, alternative="two-sided")
    assert r["statistic"] == pytest.approx(stat_ref)
    assert r["p_value"] == pytest.approx(p_ref)


def test_wilcoxon_rank_sum_mau_nho_dung_chinh_xac():
    s1 = [1.2, 2.3, 3.1, 4.0, 5.5]
    s2 = [2.0, 2.8, 3.9, 4.4, 6.1]
    r = hypothesis_test("wilcoxon_rank_sum", sample1=s1, sample2=s2, alternative="two-sided")
    u_ref, p_ref = stats.mannwhitneyu(s1, s2, alternative="two-sided")
    assert r["statistic"] == pytest.approx(u_ref)
    assert r["p_value"] == pytest.approx(p_ref)
    assert r["use_z_approx"] is False
    assert r["method"] == "exact"
    assert r["critical_value"] is None
    assert r["reject_h0"] == bool(p_ref < ALPHA)


def test_wilcoxon_rank_sum_mau_lon_dung_xap_xi_z():
    rng = np.random.default_rng(4)
    s1 = rng.normal(10, 2, size=15).tolist()
    s2 = rng.normal(8, 2, size=18).tolist()
    r = hypothesis_test("wilcoxon_rank_sum", sample1=s1, sample2=s2, alternative="two-sided")
    assert r["use_z_approx"] is True
    assert r["method"] == "z_approx"
    # z = (U - mu_U)/sigma_U
    mu_u = len(s1) * len(s2) / 2
    sigma_u = math.sqrt(len(s1) * len(s2) * (len(s1) + len(s2) + 1) / 12)
    assert r["z"] == pytest.approx((r["statistic"] - mu_u) / sigma_u)
    assert r["reject_h0"] == bool(abs(r["z"]) > r["critical_value"])


def test_chi2_independence_khop_scipy():
    observed = [[30, 20], [25, 35]]
    r = hypothesis_test("chi2_independence", observed=observed)
    chi2_ref, p_ref, df_ref, _ = stats.chi2_contingency(np.array(observed))
    assert r["statistic"] == pytest.approx(chi2_ref)
    assert r["p_value"] == pytest.approx(p_ref)
    assert r["df"] == df_ref
    assert np.array(r["expected"]).shape == (2, 2)


def test_chi2_goodness_of_fit_khop_scipy():
    observed = [18, 22, 20, 40]
    expected = [25, 25, 25, 25]
    r = hypothesis_test("chi2_goodness_of_fit", observed=observed, expected=expected)
    chi2_ref, p_ref = stats.chisquare(observed, f_exp=expected)
    assert r["statistic"] == pytest.approx(chi2_ref)
    assert r["p_value"] == pytest.approx(p_ref)
    assert r["df"] == len(observed) - 1


# ---------------------------------------------------------------------------
# Quy tắc bác bỏ H0 theo giá trị tới hạn
# ---------------------------------------------------------------------------


def test_reject_h0_greater():
    r = hypothesis_test(
        "z_mean_sigma_known", x_bar=110, mu0=100, sigma=10, n=100,
        alpha=ALPHA, alternative="greater",
    )
    assert r["statistic"] > r["critical_value"]
    assert r["reject_h0"] is True


def test_khong_reject_h0_khi_thong_ke_nho():
    r = hypothesis_test(
        "z_mean_sigma_known", x_bar=100.1, mu0=100, sigma=10, n=100,
        alpha=ALPHA, alternative="greater",
    )
    assert r["reject_h0"] is False


def test_reject_h0_two_sided_dung_cap_gia_tri_toi_han():
    r = hypothesis_test(
        "z_mean_sigma_known", x_bar=130, mu0=100, sigma=10, n=100,
        alpha=ALPHA, alternative="two-sided",
    )
    low, high = r["critical_value"]
    assert low < 0 < high
    assert r["reject_h0"] is (r["statistic"] < low or r["statistic"] > high)


# ---------------------------------------------------------------------------
# Cấu trúc & kiểu dữ liệu kết quả
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_type,kwargs",
    [
        ("z_mean_sigma_known", dict(x_bar=10, mu0=9, sigma=2, n=30)),
        ("t_mean_sigma_unknown", dict(x_bar=10, mu0=9, s=2, n=30)),
        ("z_proportion", dict(f=20, n=50, p0=0.5)),
        ("chi2_goodness_of_fit", dict(observed=[10, 20, 30], expected=[20, 20, 20])),
    ],
)
def test_ket_qua_co_kieu_du_lieu_chuan(test_type, kwargs):
    r = hypothesis_test(test_type, **kwargs)
    assert isinstance(r["statistic"], float)
    assert isinstance(r["p_value"], float)
    assert isinstance(r["reject_h0"], bool)
    assert 0.0 <= r["p_value"] <= 1.0
    assert r["test_type"] == test_type
