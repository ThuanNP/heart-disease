"""
Công cụ tính số liệu kiểm định giả thuyết.

Công thức cài đặt theo giáo trình môn học (phần có tham số và phần phi tham số).

Chỉ tính toán và trả số liệu; không đặt sẵn giả thuyết nghiên cứu.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats

Alternative = Literal["greater", "less", "two-sided"]

# Wilcoxon rank-sum: n1, n2 > 10 → xấp xỉ Z; ngược lại → phân phối chính xác / tra bảng U
WILCOXON_RANK_SUM_Z_MIN_N = 10

# Danh mục loại kiểm định: mô tả tiếng Việt + tham số bắt buộc khi gọi hypothesis_test(...)
TEST_PARAMS: dict[str, dict[str, str | tuple[str, ...]]] = {
    # --- Chương 04_01: kiểm định có tham số ---
    "z_mean_sigma_known": {
        "mo_ta": "Kiểm định Z về trung bình tổng thể (đã biết độ lệch chuẩn σ)",
        "tham_so": ("x_bar", "mu0", "sigma", "n"),
    },
    "t_mean_sigma_unknown": {
        "mo_ta": "Kiểm định t về trung bình tổng thể (chưa biết σ, dùng s mẫu)",
        "tham_so": ("x_bar", "mu0", "s", "n"),
    },
    "z_proportion": {
        "mo_ta": "Kiểm định Z về tỉ lệ của tổng thể",
        "tham_so": ("f", "n", "p0"),
    },
    "z_two_means_independent": {
        "mo_ta": "Kiểm định Z khác biệt hai trung bình (2 mẫu độc lập, σ tổng thể đã biết)",
        "tham_so": ("x1", "x2", "sigma1", "sigma2", "n1", "n2"),
    },
    "t_two_means_independent": {
        "mo_ta": "Kiểm định t khác biệt hai trung bình (2 mẫu độc lập, σ chưa biết, gộp phương sai)",
        "tham_so": ("x1", "x2", "s1", "s2", "n1", "n2"),
    },
    "t_two_means_welch": {
        "mo_ta": "Kiểm định Welch t khác biệt hai trung bình (2 mẫu độc lập, không giả định phương sai bằng nhau)",
        "tham_so": ("x1", "x2", "s1", "s2", "n1", "n2"),
    },
    "t_paired": {
        "mo_ta": "Kiểm định t khác biệt hai trung bình (2 mẫu cặp / phối hợp từng cặp)",
        "tham_so": ("d_bar", "sd", "n"),
    },
    "z_two_proportions": {
        "mo_ta": "Kiểm định Z khác biệt tỉ lệ của hai tổng thể",
        "tham_so": ("f1", "n1", "f2", "n2"),
    },
    "chi2_variance": {
        "mo_ta": "Kiểm định χ² về phương sai của tổng thể",
        "tham_so": ("s2", "sigma0_2", "n"),
    },
    "f_two_variances": {
        "mo_ta": "Kiểm định F khác biệt phương sai của hai tổng thể",
        "tham_so": ("s1_2", "s2_2", "n1", "n2"),
    },
    # --- Chương 04_02: kiểm định phi tham số ---
    "kruskal_wallis": {
        "mo_ta": "Kiểm định Kruskal-Wallis (so sánh ≥ 3 nhóm độc lập, không cần phân phối chuẩn)",
        "tham_so": ("groups",),
    },
    "sign_test": {
        "mo_ta": "Kiểm định dấu (mẫu cặp, chỉ xét tăng/giảm sau sai lệch)",
        "tham_so": ("differences",),
    },
    "wilcoxon_signed_rank": {
        "mo_ta": "Kiểm định Wilcoxon dấu và hạng (mẫu cặp, phi tham số)",
        "tham_so": ("x", "y"),
    },
    "wilcoxon_rank_sum": {
        "mo_ta": "Kiểm định Wilcoxon tổng hạng / Mann-Whitney (2 mẫu độc lập; trả U và xấp xỉ Z)",
        "tham_so": ("sample1", "sample2"),
    },
    "chi2_independence": {
        "mo_ta": "Kiểm định χ² sự độc lập giữa hai biến định tính (bảng chéo)",
        "tham_so": ("observed",),
    },
    "chi2_goodness_of_fit": {
        "mo_ta": "Kiểm định χ² sự phù hợp (so khớp tần số quan sát với tần số kỳ vọng)",
        "tham_so": ("observed", "expected"),
    },
}


def list_test_types() -> dict[str, dict[str, str | tuple[str, ...]]]:
    """Liệt kê loại kiểm định (mô tả + tham số cần truyền)."""
    return dict(TEST_PARAMS)


def _check_params(test_type: str, kwargs: dict[str, Any]) -> None:
    if test_type not in TEST_PARAMS:
        valid = ", ".join(sorted(TEST_PARAMS))
        raise ValueError(f"Loại kiểm định không hợp lệ: {test_type!r}. Chọn: {valid}")
    missing = [p for p in TEST_PARAMS[test_type]["tham_so"] if p not in kwargs]  # type: ignore[union-attr]
    if missing:
        raise ValueError(f"{test_type}: thiếu tham số {missing}")


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _result(
    test_type: str,
    alpha: float,
    alternative: Alternative,
    statistic: float,
    statistic_name: str,
    p_value: float,
    critical_value: float | tuple[float, float] | None,
    critical_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gói kết quả và xác định reject_h0 theo phương pháp giá trị tới hạn."""
    reject = False
    if critical_value is not None:
        if alternative == "two-sided" and isinstance(critical_value, tuple):
            low, high = critical_value
            reject = statistic < low or statistic > high
        elif alternative == "greater":
            reject = statistic > critical_value  # type: ignore[operator]
        elif alternative == "less":
            reject = statistic < critical_value  # type: ignore[operator]

    out: dict[str, Any] = {
        "test_type": test_type,
        "alpha": alpha,
        "alternative": alternative,
        "statistic": float(statistic),
        "statistic_name": statistic_name,
        "p_value": float(p_value),
        "critical_value": critical_value,
        "critical_name": critical_name,
        "reject_h0": bool(reject),
    }
    if extra:
        out.update(extra)
    return out


def _z_critical(alpha: float, alternative: Alternative) -> float | tuple[float, float]:
    if alternative == "two-sided":
        z = float(stats.norm.ppf(1 - alpha / 2))
        return (-z, z)
    if alternative == "greater":
        return float(stats.norm.ppf(1 - alpha))
    return float(stats.norm.ppf(alpha))


def _t_critical(
    df: float, alpha: float, alternative: Alternative
) -> float | tuple[float, float]:
    if alternative == "two-sided":
        t = float(stats.t.ppf(1 - alpha / 2, df))
        return (-t, t)
    if alternative == "greater":
        return float(stats.t.ppf(1 - alpha, df))
    return float(stats.t.ppf(alpha, df))


def _z_pvalue(z: float, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return float(2 * stats.norm.sf(abs(z)))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    return float(stats.norm.cdf(z))


def _t_pvalue(t: float, df: float, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return float(2 * stats.t.sf(abs(t), df))
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    return float(stats.t.cdf(t, df))


# ---------------------------------------------------------------------------
# Chương 04_01 — có tham số
# ---------------------------------------------------------------------------


def _z_mean_sigma_known(
    x_bar: float,
    mu0: float,
    sigma: float,
    n: int,
    alpha: float,
    alternative: Alternative,
) -> dict[str, Any]:
    se = sigma / np.sqrt(n)
    z = (x_bar - mu0) / se
    return _result(
        "z_mean_sigma_known",
        alpha,
        alternative,
        z,
        "Z",
        _z_pvalue(z, alternative),
        _z_critical(alpha, alternative),
        "Z_alpha" if alternative != "two-sided" else "±Z_alpha/2",
        {"x_bar": x_bar, "mu0": mu0, "sigma": sigma, "n": n, "se": float(se)},
    )


def _t_mean_sigma_unknown(
    x_bar: float, mu0: float, s: float, n: int, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    se = s / np.sqrt(n)
    t = (x_bar - mu0) / se
    df = n - 1
    return _result(
        "t_mean_sigma_unknown",
        alpha,
        alternative,
        t,
        "T",
        _t_pvalue(t, df, alternative),
        _t_critical(df, alpha, alternative),
        f"t_{{alpha; {df}}}",
        {"x_bar": x_bar, "mu0": mu0, "s": s, "n": n, "df": df, "se": float(se)},
    )


def _z_proportion(
    f: int, n: int, p0: float, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    p_hat = f / n
    se = np.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / se
    return _result(
        "z_proportion",
        alpha,
        alternative,
        z,
        "Z",
        _z_pvalue(z, alternative),
        _z_critical(alpha, alternative),
        "Z_alpha" if alternative != "two-sided" else "±Z_alpha/2",
        {"f": f, "n": n, "p0": p0, "p_hat": float(p_hat), "se": float(se)},
    )


def _z_two_means_independent(
    x1: float,
    x2: float,
    sigma1: float,
    sigma2: float,
    n1: int,
    n2: int,
    alpha: float,
    alternative: Alternative,
    d0: float = 0.0,
) -> dict[str, Any]:
    se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
    z = ((x1 - x2) - d0) / se
    return _result(
        "z_two_means_independent",
        alpha,
        alternative,
        z,
        "Z",
        _z_pvalue(z, alternative),
        _z_critical(alpha, alternative),
        "Z_alpha" if alternative != "two-sided" else "±Z_alpha/2",
        {
            "x1": x1,
            "x2": x2,
            "d0": d0,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "n1": n1,
            "n2": n2,
            "se": float(se),
        },
    )


def _t_two_means_independent(
    x1: float,
    x2: float,
    s1: float,
    s2: float,
    n1: int,
    n2: int,
    alpha: float,
    alternative: Alternative,
    d0: float = 0.0,
) -> dict[str, Any]:
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    sp = np.sqrt(sp2)
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    t = ((x1 - x2) - d0) / se
    df = n1 + n2 - 2
    return _result(
        "t_two_means_independent",
        alpha,
        alternative,
        t,
        "T",
        _t_pvalue(t, df, alternative),
        _t_critical(df, alpha, alternative),
        f"t_{{alpha; {df}}}",
        {
            "x1": x1,
            "x2": x2,
            "d0": d0,
            "s1": s1,
            "s2": s2,
            "sp": float(sp),
            "n1": n1,
            "n2": n2,
            "df": df,
            "se": float(se),
        },
    )


def _t_two_means_welch(
    x1: float,
    x2: float,
    s1: float,
    s2: float,
    n1: int,
    n2: int,
    alpha: float,
    alternative: Alternative,
    d0: float = 0.0,
) -> dict[str, Any]:
    v1 = s1**2
    v2 = s2**2
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = ((x1 - x2) - d0) / se
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = float(num / den)
    return _result(
        "t_two_means_welch",
        alpha,
        alternative,
        t,
        "T",
        _t_pvalue(t, df, alternative),
        _t_critical(df, alpha, alternative),
        f"t_{{alpha; {df:.2f}}}",
        {
            "x1": x1,
            "x2": x2,
            "d0": d0,
            "s1": s1,
            "s2": s2,
            "s1_2": float(v1),
            "s2_2": float(v2),
            "n1": n1,
            "n2": n2,
            "df": df,
            "se": float(se),
        },
    )


def _t_paired(
    d_bar: float,
    sd: float,
    n: int,
    alpha: float,
    alternative: Alternative,
    d0: float = 0.0,
) -> dict[str, Any]:
    se = sd / np.sqrt(n)
    t = (d_bar - d0) / se
    df = n - 1
    return _result(
        "t_paired",
        alpha,
        alternative,
        t,
        "T",
        _t_pvalue(t, df, alternative),
        _t_critical(df, alpha, alternative),
        f"t_{{alpha; {df}}}",
        {"d_bar": d_bar, "d0": d0, "sd": sd, "n": n, "df": df, "se": float(se)},
    )


def _z_two_proportions(
    f1: int,
    n1: int,
    f2: int,
    n2: int,
    alpha: float,
    alternative: Alternative,
    p0: float = 0.0,
) -> dict[str, Any]:
    p1 = f1 / n1
    p2 = f2 / n2
    p_bar = (f1 + f2) / (n1 + n2)
    se = np.sqrt(p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
    z = ((p1 - p2) - p0) / se
    return _result(
        "z_two_proportions",
        alpha,
        alternative,
        z,
        "Z",
        _z_pvalue(z, alternative),
        _z_critical(alpha, alternative),
        "Z_alpha" if alternative != "two-sided" else "±Z_alpha/2",
        {
            "f1": f1,
            "n1": n1,
            "f2": f2,
            "n2": n2,
            "p0": p0,
            "p1": float(p1),
            "p2": float(p2),
            "p_bar": float(p_bar),
            "se": float(se),
        },
    )


def _chi2_variance(
    s2: float, sigma0_2: float, n: int, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    chi2 = (n - 1) * s2 / sigma0_2
    df = n - 1
    if alternative == "two-sided":
        low = float(stats.chi2.ppf(alpha / 2, df))
        high = float(stats.chi2.ppf(1 - alpha / 2, df))
        crit: float | tuple[float, float] = (low, high)
        p_value = float(2 * min(stats.chi2.cdf(chi2, df), stats.chi2.sf(chi2, df)))
    elif alternative == "greater":
        crit = float(stats.chi2.ppf(1 - alpha, df))
        p_value = float(stats.chi2.sf(chi2, df))
    else:
        crit = float(stats.chi2.ppf(alpha, df))
        p_value = float(stats.chi2.cdf(chi2, df))
    return _result(
        "chi2_variance",
        alpha,
        alternative,
        chi2,
        "chi2",
        p_value,
        crit,
        "chi2_alpha",
        {"s2": s2, "sigma0_2": sigma0_2, "n": n, "df": df},
    )


def _f_two_variances(
    s1_2: float,
    s2_2: float,
    n1: int,
    n2: int,
    alpha: float,
    alternative: Alternative,
) -> dict[str, Any]:
    f = s1_2 / s2_2
    df1, df2 = n1 - 1, n2 - 1
    if alternative == "two-sided":
        low = float(stats.f.ppf(alpha / 2, df1, df2))
        high = float(stats.f.ppf(1 - alpha / 2, df1, df2))
        crit = (low, high)
        p_value = float(2 * min(stats.f.cdf(f, df1, df2), stats.f.sf(f, df1, df2)))
    elif alternative == "greater":
        crit = float(stats.f.ppf(1 - alpha, df1, df2))
        p_value = float(stats.f.sf(f, df1, df2))
    else:
        crit = float(stats.f.ppf(alpha, df1, df2))
        p_value = float(stats.f.cdf(f, df1, df2))
    return _result(
        "f_two_variances",
        alpha,
        alternative,
        f,
        "F",
        p_value,
        crit,
        "F_alpha",
        {"s1_2": s1_2, "s2_2": s2_2, "n1": n1, "n2": n2, "df1": df1, "df2": df2},
    )


# ---------------------------------------------------------------------------
# Chương 04_02 — phi tham số
# ---------------------------------------------------------------------------


def _kruskal_wallis(
    groups: list, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    arrays = [_as_array(g) for g in groups]
    k = len(arrays)
    df = k - 1
    h, p_value = stats.kruskal(*arrays)
    crit = float(stats.chi2.ppf(1 - alpha, df))
    # Kruskal-Wallis mặc định hai phía: H > chi2_{alpha; k-1}
    return _result(
        "kruskal_wallis",
        alpha,
        "two-sided",
        float(h),
        "H",
        float(p_value),
        crit,
        f"chi2_{{alpha; {df}}}",
        {"k": k, "df": df, "group_sizes": [len(a) for a in arrays]},
    )


def _sign_test(
    differences: list, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    diffs = _as_array(differences)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    n_pos = int(np.sum(diffs > 0))
    n_neg = n - n_pos
    if alternative == "two-sided":
        p_value = float(
            2 * min(stats.binom.cdf(n_pos, n, 0.5), stats.binom.sf(n_pos - 1, n, 0.5))
        )
    elif alternative == "greater":
        p_value = float(stats.binom.sf(n_pos - 1, n, 0.5))
    else:
        p_value = float(stats.binom.cdf(n_pos, n, 0.5))
    return _result(
        "sign_test",
        alpha,
        alternative,
        float(n_pos),
        "n_positive",
        p_value,
        None,
        "binomial",
        {"n": n, "n_positive": n_pos, "n_negative": n_neg},
    )


def _wilcoxon_signed_rank(
    x: list, y: list, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    x_arr, y_arr = _as_array(x), _as_array(y)
    stat, p_value = stats.wilcoxon(x_arr, y_arr, alternative=alternative)
    return _result(
        "wilcoxon_signed_rank",
        alpha,
        alternative,
        float(stat),
        "T",
        float(p_value),
        None,
        "tra_bang_Wilcoxon",
        {"n_pairs": len(x_arr)},
    )


def _wilcoxon_rank_sum(
    sample1: list, sample2: list, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    s1, s2 = _as_array(sample1), _as_array(sample2)
    n1, n2 = len(s1), len(s2)
    # use_continuity=False: giáo trình (Chapter_04_02) chỉ dạy hiệu chỉnh đồng hạng,
    # không dạy hiệu chỉnh liên tục. Để mặc định True thì p_value đến từ một công thức
    # khác với Z tính bên dưới, và hai con số báo cáo ra sẽ không tương thích nhau.
    u, p_value = stats.mannwhitneyu(
        s1, s2, alternative=alternative, use_continuity=False
    )
    mu_u = n1 * n2 / 2.0
    # Hiệu chỉnh đồng hạng. Giáo trình nêu rõ công thức n1·n2·(N+1)/12 chỉ đúng "khi tần
    # số xuất hiện của các giá trị đều = 1", và cho công thức riêng với t_j là kích thước
    # nhóm đồng hạng thứ j. Không có đồng hạng thì số hạng hiệu chỉnh bằng 0 và biểu thức
    # thu về đúng dạng quen thuộc, nên nhánh này an toàn cho mọi dữ liệu.
    n = n1 + n2
    _, tie_sizes = np.unique(np.concatenate([s1, s2]), return_counts=True)
    tie_term = float(np.sum(tie_sizes**3 - tie_sizes))
    sigma_u = float(np.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))))
    z = (float(u) - mu_u) / sigma_u
    use_z_approx = n1 > WILCOXON_RANK_SUM_Z_MIN_N and n2 > WILCOXON_RANK_SUM_Z_MIN_N

    if use_z_approx:
        z_crit = _z_critical(alpha, alternative)
        if alternative == "two-sided":
            assert isinstance(z_crit, tuple)
            z_threshold = z_crit[1]
            critical_name = "Z_alpha/2"
        else:
            z_threshold = float(z_crit)  # type: ignore[arg-type]
            critical_name = "Z_alpha"
        # Bước 3 (mẫu lớn): 1 phía |Z| > Z_α; 2 phía |Z| > Z_{α/2}
        reject = abs(z) > z_threshold
        method = "z_approx"
    else:
        z_crit = None
        z_threshold = None
        critical_name = "bang_W"
        reject = float(p_value) < alpha
        method = "exact"

    out = _result(
        "wilcoxon_rank_sum",
        alpha,
        alternative,
        float(u),
        "U",
        float(p_value),
        z_threshold,
        critical_name,
        {
            "n1": n1,
            "n2": n2,
            "mu_u": mu_u,
            "sigma_u": sigma_u,
            "z": float(z),
            "method": method,
            "use_z_approx": use_z_approx,
        },
    )
    out["reject_h0"] = bool(reject)
    if z_crit is not None:
        out["z_critical"] = z_crit
    return out


def _chi2_independence(
    observed: list, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    table = np.asarray(observed, dtype=float)
    chi2, p_value, df, expected = stats.chi2_contingency(table)
    crit = float(stats.chi2.ppf(1 - alpha, df))
    return _result(
        "chi2_independence",
        alpha,
        "two-sided",
        float(chi2),
        "chi2",
        float(p_value),
        crit,
        f"chi2_{{alpha; {df}}}",
        {"df": int(df), "expected": expected.tolist()},
    )


def _chi2_goodness_of_fit(
    observed: list, expected: list, alpha: float, alternative: Alternative
) -> dict[str, Any]:
    obs = _as_array(observed)
    exp = _as_array(expected)
    chi2, p_value = stats.chisquare(obs, f_exp=exp)
    df = len(obs) - 1
    crit = float(stats.chi2.ppf(1 - alpha, df))
    return _result(
        "chi2_goodness_of_fit",
        alpha,
        "two-sided",
        float(chi2),
        "chi2",
        float(p_value),
        crit,
        f"chi2_{{alpha; {df}}}",
        {"df": df, "observed": obs.tolist(), "expected": exp.tolist()},
    )


# ---------------------------------------------------------------------------
# Hàm điều phối
# ---------------------------------------------------------------------------

_HANDLERS = {
    "z_mean_sigma_known": _z_mean_sigma_known,
    "t_mean_sigma_unknown": _t_mean_sigma_unknown,
    "z_proportion": _z_proportion,
    "z_two_means_independent": _z_two_means_independent,
    "t_two_means_independent": _t_two_means_independent,
    "t_two_means_welch": _t_two_means_welch,
    "t_paired": _t_paired,
    "z_two_proportions": _z_two_proportions,
    "chi2_variance": _chi2_variance,
    "f_two_variances": _f_two_variances,
    "kruskal_wallis": _kruskal_wallis,
    "sign_test": _sign_test,
    "wilcoxon_signed_rank": _wilcoxon_signed_rank,
    "wilcoxon_rank_sum": _wilcoxon_rank_sum,
    "chi2_independence": _chi2_independence,
    "chi2_goodness_of_fit": _chi2_goodness_of_fit,
}


def hypothesis_test(
    test_type: str,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Tính số liệu kiểm định theo loại kiểm định và tham số truyền vào.

    Parameters
    ----------
    test_type : str
        Loại kiểm định (xem ``list_test_types()``).
    alpha : float
        Mức ý nghĩa (mặc định 0.05).
    alternative : str
        ``"greater"`` | ``"less"`` | ``"two-sided"`` (một số kiểm định phi tham số
        cố định hai phía).
    **kwargs
        Tham số theo từng loại kiểm định.

    Returns
    -------
    dict
        statistic, p_value, critical_value, reject_h0, ...

    Examples
    --------
    >>> hypothesis_test(
    ...     "t_mean_sigma_unknown",
    ...     x_bar=2020, mu0=2000, s=92, n=36,
    ...     alpha=0.10, alternative="greater",
    ... )

    >>> hypothesis_test(
    ...     "kruskal_wallis",
    ...     groups=[[45.5, 44.5], [45.8, 45.1], [43.7, 44.0]],
    ... )
    """
    test_type = test_type.strip()
    _check_params(test_type, kwargs)
    handler = _HANDLERS[test_type]
    return handler(alpha=alpha, alternative=alternative, **kwargs)
