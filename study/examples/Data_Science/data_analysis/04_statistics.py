"""
통계 분석 (Statistical Analysis)
Basic Statistical Analysis with Python

기초 통계 분석 방법을 다룹니다.
"""

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 1. 기술 통계 (Descriptive Statistics)
# =============================================================================
def descriptive_stats():
    """기술 통계량"""
    print("\n[1] 기술 통계량")
    print("=" * 50)

    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)  # 평균 100, 표준편차 15

    print(f"데이터 크기: {len(data)}")
    print(f"\n중심 경향:")
    print(f"  평균 (Mean):    {np.mean(data):.2f}")
    print(f"  중앙값 (Median): {np.median(data):.2f}")
    print(f"  최빈값 (Mode):   {stats.mode(data.round(), keepdims=False).mode:.2f}")

    print(f"\n산포도:")
    print(f"  분산 (Variance): {np.var(data, ddof=1):.2f}")
    print(f"  표준편차 (Std):  {np.std(data, ddof=1):.2f}")
    print(f"  범위 (Range):    {np.ptp(data):.2f}")
    print(f"  IQR:             {stats.iqr(data):.2f}")

    print(f"\n분위수:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(data, p):.2f}")

    print(f"\n형태:")
    print(f"  왜도 (Skewness):  {stats.skew(data):.4f}")
    print(f"  첨도 (Kurtosis):  {stats.kurtosis(data):.4f}")


# =============================================================================
# 2. 상관 분석 (Correlation Analysis)
# =============================================================================
def correlation_analysis():
    """상관 분석"""
    print("\n[2] 상관 분석")
    print("=" * 50)

    np.random.seed(42)
    n = 100

    # 상관된 데이터 생성
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5  # 강한 양의 상관
    z = -0.5 * x + np.random.randn(n)     # 약한 음의 상관
    w = np.random.randn(n)                 # 무상관

    # 피어슨 상관계수
    print("피어슨 상관계수 (Pearson):")
    corr_xy, p_xy = stats.pearsonr(x, y)
    corr_xz, p_xz = stats.pearsonr(x, z)
    corr_xw, p_xw = stats.pearsonr(x, w)

    print(f"  x-y: r = {corr_xy:.4f}, p = {p_xy:.4e}")
    print(f"  x-z: r = {corr_xz:.4f}, p = {p_xz:.4e}")
    print(f"  x-w: r = {corr_xw:.4f}, p = {p_xw:.4e}")

    # 스피어만 순위 상관계수
    print("\n스피어만 순위 상관계수 (Spearman):")
    corr_s, p_s = stats.spearmanr(x, y)
    print(f"  x-y: ρ = {corr_s:.4f}, p = {p_s:.4e}")

    # DataFrame 상관 행렬
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'w': w})
    print("\n상관 행렬:")
    print(df.corr().round(4))


# =============================================================================
# 3. 가설 검정 기초
# =============================================================================
def hypothesis_testing():
    """가설 검정"""
    print("\n[3] 가설 검정 기초")
    print("=" * 50)

    np.random.seed(42)

    # 단일 표본 t-검정
    print("\n[단일 표본 t-검정]")
    sample = np.random.normal(105, 15, 50)  # 실제 평균 105
    t_stat, p_value = stats.ttest_1samp(sample, 100)  # H0: μ = 100

    print(f"표본 평균: {np.mean(sample):.2f}")
    print(f"H0: μ = 100")
    print(f"t-통계량: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"결론: {'H0 기각' if p_value < 0.05 else 'H0 채택'} (α=0.05)")

    # 독립 표본 t-검정
    print("\n[독립 표본 t-검정]")
    group1 = np.random.normal(100, 10, 50)
    group2 = np.random.normal(105, 10, 50)

    t_stat, p_value = stats.ttest_ind(group1, group2)

    print(f"그룹1 평균: {np.mean(group1):.2f}")
    print(f"그룹2 평균: {np.mean(group2):.2f}")
    print(f"H0: μ1 = μ2")
    print(f"t-통계량: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"결론: {'H0 기각' if p_value < 0.05 else 'H0 채택'} (α=0.05)")

    # 대응 표본 t-검정
    print("\n[대응 표본 t-검정]")
    before = np.random.normal(100, 10, 30)
    after = before + np.random.normal(5, 3, 30)  # 평균 5 증가

    t_stat, p_value = stats.ttest_rel(before, after)

    print(f"사전 평균: {np.mean(before):.2f}")
    print(f"사후 평균: {np.mean(after):.2f}")
    print(f"H0: μ_차이 = 0")
    print(f"t-통계량: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"결론: {'H0 기각' if p_value < 0.05 else 'H0 채택'} (α=0.05)")


# =============================================================================
# 4. 카이제곱 검정
# =============================================================================
def chi_square_test():
    """카이제곱 검정"""
    print("\n[4] 카이제곱 검정")
    print("=" * 50)

    # 적합도 검정
    print("\n[적합도 검정]")
    observed = np.array([18, 22, 20, 15, 25])  # 관측 빈도
    expected = np.array([20, 20, 20, 20, 20])  # 기대 빈도

    chi2, p_value = stats.chisquare(observed, expected)

    print(f"관측값: {observed}")
    print(f"기대값: {expected}")
    print(f"χ² = {chi2:.4f}")
    print(f"p-value = {p_value:.4f}")
    print(f"결론: {'분포 다름' if p_value < 0.05 else '분포 같음'}")

    # 독립성 검정
    print("\n[독립성 검정 (교차표)]")
    contingency_table = np.array([
        [30, 20, 10],  # 그룹 A
        [15, 25, 20],  # 그룹 B
        [25, 15, 25]   # 그룹 C
    ])

    print("교차표:")
    print(contingency_table)

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\nχ² = {chi2:.4f}")
    print(f"자유도 = {dof}")
    print(f"p-value = {p_value:.4f}")
    print(f"결론: {'독립 아님' if p_value < 0.05 else '독립'}")


# =============================================================================
# 5. 분산 분석 (ANOVA)
# =============================================================================
def anova_test():
    """ANOVA"""
    print("\n[5] 분산 분석 (ANOVA)")
    print("=" * 50)

    np.random.seed(42)

    # 세 그룹 데이터
    group1 = np.random.normal(100, 10, 30)
    group2 = np.random.normal(105, 10, 30)
    group3 = np.random.normal(110, 10, 30)

    print(f"그룹1 평균: {np.mean(group1):.2f}")
    print(f"그룹2 평균: {np.mean(group2):.2f}")
    print(f"그룹3 평균: {np.mean(group3):.2f}")

    # 일원 분산 분석
    f_stat, p_value = stats.f_oneway(group1, group2, group3)

    print(f"\n일원 분산 분석 (One-way ANOVA)")
    print(f"H0: μ1 = μ2 = μ3")
    print(f"F-통계량: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"결론: {'그룹 간 차이 있음' if p_value < 0.05 else '그룹 간 차이 없음'}")

    # Kruskal-Wallis (비모수)
    print("\n[비모수 검정: Kruskal-Wallis]")
    h_stat, p_value = stats.kruskal(group1, group2, group3)
    print(f"H-통계량: {h_stat:.4f}")
    print(f"p-value: {p_value:.4f}")


# =============================================================================
# 6. 정규성 검정
# =============================================================================
def normality_test():
    """정규성 검정"""
    print("\n[6] 정규성 검정")
    print("=" * 50)

    np.random.seed(42)

    # 정규 분포 데이터
    normal_data = np.random.normal(0, 1, 100)

    # 비정규 분포 데이터 (지수 분포)
    skewed_data = np.random.exponential(2, 100)

    print("[정규 분포 데이터]")
    print(f"왜도: {stats.skew(normal_data):.4f}")
    print(f"첨도: {stats.kurtosis(normal_data):.4f}")

    # Shapiro-Wilk 검정
    stat, p = stats.shapiro(normal_data)
    print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")

    # Kolmogorov-Smirnov 검정
    stat, p = stats.kstest(normal_data, 'norm')
    print(f"K-S 검정: D={stat:.4f}, p={p:.4f}")

    print(f"\n결론: {'정규 분포' if p > 0.05 else '정규 분포 아님'}")

    print("\n[비정규 분포 데이터 (지수 분포)]")
    print(f"왜도: {stats.skew(skewed_data):.4f}")
    print(f"첨도: {stats.kurtosis(skewed_data):.4f}")

    stat, p = stats.shapiro(skewed_data)
    print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")
    print(f"결론: {'정규 분포' if p > 0.05 else '정규 분포 아님'}")


# =============================================================================
# 7. 신뢰 구간
# =============================================================================
def confidence_interval():
    """신뢰 구간"""
    print("\n[7] 신뢰 구간")
    print("=" * 50)

    np.random.seed(42)
    sample = np.random.normal(100, 15, 50)

    mean = np.mean(sample)
    sem = stats.sem(sample)  # 표준 오차

    # 95% 신뢰 구간
    ci_95 = stats.t.interval(0.95, len(sample)-1, loc=mean, scale=sem)
    ci_99 = stats.t.interval(0.99, len(sample)-1, loc=mean, scale=sem)

    print(f"표본 크기: {len(sample)}")
    print(f"표본 평균: {mean:.2f}")
    print(f"표준 오차: {sem:.2f}")
    print(f"\n95% 신뢰 구간: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")
    print(f"99% 신뢰 구간: ({ci_99[0]:.2f}, {ci_99[1]:.2f})")

    # 비율의 신뢰 구간
    print("\n[비율의 신뢰 구간]")
    n_success = 70
    n_total = 100
    p_hat = n_success / n_total
    se_p = np.sqrt(p_hat * (1 - p_hat) / n_total)
    z_95 = 1.96

    ci_low = p_hat - z_95 * se_p
    ci_high = p_hat + z_95 * se_p

    print(f"성공: {n_success}/{n_total} = {p_hat:.2f}")
    print(f"95% 신뢰 구간: ({ci_low:.4f}, {ci_high:.4f})")


# =============================================================================
# 8. 효과 크기
# =============================================================================
def effect_size():
    """효과 크기 계산"""
    print("\n[8] 효과 크기")
    print("=" * 50)

    np.random.seed(42)

    group1 = np.random.normal(100, 15, 50)
    group2 = np.random.normal(110, 15, 50)

    # Cohen's d
    def cohens_d(g1, g2):
        n1, n2 = len(g1), len(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(g1) - np.mean(g2)) / pooled_std

    d = cohens_d(group2, group1)

    print(f"그룹1 평균: {np.mean(group1):.2f}")
    print(f"그룹2 평균: {np.mean(group2):.2f}")
    print(f"\nCohen's d: {d:.4f}")

    # 효과 크기 해석
    if abs(d) < 0.2:
        interpretation = "효과 없음"
    elif abs(d) < 0.5:
        interpretation = "작은 효과"
    elif abs(d) < 0.8:
        interpretation = "중간 효과"
    else:
        interpretation = "큰 효과"

    print(f"해석: {interpretation}")

    # 상관계수를 효과 크기로
    x = np.random.randn(100)
    y = 0.5 * x + np.random.randn(100) * 0.5
    r, _ = stats.pearsonr(x, y)
    r_squared = r ** 2

    print(f"\n상관계수 r: {r:.4f}")
    print(f"결정계수 r²: {r_squared:.4f}")
    print(f"(설명 분산 비율: {r_squared*100:.1f}%)")


# =============================================================================
# 메인
# =============================================================================
def main():
    print("=" * 60)
    print("통계 분석 예제")
    print("=" * 60)

    descriptive_stats()
    correlation_analysis()
    hypothesis_testing()
    chi_square_test()
    anova_test()
    normality_test()
    confidence_interval()
    effect_size()

    print("\n" + "=" * 60)
    print("통계 분석 핵심 정리")
    print("=" * 60)
    print("""
    기술 통계:
    - 중심: 평균, 중앙값, 최빈값
    - 산포: 분산, 표준편차, IQR
    - 형태: 왜도, 첨도

    추론 통계:
    - t-검정: 평균 비교 (1표본, 독립, 대응)
    - ANOVA: 3개 이상 그룹 평균 비교
    - 카이제곱: 범주형 변수 관계
    - 상관분석: 연속형 변수 관계

    가설 검정 절차:
    1. 귀무가설(H0)과 대립가설(H1) 설정
    2. 유의수준 결정 (보통 α=0.05)
    3. 검정 통계량 계산
    4. p-value와 유의수준 비교
    5. 결론 도출

    p-value 해석:
    - p < 0.05: 통계적으로 유의함
    - p ≥ 0.05: 통계적으로 유의하지 않음

    주의:
    - 통계적 유의성 ≠ 실질적 중요성
    - 효과 크기도 함께 보고
    - 다중 비교 시 보정 필요 (Bonferroni 등)
    """)


if __name__ == "__main__":
    main()
