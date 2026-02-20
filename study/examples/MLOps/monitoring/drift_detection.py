"""
Drift Detection Example
=======================

Evidently AI를 사용한 데이터 드리프트 감지 예제입니다.

실행 방법:
    pip install evidently pandas numpy scikit-learn
    python drift_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Evidently 임포트 (선택적)
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Evidently가 설치되지 않았습니다. 기본 통계 방법을 사용합니다.")


# ============================================================
# 기본 통계적 드리프트 감지
# ============================================================

class StatisticalDriftDetector:
    """통계적 방법을 사용한 드리프트 감지"""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Kolmogorov-Smirnov 검정"""
        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            "test": "ks",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < self.significance_level
        }

    def psi(self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index"""
        # 히스토그램 생성
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # 비율로 변환
        ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
        cur_pct = (cur_counts + 1) / (len(current) + n_bins)

        # PSI 계산
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def wasserstein_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Wasserstein 거리"""
        return float(stats.wasserstein_distance(reference, current))

    def detect_column_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column: str
    ) -> Dict[str, Any]:
        """단일 컬럼 드리프트 감지"""
        ref_col = reference[column].dropna().values
        cur_col = current[column].dropna().values

        ks_result = self.ks_test(ref_col, cur_col)
        psi_value = self.psi(ref_col, cur_col)
        wasserstein = self.wasserstein_distance(ref_col, cur_col)

        # PSI 해석
        if psi_value < 0.1:
            psi_status = "no_drift"
        elif psi_value < 0.2:
            psi_status = "slight_drift"
        else:
            psi_status = "significant_drift"

        return {
            "column": column,
            "ks_test": ks_result,
            "psi": {
                "value": psi_value,
                "status": psi_status
            },
            "wasserstein_distance": wasserstein,
            "drift_detected": ks_result["drift_detected"] or psi_value >= 0.2
        }

    def detect_dataset_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numerical_columns: list
    ) -> Dict[str, Any]:
        """데이터셋 전체 드리프트 감지"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "columns": {},
            "summary": {
                "total_columns": len(numerical_columns),
                "drifted_columns": 0,
                "drift_detected": False
            }
        }

        for col in numerical_columns:
            if col in reference.columns and col in current.columns:
                col_result = self.detect_column_drift(reference, current, col)
                results["columns"][col] = col_result
                if col_result["drift_detected"]:
                    results["summary"]["drifted_columns"] += 1

        drift_share = results["summary"]["drifted_columns"] / results["summary"]["total_columns"]
        results["summary"]["drift_share"] = drift_share
        results["summary"]["drift_detected"] = drift_share > 0.5

        return results


# ============================================================
# Evidently 기반 드리프트 감지
# ============================================================

class EvidentlyDriftDetector:
    """Evidently AI를 사용한 드리프트 감지"""

    def __init__(self):
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently is not installed")

    def create_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: ColumnMapping = None
    ) -> Report:
        """드리프트 리포트 생성"""
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftPreset()
        ])

        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping
        )

        return report

    def run_tests(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: ColumnMapping = None
    ) -> TestSuite:
        """드리프트 테스트 실행"""
        test_suite = TestSuite(tests=[
            DataDriftTestPreset()
        ])

        test_suite.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping
        )

        return test_suite

    def get_drift_summary(self, report: Report) -> Dict[str, Any]:
        """리포트에서 드리프트 요약 추출"""
        result = report.as_dict()

        # DatasetDriftMetric 결과 추출
        for metric in result.get("metrics", []):
            if "DatasetDriftMetric" in str(metric.get("metric", "")):
                drift_result = metric.get("result", {})
                return {
                    "dataset_drift": drift_result.get("dataset_drift", False),
                    "drift_share": drift_result.get("drift_share", 0),
                    "number_of_columns": drift_result.get("number_of_columns", 0),
                    "number_of_drifted_columns": drift_result.get("number_of_drifted_columns", 0)
                }

        return {"error": "Could not extract drift summary"}


# ============================================================
# 드리프트 모니터링 시스템
# ============================================================

class DriftMonitor:
    """드리프트 모니터링 시스템"""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_columns: list,
        alert_threshold: float = 0.3
    ):
        self.reference_data = reference_data
        self.numerical_columns = numerical_columns
        self.alert_threshold = alert_threshold
        self.detector = StatisticalDriftDetector()
        self.history = []

    def check(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """드리프트 체크"""
        result = self.detector.detect_dataset_drift(
            self.reference_data,
            current_data,
            self.numerical_columns
        )

        # 히스토리에 추가
        self.history.append({
            "timestamp": result["timestamp"],
            "drift_share": result["summary"]["drift_share"],
            "drift_detected": result["summary"]["drift_detected"]
        })

        # 알림 생성
        result["alerts"] = self._generate_alerts(result)

        return result

    def _generate_alerts(self, result: Dict) -> list:
        """알림 생성"""
        alerts = []

        if result["summary"]["drift_detected"]:
            alerts.append({
                "level": "critical",
                "message": f"Dataset drift detected: {result['summary']['drift_share']:.1%} of columns drifted"
            })

        for col, col_result in result["columns"].items():
            if col_result["drift_detected"]:
                psi = col_result["psi"]["value"]
                if psi >= 0.25:
                    alerts.append({
                        "level": "warning",
                        "message": f"Significant drift in '{col}': PSI={psi:.3f}"
                    })

        return alerts

    def get_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """드리프트 트렌드 분석"""
        if len(self.history) < 2:
            return {"message": "Not enough data for trend analysis"}

        recent = self.history[-window_size:]
        drift_shares = [h["drift_share"] for h in recent]

        return {
            "window_size": len(recent),
            "avg_drift_share": np.mean(drift_shares),
            "max_drift_share": max(drift_shares),
            "drift_count": sum(1 for h in recent if h["drift_detected"]),
            "trend": "increasing" if len(drift_shares) > 1 and drift_shares[-1] > drift_shares[0] else "stable"
        }


# ============================================================
# 예제 실행
# ============================================================

def generate_sample_data(n_samples: int = 1000, drift: bool = False) -> pd.DataFrame:
    """샘플 데이터 생성"""
    np.random.seed(42 if not drift else 123)

    data = {
        "feature_1": np.random.normal(0, 1, n_samples),
        "feature_2": np.random.normal(5, 2, n_samples),
        "feature_3": np.random.exponential(2, n_samples),
        "feature_4": np.random.uniform(0, 10, n_samples)
    }

    if drift:
        # 일부 피처에 드리프트 추가
        data["feature_1"] = np.random.normal(0.5, 1.2, n_samples)  # 평균, 분산 변화
        data["feature_3"] = np.random.exponential(3, n_samples)    # 분포 변화

    return pd.DataFrame(data)


def main():
    """메인 실행 함수"""
    print("="*60)
    print("드리프트 감지 예제")
    print("="*60)

    # 1. 데이터 생성
    print("\n[1] 데이터 생성...")
    reference_data = generate_sample_data(1000, drift=False)
    current_data_no_drift = generate_sample_data(500, drift=False)
    current_data_with_drift = generate_sample_data(500, drift=True)

    print(f"  참조 데이터: {len(reference_data)} 샘플")
    print(f"  현재 데이터 (드리프트 없음): {len(current_data_no_drift)} 샘플")
    print(f"  현재 데이터 (드리프트 있음): {len(current_data_with_drift)} 샘플")

    # 2. 기본 통계적 드리프트 감지
    print("\n[2] 통계적 드리프트 감지...")
    detector = StatisticalDriftDetector()
    numerical_cols = ["feature_1", "feature_2", "feature_3", "feature_4"]

    # 드리프트 없는 데이터
    print("\n  --- 드리프트 없는 데이터 ---")
    result_no_drift = detector.detect_dataset_drift(
        reference_data, current_data_no_drift, numerical_cols
    )
    print(f"  드리프트 감지: {result_no_drift['summary']['drift_detected']}")
    print(f"  드리프트 비율: {result_no_drift['summary']['drift_share']:.1%}")

    # 드리프트 있는 데이터
    print("\n  --- 드리프트 있는 데이터 ---")
    result_with_drift = detector.detect_dataset_drift(
        reference_data, current_data_with_drift, numerical_cols
    )
    print(f"  드리프트 감지: {result_with_drift['summary']['drift_detected']}")
    print(f"  드리프트 비율: {result_with_drift['summary']['drift_share']:.1%}")

    # 컬럼별 상세 결과
    print("\n  컬럼별 상세:")
    for col, col_result in result_with_drift["columns"].items():
        drift_status = "DRIFT" if col_result["drift_detected"] else "OK"
        psi = col_result["psi"]["value"]
        print(f"    {col}: PSI={psi:.4f} [{drift_status}]")

    # 3. Evidently 기반 감지 (설치된 경우)
    if EVIDENTLY_AVAILABLE:
        print("\n[3] Evidently 드리프트 감지...")
        evidently_detector = EvidentlyDriftDetector()

        report = evidently_detector.create_report(
            reference_data, current_data_with_drift
        )

        summary = evidently_detector.get_drift_summary(report)
        print(f"  Dataset Drift: {summary.get('dataset_drift', 'N/A')}")
        print(f"  Drift Share: {summary.get('drift_share', 0):.1%}")
        print(f"  Drifted Columns: {summary.get('number_of_drifted_columns', 0)}/{summary.get('number_of_columns', 0)}")

        # HTML 리포트 저장
        report.save_html("drift_report.html")
        print("\n  HTML 리포트 저장: drift_report.html")
    else:
        print("\n[3] Evidently 설치 필요 (pip install evidently)")

    # 4. 모니터링 시스템 시뮬레이션
    print("\n[4] 모니터링 시스템 시뮬레이션...")
    monitor = DriftMonitor(
        reference_data=reference_data,
        numerical_columns=numerical_cols
    )

    # 여러 시점 데이터로 체크
    for i in range(5):
        # 시간이 지남에 따라 점진적 드리프트
        drift_factor = i * 0.1
        test_data = reference_data.copy()
        test_data["feature_1"] = test_data["feature_1"] + drift_factor
        test_data["feature_3"] = test_data["feature_3"] * (1 + drift_factor)

        result = monitor.check(test_data.sample(500))
        print(f"\n  시점 {i+1}:")
        print(f"    드리프트 감지: {result['summary']['drift_detected']}")
        print(f"    드리프트 비율: {result['summary']['drift_share']:.1%}")
        if result["alerts"]:
            for alert in result["alerts"]:
                print(f"    [{alert['level'].upper()}] {alert['message']}")

    # 트렌드 분석
    trend = monitor.get_trend()
    print(f"\n  트렌드 분석:")
    print(f"    평균 드리프트 비율: {trend['avg_drift_share']:.1%}")
    print(f"    드리프트 감지 횟수: {trend['drift_count']}")
    print(f"    트렌드: {trend['trend']}")

    print("\n" + "="*60)
    print("예제 완료!")


if __name__ == "__main__":
    main()
