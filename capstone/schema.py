"""Explicit feature allowlist and immutable study boundaries."""

SOURCE_REPO = "https://github.com/wplso/xArm-model-training"
SOURCE_COMMIT = "2fdaa693e3aaafe3a221f5a775a12bc340f4d562"
ROOTS = ("원본 수집 데이터/V4_V51_보완원시", "원본 수집 데이터/볼트_하중_보완원시")
CLASSES = ("normal", "A121", "B82", "C182")
LABELS = dict(zip(("normal", "A", "B", "C"), CLASSES))
DEV_DATES = ("2026-11-06", "2026-11-10", "2026-11-11")
TEST_DATE = "2026-11-12"
BASE_FEATURES = (
    "current_A_mean", "current_A_rms", "current_A_std", "current_A_abs_p95_nearest_rank",
    "voltage_V_mean", "voltage_V_std", "x_g_std", "y_g_std", "z_g_std",
    "mean_removed_vector_g_rms", "mean_removed_vector_g_abs_peak",
    *(f"position_ID{i}_raw_{stat}" for i in range(1, 7) for stat in ("range", "std")),
    "ID1_3_5s_IQR",
)
PHASES = {"S3": (3.0, 4.0, 8), "S4": (4.0, 5.0, 8), "S5": (5.0, 6.5, 12)}
PHASE_FEATURES = tuple(f"{phase}_current_mean" for phase in PHASES)
FEATURE_SETS = {"base24": BASE_FEATURES, "phase27": BASE_FEATURES + PHASE_FEATURES}
OPTIONAL_FEATURES = frozenset(("ID1_3_5s_IQR", *PHASE_FEATURES))
REQUIRED_FEATURES = tuple(f for f in BASE_FEATURES if f not in OPTIONAL_FEATURES)
EXPECTED_COUNTS = {
    "2026-11-06": {"normal": 479, "A121": 30, "B82": 30, "C182": 30},
    "2026-11-10": {"normal": 80},
    "2026-11-11": {"normal": 330, "A121": 30, "B82": 30, "C182": 30},
    "2026-11-12": {"normal": 30, "B82": 30, "C182": 30},
}
REFERENCE_FEATURE_FILES = (
    "models/v5.1/CYCLE_FEATURES.csv",
    "models/bolt-known-load/abc-interleaved/frozen_evaluation/CYCLE_FEATURES.csv",
    "models/bolt-known-load/retrain/retrained_bolt_normal/NEW_NORMAL_FEATURES.csv",
    "models/bolt-known-load/bc-independent/CYCLE_FEATURES.csv",
)
FROZEN_MODELS = {
    "frozen_v4": "models/bolt-known-load/prospective/models/V4_PCA_Q.joblib",
    "frozen_logistic": "models/bolt-known-load/retrain/retrained_bolt_normal/models/LOGISTIC_BOLT_NORMAL_EXPANDED.joblib",
}
