"""Model definitions and explicit inference; fitting is only called by train."""

import json
import math
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .schema import CLASSES, FEATURE_SETS, OPTIONAL_FEATURES


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name("experiment.json")).read_text(encoding="utf-8"))


def candidates(config):
    result = []
    for feature_set in config["feature_sets"]:
        if feature_set not in FEATURE_SETS:
            raise ValueError("Unknown feature allowlist")
        for c in config["logistic"]["C"]:
            result.append({"id": f"logistic_{feature_set}_C{c:g}", "kind": "logistic", "feature_set": feature_set,
                           "parameters": {"C": c, "max_iter": config["logistic"]["max_iter"],
                                          "solver": "lbfgs", "class_weight": "balanced", "random_state": config["seed"]}})
        for leaves in config["boosting"]["max_leaf_nodes"]:
            params = {k: v for k, v in config["boosting"].items() if k != "max_leaf_nodes"}
            if params["early_stopping"] is not False:
                raise ValueError("Automatic within-date validation is forbidden")
            params.update(max_leaf_nodes=leaves, class_weight="balanced", random_state=config["seed"])
            result.append({"id": f"boosting_{feature_set}_L{leaves}", "kind": "boosting", "feature_set": feature_set,
                           "parameters": params})
    if "logistic_base24_C1" not in {c["id"] for c in result}:
        raise ValueError("The matched logistic base24 C=1 baseline is required")
    return result


def input_quality(frame, features):
    if "cycle_id" not in frame or frame.empty:
        raise ValueError("Expected nonempty input with cycle_id")
    if frame.columns.duplicated().any() or frame.cycle_id.isna().any() or frame.cycle_id.astype(str).str.strip().eq("").any() or frame.cycle_id.duplicated().any():
        raise ValueError("Column names and nonempty cycle IDs must be unique")
    frame = frame.reset_index(drop=True)
    reasons = np.full(len(frame), "", dtype=object)
    matrix = pd.DataFrame(index=frame.index)
    if "quality_status" in frame:
        reasons[~frame.quality_status.eq("valid").to_numpy()] = "upstream_quality_failure"
    for name in features:
        raw = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
        numeric = pd.to_numeric(raw, errors="coerce")
        bad = (raw.notna() & numeric.isna()) | np.isinf(numeric)
        if name not in OPTIONAL_FEATURES:
            bad |= numeric.isna()
        reasons[bad.to_numpy()] = "invalid_feature:" + name
        matrix[name] = numeric.astype(float)
    return matrix, reasons


def make_pipeline(candidate):
    estimator = (LogisticRegression(**candidate["parameters"]) if candidate["kind"] == "logistic"
                 else HistGradientBoostingClassifier(**candidate["parameters"]))
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if candidate["kind"] == "logistic":
        steps.append(("scaler", StandardScaler()))
    return Pipeline([*steps, ("classifier", estimator)])


def fit_candidate(frame, candidate):
    features = FEATURE_SETS[candidate["feature_set"]]
    matrix, reasons = input_quality(frame, features)
    if (reasons != "").any() or matrix.isna().all().any():
        raise ValueError("Invalid or entirely missing training features")
    if set(frame.label) != set(CLASSES):
        raise ValueError("Every training fold must contain all four classes")
    pipeline = make_pipeline(candidate)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(matrix.to_numpy(), frame.label.to_numpy())
    return pipeline


def ordered_probabilities(pipeline, matrix):
    with threadpool_limits(limits=1):
        values = pipeline.predict_proba(matrix.to_numpy())
    order = [list(pipeline.classes_).index(label) for label in CLASSES]
    values = values[:, order]
    if values.shape != (len(matrix), 4) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError("Invalid model probabilities")
    return values


def decisions(probabilities, threshold):
    scores = 1 - probabilities[:, 0]
    alarm = scores > threshold
    kinds = np.asarray(CLASSES[1:])[probabilities[:, 1:].argmax(axis=1)]
    return scores, alarm, np.where(alarm, kinds, "normal")


def load_bundle(path):
    bundle = joblib.load(path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported Capstone model schema")
    if tuple(bundle["features"]) != FEATURE_SETS.get(bundle["candidate"]["feature_set"]):
        raise ValueError("Model feature whitelist mismatch")
    if not math.isfinite(bundle["threshold"]) or not 0 <= bundle["threshold"] <= 1:
        raise ValueError("Invalid model threshold")
    return bundle


def predict_frame(bundle, frame):
    frame = frame.reset_index(drop=True)
    matrix, reasons = input_quality(frame, bundle["features"])
    valid = reasons == ""
    output = pd.DataFrame({"cycle_id": frame.cycle_id, "predicted_label": pd.Series(pd.NA, index=frame.index, dtype="string"),
                           "anomaly_score": np.nan, "threshold": bundle["threshold"],
                           "alarm": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
                           "quality_status": np.where(valid, "valid", "invalid_data"), "quality_reason": reasons,
                           "model_version": bundle["model_version"]})
    for label in CLASSES:
        output["p_" + label] = np.nan
    if valid.any():
        probabilities = ordered_probabilities(bundle["pipeline"], matrix.loc[valid])
        scores, alarm, labels = decisions(probabilities, bundle["threshold"])
        output.loc[valid, "anomaly_score"] = scores
        output.loc[valid, "alarm"] = alarm
        output.loc[valid, "predicted_label"] = labels
        output.loc[valid, ["p_" + label for label in CLASSES]] = probabilities
    return output
