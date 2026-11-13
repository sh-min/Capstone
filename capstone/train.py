"""Date-held-out selection. This module does nothing until train() is invoked."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .io import read_csv, new_directory, sha256, write_json, environment, code_hashes
from .metrics import objective
from .model import (load_config, candidates, fit_candidate, ordered_probabilities, input_quality,
                    decisions, predict_frame, load_bundle)
from .schema import DEV_DATES, TEST_DATE, CLASSES, FEATURE_SETS, SOURCE_COMMIT


def date_folds(frame):
    if not frame.date_kst.isin(DEV_DATES).all() or set(frame.date_kst) != set(DEV_DATES):
        raise ValueError("Only the three fixed development dates may enter training")
    if not frame.split.eq("development").all():
        raise ValueError("Test rows entered development")
    if frame.cycle_id.duplicated().any() or frame.cycle_id.isna().any():
        raise ValueError("Duplicate/missing development cycle IDs")
    if not frame.label.isin(CLASSES).all():
        raise ValueError("Unknown target label")
    for key in ("session_id", "leakage_group_id"):
        if frame[key].isna().any() or frame.groupby(key).date_kst.nunique().max() > 1:
            raise ValueError("Connected groups cross date folds: " + key)
    for date in DEV_DATES:
        validation = frame.date_kst.eq(date).to_numpy()
        training = ~validation
        if set(frame.loc[training, "label"]) != set(CLASSES):
            raise ValueError("A training fold lacks one or more classes")
        yield date, np.flatnonzero(training), np.flatnonzero(validation)


def ranking_key(row, candidate):
    regularization = candidate["parameters"].get("C", candidate["parameters"].get("max_leaf_nodes"))
    return (-round(row["objective"], 12), round(row["date_macro_fpr"], 12),
            candidate["kind"] != "logistic", candidate["feature_set"] != "base24", regularization, -row["threshold"])


def train(data, output, config_path=None):
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError("Use a new training output directory")
    provenance = json.loads((data / "provenance.json").read_text(encoding="utf-8"))
    if provenance["source_commit"] != SOURCE_COMMIT or sha256(data / "development.csv") != provenance["files"]["development.csv"]:
        raise ValueError("Development data provenance mismatch")
    all_dev = read_csv(data / "development.csv")
    dev = all_dev.loc[all_dev.quality_status.eq("valid")].reset_index(drop=True)
    folds = list(date_folds(dev))
    config = load_config(config_path)
    choices = candidates(config)
    grid = config["thresholds"]
    if grid != {"start": 0.0, "stop": 1.0, "step": 0.01}:
        raise ValueError("The frozen threshold grid must be 0..1 in steps of .01")
    thresholds = np.linspace(0, 1, 101)
    new_directory(output)
    write_json(output / "config.json", config)
    all_oof, sweeps, winners = [], [], []
    fold_rows = []
    for date, fit, validation in folds:
        for role, indices in (("train", fit), ("validation", validation)):
            fold_rows.extend({"fold": date, "role": role, "cycle_id": dev.iloc[i].cycle_id,
                              "session_id": dev.iloc[i].session_id, "date_kst": dev.iloc[i].date_kst,
                              "leakage_group_id": dev.iloc[i].leakage_group_id} for i in indices)
    pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
    for candidate in choices:
        print("Training date folds: " + candidate["id"], flush=True)
        probabilities = np.full((len(dev), 4), np.nan)
        for date, fit, validation in folds:
            pipeline = fit_candidate(dev.iloc[fit], candidate)
            matrix, reasons = input_quality(dev.iloc[validation], FEATURE_SETS[candidate["feature_set"]])
            if (reasons != "").any():
                raise ValueError("Invalid validation features")
            probabilities[validation] = ordered_probabilities(pipeline, matrix)
        scores = 1 - probabilities[:, 0]
        if not np.isfinite(scores).all():
            raise ValueError("Incomplete out-of-fold predictions")
        rows = [{"candidate_id": candidate["id"], "threshold": float(t), **objective(dev, scores, float(t))} for t in thresholds]
        best = min(rows, key=lambda row: ranking_key(row, candidate))
        winners.append((candidate, best))
        sweeps.extend(rows)
        oof = dev[["cycle_id", "session_id", "date_kst", "label", "leakage_group_id"]].copy()
        oof["candidate_id"] = candidate["id"]
        oof["anomaly_score"] = scores
        oof["threshold"] = best["threshold"]
        _, oof["alarm"], oof["predicted_label"] = decisions(probabilities, best["threshold"])
        oof["quality_status"] = "valid"
        for i, label in enumerate(CLASSES):
            oof["p_" + label] = probabilities[:, i]
        all_oof.append(oof)
    selected, selected_metrics = min(winners, key=lambda pair: ranking_key(pair[1], pair[0]))
    baseline, baseline_metrics = next(pair for pair in winners if pair[0]["id"] == "logistic_base24_C1")
    fingerprints = {"development": sha256(data / "development.csv"), "config": sha256(output / "config.json"), "code": code_hashes()}
    run_id = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:12]
    bundles = {}
    for name, candidate, metrics in (("model", selected, selected_metrics), ("baseline", baseline, baseline_metrics)):
        if candidate["id"] in bundles:
            bundle = bundles[candidate["id"]]
        else:
            print("Final development fit: " + candidate["id"], flush=True)
            bundle = {"schema_version": 1, "pipeline": fit_candidate(dev, candidate),
                      "features": list(FEATURE_SETS[candidate["feature_set"]]), "threshold": metrics["threshold"],
                      "candidate": candidate, "classes": list(CLASSES), "seed": config["seed"],
                      "model_version": candidate["id"] + "-" + run_id,
                      "source_commit": SOURCE_COMMIT, "fingerprints": fingerprints,
                      "training_cycle_ids": dev.cycle_id.tolist(), "training_dates": list(DEV_DATES),
                      "training_group_ids": sorted(dev.leakage_group_id.unique().tolist()),
                      "threshold_source": "date-held-out development predictions; selection score is not an independent estimate"}
            bundles[candidate["id"]] = bundle
        path = output / (name + ".joblib")
        joblib.dump(bundle, path)
        before = predict_frame(bundle, dev)
        after = predict_frame(load_bundle(path), dev)
        np.testing.assert_array_equal(before.anomaly_score, after.anomaly_score)
        if not before.alarm.equals(after.alarm):
            raise AssertionError("Serialization changed decisions")
    pd.concat(all_oof, ignore_index=True).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(sweeps).to_csv(output / "threshold_sweep.csv", index=False)
    pd.DataFrame([{"candidate_id": c["id"], **m} for c, m in winners]).to_csv(output / "candidates.csv", index=False)
    write_json(output / "metadata.json", {
        "run_id": run_id, "selected": selected, "selected_development_metrics": selected_metrics,
        "baseline": baseline, "baseline_development_metrics": baseline_metrics,
        "n_development": len(all_dev), "n_training_valid": len(dev), "n_excluded": len(all_dev) - len(dev),
        "test_date": TEST_DATE, "test_used_for_selection": False, "test_data_read": False,
        "environment": environment(), "fingerprints": fingerprints,
        "prepared_provenance_sha256": sha256(data / "provenance.json"),
        "artifacts": {name: sha256(output / name) for name in ("model.joblib", "baseline.joblib")},
        "scope": "Retrospective development selection; final test has no A121 examples",
    })
    print(f"Saved {output / 'model.joblib'}; final test has not been evaluated", flush=True)
    return selected_metrics
