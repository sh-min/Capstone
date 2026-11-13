"""Evaluate frozen models; this module never fits or changes a threshold."""

import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .io import read_csv, write_json, sha256, new_directory
from .metrics import summary, grouped_metrics
from .model import load_bundle, predict_frame, input_quality, ordered_probabilities, decisions
from .schema import CLASSES, TEST_DATE, SOURCE_COMMIT, FROZEN_MODELS


def attach_metadata(frame, prediction):
    return frame[["cycle_id", "session_id", "date_kst", "label", "leakage_group_id"]].merge(
        prediction, on="cycle_id", validate="one_to_one")


def verify_test_boundary(bundle, frame):
    if frame.empty or not frame.date_kst.eq(TEST_DATE).all() or not frame.split.eq("test").all():
        raise ValueError("Evaluation requires the fixed 2026-11-12 test partition")
    if set(frame.cycle_id) & set(bundle["training_cycle_ids"]):
        raise ValueError("Training cycles appeared in final evaluation")
    if set(frame.leakage_group_id) & set(bundle["training_group_ids"]) or TEST_DATE in bundle["training_dates"]:
        raise ValueError("Training groups/dates appeared in final evaluation")


def frozen_predictions(frame, provenance):
    runtime = Path(provenance["frozen_runtime"])
    runtime_key = "models/model_runtime/normal_models.py"
    if sha256(runtime) != provenance["source_files"][runtime_key]["sha256"]:
        raise ValueError("Frozen runtime integrity mismatch")
    previous = sys.modules.get("normal_models")
    spec = importlib.util.spec_from_file_location("normal_models", runtime)
    module = importlib.util.module_from_spec(spec)
    sys.modules["normal_models"] = module
    try:
        spec.loader.exec_module(module)
        bundles = {}
        for name, source_path in FROZEN_MODELS.items():
            path = Path(provenance["frozen_models"][name])
            if sha256(path) != provenance["source_files"][source_path]["sha256"]:
                raise ValueError("Frozen model integrity mismatch")
            bundles[name] = joblib.load(path)
    finally:
        if previous is None:
            sys.modules.pop("normal_models", None)
        else:
            sys.modules["normal_models"] = previous
    results = {}
    for name, bundle in bundles.items():
        matrix, reasons = input_quality(frame, bundle["features"])
        valid = reasons == ""
        scores = np.full(len(frame), np.nan)
        probabilities = None
        with threadpool_limits(limits=1):
            if "detector" in bundle and valid.any():
                scores[valid] = bundle["detector"].score(matrix.loc[valid].to_numpy())
            elif valid.any():
                # Original supervised pipelines were fitted with named DataFrames.
                model = bundle["model"]
                raw = model.predict_proba(matrix.loc[valid])
                probabilities = raw[:, [list(model.classes_).index(label) for label in CLASSES]]
                scores[valid] = 1 - probabilities[:, 0]
        thresholds = [(name, bundle["threshold"])]
        if name == "frozen_logistic":
            thresholds.append(("frozen_logistic_040", 0.4))
        for identifier, threshold in thresholds:
            prediction = pd.DataFrame({"cycle_id": frame.cycle_id, "anomaly_score": scores,
                                       "threshold": threshold, "quality_status": np.where(valid, "valid", "invalid_data"),
                                       "quality_reason": reasons, "model_version": identifier})
            prediction["alarm"] = pd.Series(pd.NA, index=prediction.index, dtype="boolean")
            prediction["predicted_label"] = pd.Series(pd.NA, index=prediction.index, dtype="string")
            prediction.loc[valid, "alarm"] = scores[valid] > threshold
            if probabilities is None:
                prediction.loc[valid, "predicted_label"] = np.where(scores[valid] > threshold, "anomaly", "normal")
            else:
                _, _, types = decisions(probabilities, threshold)
                prediction.loc[valid, "predicted_label"] = types
                prediction[["p_" + label for label in CLASSES]] = np.nan
                prediction.loc[valid, ["p_" + label for label in CLASSES]] = probabilities
            results[identifier] = attach_metadata(frame, prediction)
    recorded_path = Path(provenance["frozen_recorded_scores"])
    source_key = "models/bolt-known-load/bc-independent/CYCLE_SCORES.csv"
    if sha256(recorded_path) != provenance["source_files"][source_key]["sha256"]:
        raise ValueError("Frozen reference scores integrity mismatch")
    recorded = read_csv(recorded_path).set_index("cycle_id")
    result = results["frozen_logistic"].set_index("cycle_id")
    valid_ids = result.index[result.quality_status.eq("valid")]
    if not valid_ids.isin(recorded.index).all():
        raise ValueError("Frozen reference scores lack test cycles")
    np.testing.assert_allclose(result.loc[valid_ids, "anomaly_score"], recorded.loc[valid_ids, "score"], rtol=1e-10, atol=1e-10)
    if not np.array_equal(result.loc[valid_ids, "alarm"].to_numpy(bool), recorded.loc[valid_ids, "stored_alarm"].to_numpy(bool)):
        raise AssertionError("Frozen reference alarms changed")
    candidate = results["frozen_logistic_040"].set_index("cycle_id")
    if not np.array_equal(candidate.loc[valid_ids, "alarm"].to_numpy(bool), recorded.loc[valid_ids, "candidate_alarm"].to_numpy(bool)):
        raise AssertionError("Frozen 0.40 reference alarms changed")
    return results


def render_report(output, predictions, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    lines = ["# xArm 하중 판정: 고정 시험셋 회고 평가", "",
             "모델과 임계값을 고정한 뒤 2026-11-12 자료를 평가했습니다. 이 자료는 이전 연구에서 분석한 기록이며 새 현장 검증이 아닙니다.",
             "A121 시험 회차가 없으므로 A121의 최종 탐지율은 산출하지 않습니다. Macro-F1은 실제 정답이 존재하는 정상·B82·C182의 평균입니다.", "",
             "| 모델 | 유효/전체 | 정상 오탐 | B82 탐지 | C182 탐지 | 균형 점수 J | 유형 Macro-F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    comparison = []
    for name, info in metrics.items():
        conditions = {x["label"]: x for x in info["conditions"]}
        rates = [f"{conditions[label]['alarms']}/{conditions[label]['n_valid']}" for label in ("normal", "B82", "C182")]
        lines.append(f"| {name} | {info['n_valid']}/{info['n_total']} | {' | '.join(rates)} | {fmt(info['objective'])} | {fmt(info['macro_f1'])} |")
        comparison.append({"model": name, "n_total": info["n_total"], "n_valid": info["n_valid"], "n_invalid": info["n_invalid"],
                           "objective": info["objective"], "macro_f1": info["macro_f1"],
                           **{label + "_alarm_rate": conditions[label]["alarm_rate"] for label in CLASSES}})
    lines += ["", "J는 정상 판정률과 B82·C182 평균 탐지율을 각각 50% 반영합니다. A121이 포함된 개발 선택 점수와 직접 같은 척도로 순위를 매기지 않습니다.",
              "경보는 맞았지만 유형을 틀린 경우 탐지에는 성공, 유형 분류에는 실패로 집계합니다. 품질 불량 회차는 정상으로 처리하지 않으며, 분모·판정 보류·검출 비율을 metrics.json에 함께 기록합니다.",
              "", "## 산출물", "", "- predictions.csv: 모든 모델의 회차별 점수·경보·유형",
              "- metrics.json / comparison.csv: 분모를 포함한 성능 비교",
              "- by_date.csv / by_session.csv: 날짜·세션별 오탐 및 탐지",
              "- confusion.csv / confusion.png: 선택 모델의 유형 혼동",
              "- scores.png / condition_rates.png: 점수 분포와 조건별 경보율", "",
              "시험 결과에 따른 자동 재학습이나 임계값 변경은 수행하지 않습니다."]
    (output / "REPORT_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    selected = predictions["selected"].loc[predictions["selected"].quality_status.eq("valid")]
    confusion = pd.crosstab(selected.label, selected.predicted_label).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    confusion.to_csv(output / "confusion.csv")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(confusion, cmap="Blues")
    ax.set(xticks=range(4), xticklabels=CLASSES, yticks=range(4), yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Observed", title="Historical test; A121 has no test examples")
    for y in range(4):
        for x in range(4):
            ax.text(x, y, str(confusion.iloc[y, x]), ha="center", va="center")
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for index, label in enumerate(CLASSES):
        group = selected.loc[selected.label.eq(label)]
        if len(group):
            ax.scatter(np.full(len(group), index), group.anomaly_score, alpha=0.55, label=label)
    if len(selected):
        ax.axhline(float(selected.threshold.iloc[0]), color="black", linestyle="--", label="Frozen threshold")
    ax.set(xticks=range(4), xticklabels=CLASSES, ylabel="1 - model P(normal)", title="Selected model scores; A121 not evaluated")
    ax.legend(fontsize=8)
    fig.savefig(output / "scores.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    names = list(metrics)
    width = 0.8 / len(names)
    for index, name in enumerate(names):
        values = {r["label"]: r["alarm_rate"] for r in metrics[name]["conditions"]}
        ax.bar(np.arange(3) - 0.4 + width * (index + 0.5), [values[k] if values[k] is not None else np.nan for k in ("normal", "B82", "C182")], width, label=name)
    ax.set(xticks=range(3), xticklabels=("Normal FPR", "B82 detection", "C182 detection"), ylim=(0, 1.05), ylabel="Rate among valid cycles")
    ax.legend(fontsize=7)
    fig.savefig(output / "condition_rates.png", dpi=160)
    plt.close(fig)


def evaluate(model_path, data, output, references=True):
    model_path, data = Path(model_path), Path(data)
    if Path(output).exists():
        raise FileExistsError("Use a new evaluation output directory")
    provenance = json.loads((data / "provenance.json").read_text(encoding="utf-8"))
    if provenance["source_commit"] != SOURCE_COMMIT or sha256(data / "test.csv") != provenance["files"]["test.csv"]:
        raise ValueError("Test data provenance mismatch")
    test = read_csv(data / "test.csv")
    selected = load_bundle(model_path)
    verify_test_boundary(selected, test)
    predictions = {"selected": attach_metadata(test, predict_frame(selected, test))}
    baseline_path = model_path.with_name("baseline.joblib")
    if baseline_path.is_file():
        baseline = load_bundle(baseline_path)
        verify_test_boundary(baseline, test)
        predictions["matched_logistic"] = attach_metadata(test, predict_frame(baseline, test))
    if references:
        predictions.update(frozen_predictions(test, provenance))
    metrics = {name: summary(frame) for name, frame in predictions.items()}
    output = new_directory(output)
    pd.concat([frame.assign(model_id=name) for name, frame in predictions.items()], ignore_index=True).to_csv(output / "predictions.csv", index=False)
    for key in ("date_kst", "session_id"):
        tables = [grouped_metrics(frame, key).assign(model_id=name) for name, frame in predictions.items()]
        pd.concat(tables, ignore_index=True).to_csv(output / ("by_date.csv" if key == "date_kst" else "by_session.csv"), index=False)
    write_json(output / "metrics.json", metrics)
    render_report(output, predictions, metrics)
    write_json(output / "verification.json", {"model_sha256": sha256(model_path), "test_sha256": sha256(data / "test.csv"),
                                               "model_changed": False, "threshold_changed": False, "training_executed": False,
                                               "frozen_score_replay_checked": references, "A121_final_test_available": False})
    print("Saved retrospective evaluation to " + str(output), flush=True)
    return metrics
