"""Archive an existing run and derive diagnostic tables/figures; never fit a model."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from capstone.io import sha256, write_json, read_csv, code_hashes
from capstone.metrics import grouped_metrics, summary
from capstone.schema import CLASSES


def plot_candidates(table, selected, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = table.sort_values("objective")
    colors = ["#7c3aed" if name == selected else "#0d9488" if name.startswith("boosting") else "#94a3b8" for name in table.candidate_id]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    ax.barh(table.candidate_id, table.objective, color=colors)
    for i, row in enumerate(table.itertuples()):
        ax.text(row.objective + 0.008, i, f"J={row.objective:.3f}, threshold={row.threshold:.2f}", va="center", fontsize=9)
    ax.set(xlim=(0, 0.9), xlabel="Date-balanced development selection score J",
           title="10 candidates: selection uses development OOF predictions only")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "candidate_scores.png", dpi=180)
    plt.close(fig)


def plot_date_metrics(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = sorted(table.date_kst.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), width_ratios=[1, 3], layout="constrained")
    for ax, labels, color, title in zip(axes, [["normal"], list(CLASSES[1:])], ["Oranges", "Blues"],
                                       ["Normal false alarms (lower is better)", "Load detection (higher is better)"]):
        values = np.full((len(dates), len(labels)), np.nan)
        annotations = {}
        for i, date in enumerate(dates):
            for j, label in enumerate(labels):
                rows = table.loc[table.date_kst.eq(date) & table.label.eq(label)]
                if len(rows):
                    row = rows.iloc[0]
                    values[i, j] = row.alarm_rate
                    annotations[i, j] = f"{int(row.alarms)}/{int(row.n_valid)}\n{row.alarm_rate:.1%}"
        cmap = plt.get_cmap(color).copy()
        cmap.set_bad("#e5e7eb")
        ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(dates)), yticklabels=dates, title=title)
        for i in range(len(dates)):
            for j in range(len(labels)):
                ax.text(j, i, annotations.get((i, j), "not collected"), ha="center", va="center", fontsize=10,
                        color="white" if np.isfinite(values[i, j]) and values[i, j] > 0.7 else "#111827")
    fig.suptitle("Selected model: date-held-out development predictions", fontsize=12)
    fig.savefig(output / "oof_by_date.png", dpi=180)
    plt.close(fig)


def archive(run, evaluation, prepared, output):
    run, evaluation, prepared, output = map(Path, (run, evaluation, prepared, output))
    if output.exists():
        raise FileExistsError("Choose a new immutable snapshot directory")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    verification = json.loads((evaluation / "verification.json").read_text(encoding="utf-8"))
    provenance = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
    for name in ("model.joblib", "baseline.joblib"):
        if sha256(run / name) != metadata["artifacts"][name]:
            raise ValueError("Training artifact hash mismatch: " + name)
    if verification["model_sha256"] != sha256(run / "model.joblib"):
        raise ValueError("Evaluation belongs to a different model")
    if verification["test_sha256"] != provenance["files"]["test.csv"]:
        raise ValueError("Evaluation belongs to different test data")
    if metadata["fingerprints"]["code"] != code_hashes():
        raise ValueError("Current package source differs from the recorded training code")
    predictions = read_csv(evaluation / "predictions.csv")
    recorded_metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    for name, rows in predictions.groupby("model_id"):
        actual = summary(rows)
        for key in ("objective", "macro_f1", "coverage"):
            expected = recorded_metrics[name][key]
            if expected is None:
                if actual[key] is not None:
                    raise AssertionError("Unexpected metric value")
            else:
                np.testing.assert_allclose(actual[key], expected, rtol=1e-12, atol=1e-12)
    output.mkdir(parents=True)
    original_files = {}

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise AssertionError("Archive copy changed file bytes")
        original_files[str(destination.relative_to(output))] = str(source)

    for name in ("model.joblib", "baseline.joblib", "metadata.json", "config.json", "candidates.csv",
                 "threshold_sweep.csv", "oof_predictions.csv", "folds.csv"):
        copy(run / name, output / name)
    for source in sorted(evaluation.iterdir()):
        if source.is_file() and source.suffix in (".csv", ".json", ".md", ".png"):
            copy(source, output / "evaluation" / source.name)
    copy(prepared / "splits.csv", output / "data_splits.csv")
    copy(prepared / "feature_parity.csv", output / "feature_parity.csv")
    write_json(output / "data_provenance.json", {k: v for k, v in provenance.items()
                                               if k not in ("frozen_models", "frozen_runtime", "frozen_recorded_scores")})
    derived = output / "analysis"
    derived.mkdir()
    oof = read_csv(run / "oof_predictions.csv")
    selected_id = metadata["selected"]["id"]
    selected_oof = oof.loc[oof.candidate_id.eq(selected_id)]
    by_date = grouped_metrics(selected_oof, "date_kst")
    by_date.to_csv(derived / "selected_oof_by_date.csv", index=False)
    selected = predictions.loc[predictions.model_id.eq("selected")].set_index("cycle_id")
    selected.loc[selected.label.eq("normal") & selected.alarm].to_csv(derived / "normal_false_alarms.csv")
    transitions = []
    for name in ("matched_logistic", "frozen_logistic", "frozen_logistic_040"):
        other = predictions.loc[predictions.model_id.eq(name)].set_index("cycle_id").loc[selected.index]
        for condition, mask in (("normal", selected.label.eq("normal")), ("loads", selected.label.ne("normal"))):
            left, right = selected.loc[mask, "alarm"], other.loc[mask, "alarm"]
            transitions.append({"reference": name, "condition": condition, "n": int(mask.sum()),
                                "selected_only": int((left & ~right).sum()), "reference_only": int((~left & right).sum()),
                                "both": int((left & right).sum()), "neither": int((~left & ~right).sum())})
    pd.DataFrame(transitions).to_csv(derived / "paired_alarm_changes.csv", index=False)
    bundle = joblib.load(run / "model.joblib")
    classifier = bundle["pipeline"].named_steps["classifier"]
    if metadata["selected"]["kind"] == "logistic":
        coefficients = pd.DataFrame(classifier.coef_.T, index=bundle["features"], columns=classifier.classes_)
        coefficients.index.name = "standardized_feature"
        coefficients.to_csv(derived / "logistic_coefficients.csv")
        model_shape = {"coefficient_shape": list(classifier.coef_.shape), "intercept_shape": list(classifier.intercept_.shape)}
    else:
        model_shape = {"model_kind": metadata["selected"]["kind"]}
    plot_candidates(read_csv(run / "candidates.csv"), selected_id, derived)
    plot_date_metrics(by_date, derived)
    write_json(output / "snapshot.json", {
        "run_id": metadata["run_id"], "source_run": str(run), "source_evaluation": str(evaluation),
        "training_executed_by_this_export": False, "evaluation_metrics_recomputed": True,
        "model_shape": model_shape, "source_copies": original_files,
        "files": {str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(output.rglob("*")) if p.is_file() and p.name != "snapshot.json"},
    })
    print(f"Archived {metadata['run_id']} to {output}; no fitting or threshold adjustment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    archive(args.run, args.evaluation, args.prepared, args.out)
