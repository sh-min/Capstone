"""Date-balanced detection objective, independent of class-type accuracy."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from .schema import CLASSES


def objective(frame, scores, threshold):
    work = frame[["label", "date_kst"]].copy()
    work["alarm"] = np.asarray(scores) > threshold
    rates = work.groupby(["label", "date_kst"]).alarm.mean().groupby(level=0).mean()
    if "normal" not in rates or not any(label in rates for label in CLASSES[1:]):
        available_loads = rates.reindex(CLASSES[1:]).dropna()
        return {"objective": None,
                "date_macro_fpr": float(rates["normal"]) if "normal" in rates else None,
                "load_macro_date_recall": float(available_loads.mean()) if len(available_loads) else None}
    fpr = float(rates["normal"])
    recall = float(rates.reindex(CLASSES[1:]).dropna().mean())
    return {"objective": 0.5 * (1 - fpr) + 0.5 * recall,
            "date_macro_fpr": fpr, "load_macro_date_recall": recall}


def summary(frame):
    valid = frame.loc[frame.quality_status.eq("valid")].copy()
    counts = {"n_total": len(frame), "n_valid": len(valid), "n_invalid": len(frame) - len(valid),
              "coverage": len(valid) / len(frame) if len(frame) else 0.0}
    conditions = []
    for label in CLASSES:
        all_condition = frame.loc[frame.label.eq(label)]
        condition = valid.loc[valid.label.eq(label)]
        alarms = int(condition.alarm.astype(bool).sum())
        conditions.append({"label": label, "n_total": len(all_condition), "n_valid": len(condition),
                           "n_invalid": len(all_condition) - len(condition), "alarms": alarms,
                           "alarm_rate": alarms / len(condition) if len(condition) else None,
                           "detected_fraction_of_all": alarms / len(all_condition) if len(all_condition) and label != "normal" else None})
    counts["conditions"] = conditions
    if not len(valid):
        counts.update(objective=None, date_macro_fpr=None, macro_f1=None, auc={})
        return counts
    # Scores can come from models with different scales, so use the already frozen decisions.
    decisions = valid.alarm.astype(bool).to_numpy(dtype=float)
    counts.update(objective(valid, decisions, 0.5))
    typed = valid.predicted_label.isin(CLASSES).all()
    present = [label for label in CLASSES if valid.label.eq(label).any()]
    counts["macro_f1"] = float(f1_score(valid.label, valid.predicted_label, labels=present, average="macro", zero_division=0)) if typed else None
    ranking = {}
    for label in CLASSES[1:]:
        pair = valid.loc[valid.label.isin(("normal", label))]
        if pair.label.nunique() == 2:
            truth = pair.label.eq(label)
            ranking[label] = {"auroc": float(roc_auc_score(truth, pair.anomaly_score)),
                              "average_precision": float(average_precision_score(truth, pair.anomaly_score))}
        else:
            ranking[label] = None
    counts["auc"] = ranking
    return counts


def grouped_metrics(frame, key):
    rows = []
    for group, part in frame.groupby(key, dropna=False):
        info = summary(part)
        for condition in info["conditions"]:
            if condition["n_total"]:
                rows.append({key: group, **condition})
    return pd.DataFrame(rows)
