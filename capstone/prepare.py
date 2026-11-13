"""Prepare audited features and fixed date splits without fitting any model."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .features import extract_cycle
from .io import read_csv, new_directory, write_json, sha256, environment, code_hashes
from .schema import (ROOTS, SOURCE_COMMIT, SOURCE_REPO, LABELS, DEV_DATES, TEST_DATE,
                     EXPECTED_COUNTS, BASE_FEATURES, FROZEN_MODELS, REFERENCE_FEATURE_FILES)
from .source import SourceStore


def select_manifest(manifests):
    frames = []
    for root, frame in manifests:
        frame = frame.copy()
        frame["source_root"] = root
        frame["source_relative"] = [str(PurePosixPath(root) / str(p)) for p in frame.source]
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    selected = all_rows.loc[
        all_rows.sensor_mount.eq("bolt_and_nut_user_reported")
        & all_rows.role.ne("study_only") & all_rows.condition.isin(LABELS)
        & all_rows.quality_valid.astype(str).str.lower().eq("true")
    ].copy()
    if selected.cycle_id.isna().any() or selected.cycle_id.duplicated().any():
        raise ValueError("Non-unique or missing source cycle IDs")
    selected["label"] = selected.condition.map(LABELS)
    if not selected.date_kst.isin((*DEV_DATES, TEST_DATE)).all():
        raise ValueError("Unplanned collection date")
    selected["split"] = np.where(selected.date_kst.eq(TEST_DATE), "test", "development")
    for key in ("session_id", "leakage_group_id"):
        if selected.groupby(key).split.nunique().max() > 1:
            raise ValueError("Split leakage through " + key)
    expected_loads = {"normal": (41, 0, "none"), "A121": (121, 0, "none"),
                      "B82": (41, 82, "upper"), "C182": (41, 182, "lower")}
    for row in selected.to_dict("records"):
        if (float(row["payload_mass_g"]), float(row["frame_added_mass_g"]), row["frame_site"]) != expected_loads[row["label"]]:
            raise ValueError("Condition metadata mismatch: " + row["cycle_id"])
    actual = {date: group.label.value_counts().to_dict() for date, group in selected.groupby("date_kst")}
    if actual != EXPECTED_COUNTS:
        raise ValueError(f"Pinned source counts changed: {actual}")
    return selected.sort_values(["date_kst", "session_id", "cycle_index"]).reset_index(drop=True)


def verify_parity(features, reference_frames):
    comparisons = []
    for name, reference in reference_frames:
        required = {"cycle_id", *BASE_FEATURES}
        if not required.issubset(reference.columns):
            raise ValueError("Reference feature schema mismatch: " + name)
        if reference.cycle_id.duplicated().any():
            raise ValueError("Duplicate reference cycle: " + name)
        valid = features.loc[features.quality_status.eq("valid")].set_index("cycle_id")
        reference = reference.set_index("cycle_id")
        common = valid.index.intersection(reference.index)
        if not len(common):
            continue
        actual = valid.loc[common, list(BASE_FEATURES)].to_numpy(float)
        expected = reference.loc[common, list(BASE_FEATURES)].to_numpy(float)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True,
                                   err_msg="Raw feature parity: " + name)
        errors = np.nanmax(np.abs(actual - expected), axis=1)
        comparisons.extend({"cycle_id": cid, "reference": name, "max_abs_error": float(error)}
                           for cid, error in zip(common, errors))
    if not comparisons:
        raise ValueError("No source feature rows were reproduced")
    return pd.DataFrame(comparisons)


def prepare(repository, output, cache=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new preparation output directory")
    store = SourceStore(repository, cache or output.parent / "source")
    manifests = [(root, read_csv(store.fetch(root + "/CYCLE_MANIFEST.csv"))) for root in ROOTS]
    selected = select_manifest(manifests)
    paths = [source + "/" + name for source in sorted(selected.source_relative.unique())
             for name in ("measurements.csv", "adxl345_1.csv", "single_queries.csv")]
    print(f"Selected {len(selected)} cycles / {selected.session_id.nunique()} sessions; fetching {len(paths)} raw files", flush=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, _ in enumerate(executor.map(store.fetch, paths), start=1):
            if index % 12 == 0 or index == len(paths):
                print(f"Verified raw files {index}/{len(paths)}", flush=True)
    rows = []
    for index, (source, group) in enumerate(selected.groupby("source_relative", sort=False), start=1):
        folder = store.cache / source
        ina = read_csv(folder / "measurements.csv")
        accel = read_csv(folder / "adxl345_1.csv")
        pos = read_csv(folder / "single_queries.csv")
        session = group.session_id.iloc[0]
        if not ina.session_id.eq(session).all() or not accel.session_id.eq(session).all():
            raise ValueError("Raw stream session mismatch: " + source)
        for row in group.to_dict("records"):
            row.update(extract_cycle(row, ina, accel, pos))
            rows.append(row)
        print(f"Extracted session {index}/{selected.session_id.nunique()}", flush=True)
    features = pd.DataFrame(rows)
    references = [(path, read_csv(store.fetch(path))) for path in REFERENCE_FEATURE_FILES]
    parity = verify_parity(features, references)
    frozen = {key: str(store.fetch(path)) for key, path in FROZEN_MODELS.items()}
    runtime = store.fetch("models/model_runtime/normal_models.py")
    recorded_scores = store.fetch("models/bolt-known-load/bc-independent/CYCLE_SCORES.csv")
    new_directory(output)
    selected.to_csv(output / "manifest.csv", index=False)
    features.to_csv(output / "features.csv", index=False)
    for split in ("development", "test"):
        features.loc[features.split.eq(split)].to_csv(output / (split + ".csv"), index=False)
    features[["cycle_id", "session_id", "date_kst", "label", "quality_status", "quality_reason", "optional_missing"]].to_csv(output / "quality.csv", index=False)
    selected[["cycle_id", "session_id", "date_kst", "leakage_group_id", "role", "split", "label"]].to_csv(output / "splits.csv", index=False)
    parity.to_csv(output / "feature_parity.csv", index=False)
    write_json(output / "provenance.json", {
        "source_repository": SOURCE_REPO, "source_commit": SOURCE_COMMIT,
        "source_files": dict(sorted(store.records.items())), "environment": environment(), "code_sha256": code_hashes(),
        "files": {p.name: sha256(p) for p in sorted(output.glob("*.csv"))},
        "frozen_models": frozen, "frozen_runtime": str(runtime), "frozen_recorded_scores": str(recorded_scores),
        "counts": {date: group.label.value_counts().to_dict() for date, group in selected.groupby("date_kst")},
        "quality_counts": features.quality_status.value_counts().to_dict(),
        "parity_cycles": int(parity.cycle_id.nunique()), "parity_max_abs_error": float(parity.max_abs_error.max()),
        "training_executed": False,
        "scope": "Retrospective; A121 has no final-test examples; previously analyzed historical data",
    })
    print(f"Prepared {len(features)} cycles at {output}; no model was fitted", flush=True)
    return features
