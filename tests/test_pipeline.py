"""All tests prohibit model fitting, including integration tests."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from capstone.__main__ import main
from capstone.features import extract_cycle
from capstone.io import sha256, write_json, read_csv
from capstone.metrics import objective, summary
from capstone.model import (candidates, load_config, make_pipeline, predict_frame,
                            load_bundle, decisions, input_quality)
from capstone.prepare import verify_parity
from capstone.schema import (BASE_FEATURES, PHASE_FEATURES, CLASSES, DEV_DATES, TEST_DATE,
                             SOURCE_COMMIT)
from capstone.train import date_folds, ranking_key
from capstone.evaluate import frozen_predictions, verify_test_boundary


def setUpModule():
    # A test accidentally reaching a real estimator fit fails immediately.
    guard = patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("Training is prohibited in this test suite"))
    guard.start()
    unittest.addModuleCleanup(guard.stop)
    guard = patch("capstone.model.fit_candidate", side_effect=AssertionError("Training is prohibited in this test suite"))
    guard.start()
    unittest.addModuleCleanup(guard.stop)


class FixedProbabilities:
    """Deterministic inference stub with deliberately shuffled class order; no fit method."""
    classes_ = np.array(["C182", "normal", "B82", "A121"])

    def predict_proba(self, x):
        normal = np.where(x[:, 0] < 0.5, 0.8, 0.1)
        rest = 1 - normal
        return np.column_stack([rest * 0.1, normal, rest * 0.6, rest * 0.3])


def stub_bundle():
    return {"schema_version": 1, "pipeline": FixedProbabilities(), "features": list(BASE_FEATURES),
            "candidate": {"feature_set": "base24"}, "threshold": 0.4, "model_version": "test-stub",
            "training_cycle_ids": ["training-only"], "training_group_ids": ["training-group"],
            "training_dates": list(DEV_DATES)}


def feature_frame(n=3):
    frame = pd.DataFrame(np.ones((n, len(BASE_FEATURES))), columns=BASE_FEATURES)
    frame["cycle_id"] = ["cycle-" + str(i) for i in range(n)]
    frame["quality_status"] = "valid"
    frame.loc[0, "current_A_mean"] = 0.0
    return frame


def sensor_fixture():
    begin = 1000_000_000_000
    t = np.arange(91) / 10
    ina = pd.DataFrame({"host_receive_monotonic_ns": begin + np.arange(91) * 100_000_000,
                        "current_A": t, "voltage_V": 8.0, "valid": 1, "status": "OK", "seq": np.arange(91)})
    accel = pd.DataFrame({"host_receive_monotonic_ns": begin + np.arange(901) * 10_000_000,
                          "x_g": np.sin(np.arange(901)), "y_g": 0.01, "z_g": 1.0,
                          "sample_seq": np.arange(901)})
    times = [2.9, 3, 3.5, 4, 4.5, 5, 6]
    positions = pd.DataFrame([{"cycle_id": "c1", "phase": "No3_A_to_B", "servo_id": servo,
                               "response_ns": begin + int(t * 1e9), "position_raw": 10 * i + servo,
                               "status": "complete", "value_valid": 1}
                              for servo in range(1, 7) for i, t in enumerate(times)])
    row = {"cycle_id": "c1", "No3_begin_ns": begin, "No3_end_ns": begin + 9_000_000_000}
    return row, ina, accel, positions


class FeaturesTest(unittest.TestCase):
    def test_native_windows_and_grip_iqr(self):
        row, ina, accel, pos = sensor_fixture()
        result = extract_cycle(row, ina, accel, pos)
        self.assertEqual(result["quality_status"], "valid")
        self.assertAlmostEqual(result["ID1_3_5s_IQR"], 15.0)
        self.assertAlmostEqual(result["S3_current_mean"], 3.45)
        self.assertAlmostEqual(result["S4_current_mean"], 4.45)
        self.assertAlmostEqual(result["S5_current_mean"], 5.7)
        self.assertAlmostEqual(result["current_A_std"], np.std(ina.current_A, ddof=0))

    def test_return_samples_do_not_enter_features(self):
        row, ina, accel, pos = sensor_fixture()
        expected = extract_cycle(row, ina, accel, pos)
        extra = ina.iloc[[-1]].copy()
        extra["host_receive_monotonic_ns"] += 1
        extra["current_A"] = 9999
        actual = extract_cycle(row, pd.concat([ina, extra]), accel, pos)
        self.assertEqual(expected, actual)

    def test_sensor_discontinuity_and_bad_time_are_invalid(self):
        row, ina, accel, pos = sensor_fixture()
        ina.loc[30, "seq"] += 5
        self.assertEqual(extract_cycle(row, ina, accel, pos)["quality_status"], "invalid_data")
        row["No3_end_ns"] = row["No3_begin_ns"]
        self.assertEqual(extract_cycle(row, ina, accel, pos)["quality_status"], "invalid_data")

    def test_missing_grip_sample_is_optional(self):
        row, ina, accel, pos = sensor_fixture()
        pos = pos.drop(pos.index[(pos.servo_id == 1) & (pos.position_raw == 21)])
        result = extract_cycle(row, ina, accel, pos)
        self.assertEqual(result["quality_status"], "valid")
        self.assertTrue(np.isnan(result["ID1_3_5s_IQR"]))

    def test_parity_rejects_changed_feature(self):
        computed = feature_frame()
        reference = computed.copy()
        self.assertEqual(len(verify_parity(computed, [("fixture", reference)])), 3)
        reference.loc[0, "current_A_mean"] += 0.1
        with self.assertRaises(AssertionError):
            verify_parity(computed, [("fixture", reference)])


class SplitAndSelectionTest(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([{"cycle_id": date + label, "date_kst": date, "session_id": date + label,
                              "leakage_group_id": date, "split": "development", "label": label}
                             for date in DEV_DATES for label in (CLASSES if date != DEV_DATES[1] else ["normal"])])

    def test_date_and_connected_group_separation(self):
        frame = self.fixture()
        for date, fit, validation in date_folds(frame):
            self.assertFalse(set(frame.iloc[fit].date_kst) & set(frame.iloc[validation].date_kst))
            self.assertFalse(set(frame.iloc[fit].leakage_group_id) & set(frame.iloc[validation].leakage_group_id))
            self.assertEqual(set(frame.iloc[fit].label), set(CLASSES))

    def test_test_date_and_shared_group_rejected(self):
        frame = self.fixture()
        frame.loc[0, "date_kst"] = TEST_DATE
        with self.assertRaises(ValueError):
            list(date_folds(frame))
        frame = self.fixture()
        frame.loc[:, "leakage_group_id"] = "shared"
        with self.assertRaises(ValueError):
            list(date_folds(frame))

    def test_ten_candidates_have_valid_parameters_without_fitting(self):
        options = candidates(load_config())
        self.assertEqual(len(options), 10)
        for option in options:
            pipeline = make_pipeline(option)
            pipeline.named_steps["classifier"]._validate_params()
            self.assertEqual(pipeline.named_steps["classifier"].class_weight, "balanced")
            if option["kind"] == "boosting":
                self.assertFalse(pipeline.named_steps["classifier"].early_stopping)

    def test_objective_weights_dates_not_cycle_counts(self):
        frame = pd.DataFrame({"label": ["normal"] * 101 + ["A121", "B82", "C182"],
                              "date_kst": ["d1"] * 100 + ["d2"] * 4})
        scores = np.array([1.0] * 100 + [0.0, 1.0, 0.0, 1.0])
        result = objective(frame, scores, 0.5)
        self.assertEqual(result["date_macro_fpr"], 0.5)
        self.assertAlmostEqual(result["objective"], 0.25 + 1 / 3)

    def test_ties_choose_the_planned_simpler_model(self):
        choices = candidates(load_config())
        row = {"objective": 0.8, "date_macro_fpr": 0.1, "threshold": 0.4}
        best = min(choices, key=lambda c: ranking_key(row, c))
        self.assertEqual(best["id"], "logistic_base24_C0.1")

    def test_unavailable_normal_metrics_are_null_not_nan(self):
        frame = pd.DataFrame({"label": ["B82"], "date_kst": [TEST_DATE]})
        result = objective(frame, [0.9], 0.4)
        self.assertIsNone(result["objective"])
        self.assertIsNone(result["date_macro_fpr"])
        json.dumps(result, allow_nan=False)


class PredictionTest(unittest.TestCase):
    def test_column_and_class_order(self):
        data = feature_frame()
        actual = predict_frame(stub_bundle(), data)
        shuffled = predict_frame(stub_bundle(), data[data.columns[::-1]])
        pd.testing.assert_frame_equal(actual, shuffled)
        self.assertEqual(actual.predicted_label.tolist(), ["normal", "B82", "B82"])
        np.testing.assert_allclose(actual.anomaly_score, [0.2, 0.9, 0.9])

    def test_strict_threshold(self):
        scores, alarms, labels = decisions(np.array([[0.5, 0.2, 0.2, 0.1]]), 0.5)
        self.assertFalse(alarms[0])
        self.assertEqual(labels[0], "normal")

    def test_corrupt_or_required_missing_input_abstains(self):
        data = feature_frame()
        data.loc[1, "voltage_V_mean"] = np.nan
        data.loc[2, "current_A_std"] = np.inf
        actual = predict_frame(stub_bundle(), data)
        self.assertEqual(actual.quality_status.tolist(), ["valid", "invalid_data", "invalid_data"])
        self.assertTrue(actual.loc[1:, "alarm"].isna().all())
        self.assertTrue(actual.loc[1:, "predicted_label"].isna().all())
        data = feature_frame().drop(columns="voltage_V_mean")
        self.assertTrue(predict_frame(stub_bundle(), data).alarm.isna().all())

    def test_optional_missing_allowed_but_infinity_and_text_rejected(self):
        data = feature_frame()
        data.loc[0, "ID1_3_5s_IQR"] = np.nan
        data.loc[1, "ID1_3_5s_IQR"] = np.inf
        data["ID1_3_5s_IQR"] = data.ID1_3_5s_IQR.astype(object)
        data.loc[2, "ID1_3_5s_IQR"] = "broken"
        _, reasons = input_quality(data, BASE_FEATURES)
        self.assertEqual(reasons[0], "")
        self.assertNotEqual(reasons[1], "")
        self.assertNotEqual(reasons[2], "")

    def test_duplicate_id_rejected(self):
        data = feature_frame()
        data.loc[1, "cycle_id"] = data.loc[0, "cycle_id"]
        with self.assertRaises(ValueError):
            predict_frame(stub_bundle(), data)

    def test_saved_stub_round_trip(self):
        data = feature_frame()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stub.joblib"
            joblib.dump(stub_bundle(), path)
            pd.testing.assert_frame_equal(predict_frame(stub_bundle(), data), predict_frame(load_bundle(path), data))

    def test_test_boundary_blocks_overlap(self):
        data = feature_frame()
        data["date_kst"], data["split"], data["leakage_group_id"] = TEST_DATE, "test", "test-group"
        bundle = stub_bundle()
        verify_test_boundary(bundle, data)
        bundle["training_cycle_ids"] = [data.cycle_id.iloc[0]]
        with self.assertRaises(ValueError):
            verify_test_boundary(bundle, data)


class InferenceIntegrationTest(unittest.TestCase):
    def test_cli_prediction_and_evaluation_without_training(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            model = root / "model.joblib"
            joblib.dump(stub_bundle(), model)
            frame = feature_frame()
            frame["label"] = ["normal", "B82", "C182"]
            frame["date_kst"], frame["split"], frame["leakage_group_id"], frame["session_id"] = TEST_DATE, "test", "test-group", "session"
            frame.to_csv(prepared / "test.csv", index=False)
            write_json(prepared / "provenance.json", {"source_commit": SOURCE_COMMIT, "files": {"test.csv": sha256(prepared / "test.csv")}})
            self.assertEqual(main(["predict", "--model", str(model), "--features", str(prepared / "test.csv"), "--out", str(root / "predictions.csv")]), 0)
            self.assertEqual(main(["evaluate", "--model", str(model), "--data", str(prepared), "--out", str(root / "evaluation"), "--no-reference"]), 0)
            info = json.loads((root / "evaluation/metrics.json").read_text())["selected"]
            self.assertEqual(info["objective"], 1.0)  # Both anomalies alarm, but C is mislabeled B.
            self.assertLess(info["macro_f1"], 1.0)
            self.assertIsNone(info["conditions"][1]["alarm_rate"])  # A is unavailable, not zero.
            self.assertTrue((root / "evaluation/confusion.png").is_file())


class PreparedDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(__file__).resolve().parents[1] / "data/prepared"
        if not (cls.data / "provenance.json").is_file():
            raise unittest.SkipTest("Run prepare to enable local source replay checks")
        cls.provenance = json.loads((cls.data / "provenance.json").read_text())

    def test_all_real_features_reproduced_without_fitting(self):
        self.assertEqual(self.provenance["parity_cycles"], 1159)
        self.assertLessEqual(self.provenance["parity_max_abs_error"], 1e-10)
        self.assertFalse(self.provenance["training_executed"])
        for name in ("development.csv", "test.csv", "features.csv"):
            self.assertEqual(sha256(self.data / name), self.provenance["files"][name])
        dev = read_csv(self.data / "development.csv")
        self.assertEqual(len(dev), 1069)
        self.assertEqual(len(list(date_folds(dev))), 3)

    def test_archived_model_replay_only_no_new_fit(self):
        frame = read_csv(self.data / "test.csv")
        results = frozen_predictions(frame, self.provenance)
        self.assertEqual(len(results), 3)
        stored = summary(results["frozen_logistic"])
        candidate = summary(results["frozen_logistic_040"])
        self.assertEqual([x["alarms"] for x in stored["conditions"]], [3, 0, 23, 23])
        self.assertEqual([x["alarms"] for x in candidate["conditions"]], [6, 0, 26, 28])


if __name__ == "__main__":
    unittest.main()
