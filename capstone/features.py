"""Native-sample features matching V4 / V3-C definitions; no model fitting."""

import math
import statistics

import numpy as np
import pandas as pd

from .schema import BASE_FEATURES, PHASE_FEATURES, PHASES, OPTIONAL_FEATURES


def statistics_for(values):
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("empty or nonfinite sensor window")
    return {
        "mean": statistics.fmean(values),
        "rms": math.sqrt(statistics.fmean(x * x for x in values)),
        "std": statistics.pstdev(values), "range": max(values) - min(values),
        "abs_peak": max(map(abs, values)),
        "abs_p95_nearest_rank": sorted(map(abs, values))[math.ceil(0.95 * len(values)) - 1],
    }


def between(frame, column, begin, end):
    return frame.loc[frame[column].ge(begin) & frame[column].le(end)]


def extract_cycle(row, ina, acceleration, positions):
    result = {name: np.nan for name in (*BASE_FEATURES, *PHASE_FEATURES)}
    result.update(quality_status="valid", quality_reason="", optional_missing="")
    try:
        begin, end = int(row["No3_begin_ns"]), int(row["No3_end_ns"])
        if begin >= end:
            raise ValueError("invalid No3 time bounds")
        clock = "host_receive_monotonic_ns"
        i = between(ina, clock, begin, end)
        a = between(acceleration, clock, begin, end)
        p = between(positions.loc[positions.cycle_id.eq(row["cycle_id"]) & positions.phase.eq("No3_A_to_B")],
                    "response_ns", begin, end)
        if i.empty or a.empty or p.empty:
            raise ValueError("missing sensor stream")
        if not i.valid.eq(1).all() or not i.status.eq("OK").all():
            raise ValueError("invalid INA sample status")
        if not p.status.eq("complete").all() or not p.value_valid.eq(1).all():
            raise ValueError("invalid servo response")
        if not (np.diff(i[clock]) > 0).all() or not (np.diff(i.seq) == 1).all():
            raise ValueError("INA timestamp/sequence discontinuity")
        if not (np.diff(a[clock]) >= 0).all():
            raise ValueError("ADXL receive timestamps reversed")
        if "sample_seq" in a and not (np.diff(a.sample_seq) == 1).all():
            raise ValueError("ADXL sample sequence discontinuity")
        for name in ("current_A", "voltage_V"):
            for stat, value in statistics_for(i[name]).items():
                if name + "_" + stat in result:
                    result[name + "_" + stat] = value
        for name in ("x_g", "y_g", "z_g"):
            result[name + "_std"] = statistics_for(a[name])["std"]
        xyz = a[["x_g", "y_g", "z_g"]].to_numpy(dtype=float)
        vector = statistics_for(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1))
        for stat in ("rms", "abs_peak"):
            result["mean_removed_vector_g_" + stat] = vector[stat]
        for servo in range(1, 7):
            samples = p.loc[p.servo_id.eq(servo)].sort_values("response_ns")
            if len(samples) < 2 or not (np.diff(samples.response_ns) > 0).all():
                raise ValueError(f"missing/duplicate servo {servo} samples")
            stats = statistics_for(samples.position_raw)
            for stat in ("range", "std"):
                result[f"position_ID{servo}_raw_{stat}"] = stats[stat]
        p1 = p.loc[p.servo_id.eq(1)].sort_values("response_ns")
        pt = (p1.response_ns.to_numpy() - begin) / 1e9
        grip = (pt >= 3) & (pt < 5)
        gt = pt[grip]
        if len(gt) == 4 and (np.diff(gt) > 0).all() and np.diff(gt).max() <= 0.75:
            result["ID1_3_5s_IQR"] = float(np.diff(np.quantile(p1.position_raw.to_numpy()[grip], [0.25, 0.75]))[0])
        t = (i[clock].to_numpy() - begin) / 1e9
        for phase, (lo, hi, minimum) in PHASES.items():
            mask = (t >= lo) & (t < hi)
            if mask.sum() >= minimum and np.diff(t[mask]).max() <= 0.25 and (np.diff(i.seq.to_numpy()[mask]) == 1).all():
                result[phase + "_current_mean"] = float(i.current_A.to_numpy()[mask].mean())
        result["optional_missing"] = ";".join(name for name in (*BASE_FEATURES, *PHASE_FEATURES)
                                                  if name in OPTIONAL_FEATURES and pd.isna(result[name]))
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        result["quality_status"] = "invalid_data"
        result["quality_reason"] = str(error)
    return result
