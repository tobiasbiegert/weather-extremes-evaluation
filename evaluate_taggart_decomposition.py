"""Compute decompositions of the Taggart weighted squared error.

For monthly maximum records, the script decomposes Taggart's upper-tail
component S2. For monthly minimum records, it decomposes the reflected
lower-tail component S1. Every forecast case is scored; cases are not selected
according to whether the observation crosses the threshold.

The mean score is decomposed as

    SCORE = MCB - DSC + UNC,

where SCORE is S2 for the upper tail and S1 for the lower tail. Empirical
recalibration is performed by isotonic regression after expressing forecasts
and observations as distances from their case-specific monthly thresholds.

References
----------
Taggart, R. J. (2022). Evaluation of point forecasts for extreme events using
consistent scoring functions. Quarterly Journal of the Royal Meteorological
Society, 148, 306-320.

Gneiting, T., & Resin, J. (2023). Regression diagnostics meets forecast
evaluation: Conditional calibration, reliability diagrams, and coefficient of
determination. Electronic Journal of Statistics, 17, 3226-3286.
"""
from absl import app
from absl import flags
import dask
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.optimize import isotonic_regression
import xarray as xr


PREDICTION_PATH = flags.DEFINE_string(
    "prediction_path",
    None,
    help="Path to deterministic forecast data in Zarr format."
)
TARGET_PATH = flags.DEFINE_string(
    "target_path",
    None,
    help="Path to target data in Zarr format."
)
THRESHOLD_PATH = flags.DEFINE_string(
    "threshold_path",
    None,
    help="Path to monthly record thresholds in Zarr format."
)
OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    help="Where to save the decomposition results."
)
TIME_START = flags.DEFINE_string(
    "time_start",
    None,
    help="Inclusive first forecast initialization time."
)
TIME_STOP = flags.DEFINE_string(
    "time_stop",
    None,
    help="Inclusive last forecast initialization time."
)
VARIABLES = flags.DEFINE_list(
    "variables",
    ["2m_temperature"],
    help="Variables to evaluate."
)
TAIL = flags.DEFINE_enum(
    "tail",
    None,
    ["upper", "lower"],
    help="Use 'upper' for S2 and 'lower' for the reflected S1 score.",
)
CHUNK_SIZE_LAT = flags.DEFINE_integer(
    "chunk_size_lat",
    11,
    help="Chunk size for latitude."
)
CHUNK_SIZE_LON = flags.DEFINE_integer(
    "chunk_size_lon",
    16,
    help="Chunk size for longitude."
)
LEAD_TIMES_HOURS = flags.DEFINE_list(
    "lead_times_hours",
    [],
    help="Lead times in hours. If empty, use all lead times."
)
NUM_WORKERS = flags.DEFINE_integer(
    "num_workers",
    None,
    help="Number of Dask worker threads. If unset, use Dask's default."
)
EXPECTED_INITIALIZATIONS = flags.DEFINE_integer(
    "expected_initializations",
    None,
    help="Optional expected number of initialization times."
)

# def component_names(tail):
#     """Return the stored components for one tail."""
#     score_component = "S2" if tail == "upper" else "S1"
#     return np.array(
#         ["MCB", "DSC", "UNC", score_component, f"{score_component}_VAR", "COUNT"],
#         dtype="U",
#     )

def component_names(tail):
    """Return the stored components for one tail."""
    score_component = "S2" if tail == "upper" else "S1"

    return np.array(
        [
            "MCB",
            "DSC",
            "UNC",
            score_component,
            f"{score_component}_VAR",
            f"{score_component}_HIT",
            f"{score_component}_MISS",
            f"{score_component}_FALSE_ALARM",
            "HIT_COUNT",
            "MISS_COUNT",
            "FALSE_ALARM_COUNT",
            "COUNT",
        ],
        dtype="U",
    )

def open_zarr_auto(path, **kwargs):
    """Open a local or anonymous GCS Zarr store."""
    if str(path).startswith("gs://"):
        return xr.open_zarr(store=path, storage_options={"token": "anon"}, **kwargs)
    return xr.open_zarr(store=path, **kwargs)

def upper_weighted_squared_error(x, y):
    """Evaluate the upper-tail weighted squared error with threshold zero."""
    x_plus = np.maximum(x, 0.0)
    y_plus = np.maximum(y, 0.0)

    return (
        y_plus**2
        - x_plus**2
        - 2.0 * (y - x) * x_plus
    )

# def taggart_decomposition(x, y, t, tail):
#     """
#     Compute the empirical Taggart-score decomposition and uncertainty statistics.

#     Upper-tail cases use x - t and y - t, for which the zero-threshold
#     upper-tail score S2 equals S2 on the original scale. Lower-tail cases use
#     t - x and t - y, for which the same zero-threshold upper-tail score S2
#     equals S1 on the original scale.

#     If none of the transformed observations exceeds zero, the recalibrated score S_rc
#     and marginal score S_mg are both zero. In this case, isotonic regression is
#     skipped and the complete score is attributed to MCB.

#     Parameters
#     ----------
#     x : np.ndarray
#         Forecast values over time.
#     y : np.ndarray
#         Observed values over time.
#     t : np.ndarray
#         Threshold values over time.
#     tail : {"upper", "lower"}
#         Tail being evaluated.

#     Returns
#     -------
#     np.ndarray
#         Array of length 6 containing:
#         [MCB, DSC, UNC, S, S_VAR, COUNT], where S is S2 for the upper tail and S1 for the lower tail.
#     """
#     if tail not in {"upper", "lower"}:
#         raise ValueError("tail must be 'upper' or 'lower'.")

#     mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)

#     if not np.any(mask):
#         return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.0], dtype=np.float32)

#     x = x[mask]
#     y = y[mask]
#     t = t[mask]

#     if tail == "upper":
#         x_tilde = x - t
#         y_tilde = y - t
#     else:
#         x_tilde = t - x
#         y_tilde = t - y

#     weighted_squared_error = upper_weighted_squared_error(x_tilde, y_tilde)
#     S = np.mean(weighted_squared_error)
#     S_var = np.var(weighted_squared_error, ddof=0)
#     count = x_tilde.size

#     # If no observation enters the relevant tail, both the marginal forecast and every isotonic-regression fitted value are nonpositive.
#     # Therefore, S_mg = S_rc = 0, so MCB = S and DSC = UNC = 0.
#     if np.all(y_tilde <= 0):
#         return np.array([S, 0.0, 0.0, S, S_var, count], dtype=np.float32)

#     order = np.argsort(x_tilde, kind="mergesort")
#     y_tilde_sorted = y_tilde[order]

#     x_tilde_rc_sorted = isotonic_regression(y_tilde_sorted, increasing=True).x

#     S_rc = np.mean(upper_weighted_squared_error(x_tilde_rc_sorted, y_tilde_sorted))
#     S_mg = np.mean(upper_weighted_squared_error(np.mean(y_tilde), y_tilde))

#     MCB = S - S_rc
#     DSC = S_mg - S_rc

#     return np.array([MCB, DSC, S_mg, S, S_var, count], dtype=np.float32)

def taggart_decomposition(x, y, t, tail):
    """
    Compute the empirical Taggart-score decomposition and uncertainty statistics, and contributions by contingency category.

    Upper-tail cases use x - t and y - t. The resulting zero-threshold upper-tail score equals S2 on the original scale.
    Lower-tail cases use t - x and t - y. The resulting zero-threshold upper-tail score equals S1 on the original scale.

    The selected tail score is decomposed in two different ways:

        S = MCB - DSC + UNC

    and

        S = S_HIT + S_MISS + S_FALSE_ALARM.

    The hit, miss, and false-alarm contributions are means over all valid forecast cases,
    with a value of zero whenever a case does not belong to the corresponding category.

    Parameters
    ----------
    x : np.ndarray
        Forecast values over time.
    y : np.ndarray
        Observed values over time.
    t : np.ndarray
        Threshold values over time.
    tail : {"upper", "lower"}
        Tail being evaluated.

    Returns
    -------
    np.ndarray
        Array of length 12 containing:
        [
            MCB,
            DSC,
            UNC,
            S,
            S_VAR,
            S_HIT,
            S_MISS,
            S_FALSE_ALARM,
            HIT_COUNT,
            MISS_COUNT,
            FALSE_ALARM_COUNT,
            COUNT,
        ]
        S is S2 for the upper tail and S1 for the lower tail.
    """
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'.")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)

    if not np.any(mask):
        return np.array([np.nan] * 8 + [0.0] * 4, dtype=np.float32)

    x = x[mask]
    y = y[mask]
    t = t[mask]

    if tail == "upper":
        x_tilde = x - t
        y_tilde = y - t
    else:
        x_tilde = t - x
        y_tilde = t - y

    score_values = upper_weighted_squared_error(x_tilde, y_tilde)

    # A positive transformed value means that the corresponding forecast or observation is in the relevant record tail.
    forecast_event = x_tilde > 0.0
    observed_event = y_tilde > 0.0

    hit = forecast_event & observed_event
    miss = ~forecast_event & observed_event
    false_alarm = forecast_event & ~observed_event

    # Each casewise score is assigned to exactly one relevant category. Correct non-detections receive a score of zero.
    hit_values = np.where(
        hit,
        score_values,
        0.0,
    )
    miss_values = np.where(
        miss,
        score_values,
        0.0,
    )
    false_alarm_values = np.where(
        false_alarm,
        score_values,
        0.0,
    )

    S = np.mean(score_values)
    S_var = np.var(score_values, ddof=0)

    # These are unconditional contributions. Each mean uses every valid forecast case, including zeros outside its category.
    S_hit = np.mean(hit_values)
    S_miss = np.mean(miss_values)
    S_false_alarm = np.mean(false_alarm_values)

    hit_count = np.count_nonzero(hit)
    miss_count = np.count_nonzero(miss)
    false_alarm_count = np.count_nonzero(false_alarm)
    count = x_tilde.size

    # If no observation enters the relevant tail, both the marginal forecast and every isotonic-regression fitted value are nonpositive.
    # Therefore, S_mg = S_rc = 0, so MCB = S and DSC = UNC = 0.
    if np.all(y_tilde <= 0):
        return np.array([S, 0.0, 0.0, S, S_var, S_hit, S_miss, S_false_alarm, hit_count, miss_count, false_alarm_count, count], dtype=np.float32)

    order = np.argsort(x_tilde, kind="mergesort")
    y_tilde_sorted = y_tilde[order]

    x_tilde_rc_sorted = isotonic_regression(y_tilde_sorted, increasing=True).x

    S_rc = np.mean(upper_weighted_squared_error(x_tilde_rc_sorted, y_tilde_sorted))
    S_mg = np.mean(upper_weighted_squared_error(np.mean(y_tilde), y_tilde))

    MCB = S - S_rc
    DSC = S_mg - S_rc

    return np.array([MCB, DSC, S_mg, S, S_var, S_hit, S_miss, S_false_alarm, hit_count, miss_count, false_alarm_count, count], dtype=np.float32)

def taggart_decomposition_along_time(forecast, target, threshold, tail):
    """Apply the Taggart decomposition along the time dimension."""
    names = component_names(tail)

    result = xr.apply_ufunc(
        taggart_decomposition,
        forecast,
        target,
        threshold,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"tail": tail},
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"component": len(names)}},
    )

    return result.assign_coords(component=("component", names))

def main(argv):
    del argv

    forecasts = open_zarr_auto(
        PREDICTION_PATH.value,
        decode_timedelta=True,
    ).sel(time=slice(TIME_START.value, TIME_STOP.value))

    if "lat" in forecasts.dims and "latitude" not in forecasts.dims:
        forecasts = forecasts.rename({"lat": "latitude"})
    if "lon" in forecasts.dims and "longitude" not in forecasts.dims:
        forecasts = forecasts.rename({"lon": "longitude"})

    forecasts = forecasts[VARIABLES.value].drop_vars("lead_time_secs", errors="ignore")

    if LEAD_TIMES_HOURS.value:
        forecasts = forecasts.sel(
            prediction_timedelta=[np.timedelta64(int(h), 'h') for h in LEAD_TIMES_HOURS.value]
        )

    if (EXPECTED_INITIALIZATIONS.value is not None and forecasts.sizes["time"] != EXPECTED_INITIALIZATIONS.value):
        raise ValueError(f"Found {forecasts.sizes['time']} initialization times; expected {EXPECTED_INITIALIZATIONS.value}.")

    forecast_chunks = {
        "time": -1,
        "prediction_timedelta": 1,
        "latitude": CHUNK_SIZE_LAT.value,
        "longitude": CHUNK_SIZE_LON.value,
    }
    if 'level' in forecasts.dims:
        forecast_chunks['level'] = 1

    forecasts = forecasts.chunk(forecast_chunks)

    target_start = forecasts.time[0] + forecasts.prediction_timedelta.min()
    target_stop = forecasts.time[-1] + forecasts.prediction_timedelta.max()

    targets = open_zarr_auto(TARGET_PATH.value)[VARIABLES.value].sel(time=slice(target_start, target_stop))

    required_valid_times = np.unique(
        forecasts.time.values[:, np.newaxis] + forecasts.prediction_timedelta.values[np.newaxis, :]
    ).ravel()
    missing_valid_times = np.setdiff1d(required_valid_times, targets.time.values)

    if missing_valid_times.size > 0:
        raise ValueError(f"Target data are missing {missing_valid_times.size} required valid times. First missing time: {missing_valid_times[0]}.")

    target_chunks = {
        "time": -1,
        "latitude": CHUNK_SIZE_LAT.value,
        "longitude": CHUNK_SIZE_LON.value,
    }
    if 'level' in targets.dims:
        target_chunks['level'] = 1

    targets = targets.sel(latitude=forecasts.latitude, longitude=forecasts.longitude).chunk(target_chunks)

    thresholds = open_zarr_auto(THRESHOLD_PATH.value)[VARIABLES.value]
    thresholds = (
        thresholds.sel(month=targets.time.dt.month)
        .reset_coords("month", drop=True)
        .sel(latitude=forecasts.latitude, longitude=forecasts.longitude)
        .chunk(target_chunks)
    )

    results = []

    for lt in forecasts.prediction_timedelta:
        valid_time = forecasts.time + lt

        forecast_lt = forecasts.sel(prediction_timedelta=lt).assign_coords(time=valid_time)
        target_lt = targets.sel(time=valid_time)
        threshold_lt = thresholds.sel(time=valid_time)

        results.append(
            taggart_decomposition_along_time(
                forecast_lt,
                target_lt,
                threshold_lt,
                TAIL.value,
            )
        )

    result = xr.concat(results, dim="prediction_timedelta")

    result_chunks = {
        'prediction_timedelta': 1,
        'latitude': CHUNK_SIZE_LAT.value,
        'longitude': CHUNK_SIZE_LON.value,
        'component': -1,
    }
    if 'level' in result.dims:
        result_chunks['level'] = 1

    result = result.chunk(result_chunks)

    if NUM_WORKERS.value is None:
        with ProgressBar():
            result.to_zarr(OUTPUT_PATH.value, mode="w", consolidated=True)
    else:
        with dask.config.set(scheduler="processes", num_workers=NUM_WORKERS.value):
            with ProgressBar():
                result.to_zarr(OUTPUT_PATH.value, mode="w", consolidated=True)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "prediction_path",
            "target_path",
            "threshold_path",
            "output_path",
            "time_start",
            "time_stop",
            "tail",
        ]
    )
    app.run(main)