"""
Script to compute MSE decomposition results for deterministic forecasts.

The mean squared error is decomposed as

    MSE = MCB - DSC + UNC,

where MCB denotes miscalibration, DSC discrimination, and UNC uncertainty. Empirical recalibration is performed using isotonic regression.

Reference
---------
Gneiting, T., & Resin, J. (2023).
Regression diagnostics meets forecast evaluation: Conditional calibration, reliability diagrams, and coefficient of determination.
Electronic Journal of Statistics, 17(2), 3226-3286.
"""
from absl import app
from absl import flags
import numpy as np
import xarray as xr
from scipy.optimize import isotonic_regression
from dask.diagnostics import ProgressBar
import dask


PREDICTION_PATH = flags.DEFINE_string(
    'prediction_path',
    None,
    help='Path to deterministic forecast dataset in Zarr format.',
)
TARGET_PATH = flags.DEFINE_string(
    'target_path',
    None,
    help='Path to target dataset in Zarr format.',
)
TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help='ISO 8601 initialization timestamp (inclusive) at which to start.',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help='ISO 8601 initialization timestamp (inclusive) at which to stop.',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    ['2m_temperature'],
    help='Variables to compute the MSE decomposition for.',
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='Where to save results.',
)
CHUNK_SIZE_LAT = flags.DEFINE_integer(
    'chunk_size_lat',
    11,
    help='Chunk size for latitude dimension.',
)
CHUNK_SIZE_LON = flags.DEFINE_integer(
    'chunk_size_lon',
    16,
    help='Chunk size for longitude dimension.',
)
LEVELS = flags.DEFINE_list(
    'levels',
    [],
    help='Optional comma-delimited list of pressure levels to select.',
)
LEAD_TIMES_HOURS = flags.DEFINE_list(
    'lead_times_hours',
    [],
    help='Optional list of lead times in hours. If empty, use all available lead times.',
)
THRESHOLD_PATH = flags.DEFINE_string(
    'threshold_path',
    None,
    help='Optional path to threshold dataset in Zarr format. If set, only timesteps with observed exceedances/non-exceedances relative to this threshold are used.',
)
EVENT_TARGET_PATH = flags.DEFINE_string(
    'event_target_path',
    None,
    help='Optional path to dataset used to define threshold exceedances. If unset, target_path is used.',
)
EXCEEDANCE = flags.DEFINE_bool(
    'exceedance',
    True,
    help='If True, keep times with y > threshold. If False, keep times with y < threshold.',
)
NUM_WORKERS = flags.DEFINE_integer(
    "num_workers",
    None,
    help="Number of Dask worker threads. If unset, use Dask's default.",
)
EXPECTED_INITIALIZATIONS = flags.DEFINE_integer(
    'expected_initializations',
    None,
    help='Optional expected number of forecast initialization times.',
)

COMPONENT_NAMES = np.array(['MCB', 'DSC', 'UNC', 'MSE', 'MSE_VAR', 'COUNT'], dtype='U')

def open_zarr_auto(path, **kwargs):
    """
    Open a Zarr store from either a GCS or local path.
    """
    if str(path).startswith('gs://'):
        return xr.open_zarr(store=path, storage_options={'token': 'anon'}, **kwargs)
    return xr.open_zarr(store=path, **kwargs)

def mse_decomposition(x, y):
    """
    Compute the empirical MSE decomposition and uncertainty statistics.

    Parameters
    ----------
    x : np.ndarray
        Forecast values over time.
    y : np.ndarray
        Observed values over time.

    Returns
    -------
    np.ndarray
        Array of length 6 containing:
        [MCB, DSC, UNC, MSE, MSE_VAR, COUNT]
    """
    mask = np.isfinite(x) & np.isfinite(y)

    if not np.any(mask):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.0], dtype=np.float32)

    x = x[mask]
    y = y[mask]

    order = np.argsort(x, kind='mergesort')
    y_sorted = y[order]

    x_rc_sorted = isotonic_regression(y_sorted, increasing=True).x

    squared_error = (x - y) ** 2

    S = np.mean(squared_error)
    S_var = np.var(squared_error, ddof=0)
    S_rc = np.mean((x_rc_sorted - y_sorted) ** 2)
    S_mg = np.var(y, ddof=0)

    MCB = S - S_rc
    DSC = S_mg - S_rc
    count = x.size

    return np.array([MCB, DSC, S_mg, S, S_var, count], dtype=np.float32)

def mse_decomposition_thresholded(x, y, event_target, threshold, exceedance=True):
    """
    Select only exceedances/non-exceedances relative to a threshold.
    """
    finite_mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(event_target)
        & np.isfinite(threshold)
    )

    if exceedance:
        event_mask = event_target > threshold
    else:
        event_mask = event_target < threshold

    mask = finite_mask & event_mask

    if not np.any(mask):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.0], dtype=np.float32)

    x = x[mask]
    y = y[mask]

    return mse_decomposition(x, y)

def mse_decomposition_along_time(
    forecast,
    target,
    threshold=None,
    event_target=None,
    exceedance=True,
):
    """
    Apply mse_decomposition along the time dimension.
    If threshold is given, only event-target exceedances/non-exceedances are used.
    """
    if threshold is None:
        result = xr.apply_ufunc(
            mse_decomposition,
            forecast,
            target,
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['component']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'output_sizes': {'component': len(COMPONENT_NAMES)}},
        )
    else:
        if event_target is None:
            event_target = target

        result = xr.apply_ufunc(
            mse_decomposition_thresholded,
            forecast,
            target,
            event_target,
            threshold,
            input_core_dims=[['time'], ['time'], ['time'], ['time']],
            output_core_dims=[['component']],
            vectorize=True,
            dask='parallelized',
            kwargs={'exceedance': exceedance},
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'output_sizes': {'component': len(COMPONENT_NAMES)}},
        )

    return result.assign_coords(component=('component', COMPONENT_NAMES))


def main(argv):
    del argv

    forecasts = open_zarr_auto(
        PREDICTION_PATH.value,
        decode_timedelta=True,
    ).sel(time=slice(TIME_START.value, TIME_STOP.value))

    # Handle different spatial dimension names for the high-resolution graphcast datasets
    if "lat" in forecasts.dims and "latitude" not in forecasts.dims:
        forecasts = forecasts.rename({"lat": "latitude"})
    if "lon" in forecasts.dims and "longitude" not in forecasts.dims:
        forecasts = forecasts.rename({"lon": "longitude"})

    forecasts = forecasts[VARIABLES.value].drop_vars("lead_time_secs", errors="ignore")

    if 'level' in forecasts.dims and LEVELS.value:
        forecasts = forecasts.sel(level=[int(level) for level in LEVELS.value])

    if LEAD_TIMES_HOURS.value:
        forecasts = forecasts.sel(
            prediction_timedelta=[np.timedelta64(int(h), 'h') for h in LEAD_TIMES_HOURS.value]
        )

    if (EXPECTED_INITIALIZATIONS.value is not None and forecasts.sizes["time"] != EXPECTED_INITIALIZATIONS.value):
        raise ValueError(f"Found {forecasts.sizes['time']} initialization times; expected {EXPECTED_INITIALIZATIONS.value}.")

    forecast_chunks = {
        'time': -1,
        'prediction_timedelta': 1,
        'latitude': CHUNK_SIZE_LAT.value,
        'longitude': CHUNK_SIZE_LON.value,
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
        raise ValueError(f"Target dataset is missing {missing_valid_times.size} required valid times. First missing time: {missing_valid_times[0]}.")

    if 'level' in targets.dims and LEVELS.value:
        targets = targets.sel(level=[int(level) for level in LEVELS.value])

    target_chunks = {
        'time': -1,
        'latitude': CHUNK_SIZE_LAT.value,
        'longitude': CHUNK_SIZE_LON.value,
    }
    if 'level' in targets.dims:
        target_chunks['level'] = 1

    targets = targets.sel(latitude=forecasts.latitude, longitude=forecasts.longitude).chunk(target_chunks)

    event_targets = targets
    if EVENT_TARGET_PATH.value is not None:
        event_targets = open_zarr_auto(EVENT_TARGET_PATH.value)[VARIABLES.value].sel(time=slice(target_start, target_stop))

        if 'level' in event_targets.dims and LEVELS.value:
            event_targets = event_targets.sel(level=[int(level) for level in LEVELS.value])

        event_targets = event_targets.chunk(target_chunks).sel(latitude=forecasts.latitude, longitude=forecasts.longitude)

    thresholds = None
    if THRESHOLD_PATH.value is not None:
        thresholds = open_zarr_auto(THRESHOLD_PATH.value)[VARIABLES.value]

        if 'level' in thresholds.dims and LEVELS.value:
            thresholds = thresholds.sel(level=[int(level) for level in LEVELS.value])

        thresholds = (
            thresholds
            .sel(month=targets.time.dt.month)
            .reset_coords('month', drop=True)
            .sel(latitude=forecasts.latitude, longitude=forecasts.longitude)
            .chunk(target_chunks)
        )

    results = []

    for lt in forecasts.prediction_timedelta:
        valid_time = forecasts.time + lt

        forecast_lt = forecasts.sel(prediction_timedelta=lt).assign_coords(time=valid_time)
        target_lt = targets.sel(time=valid_time)
        event_target_lt = event_targets.sel(time=valid_time)

        if thresholds is not None:
            threshold_lt = thresholds.sel(time=valid_time)
        else:
            threshold_lt = None

        results.append(
            mse_decomposition_along_time(
                forecast_lt,
                target_lt,
                threshold=threshold_lt,
                event_target=event_target_lt,
                exceedance=EXCEEDANCE.value,
            )
        )

    result = xr.concat(results, dim='prediction_timedelta')

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
        with dask.config.set(scheduler="threads", num_workers=NUM_WORKERS.value):
            with ProgressBar():
                result.to_zarr(OUTPUT_PATH.value, mode="w", consolidated=True)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'prediction_path',
        'target_path',
        'time_start',
        'time_stop',
        'output_path',
    ])
    app.run(main)