"""
Download the 2022 operational forecast and verification datasets.

The script selects the common 00 and 12 UTC initializations from 2022-01-01 to 2022-12-21,
the five evaluation lead times, and the two variables used in the analysis.
The initialization at 12 UTC on 2022-12-07 is excluded because the seven-day Pangu meridional
wind forecast is unavailable. The corresponding IFS analysis data cover all required verification times.

The datasets are stored on the common 1.5 degree and 0.25 degree model grids.
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc

OUTPUT_DIR = Path(os.environ["DATA_DIR"])

VARIABLES = ["2m_temperature", "10m_wind_speed"]

LEAD_TIMES = [
    np.timedelta64(1, "D"),
    np.timedelta64(3, "D"),
    np.timedelta64(5, "D"),
    np.timedelta64(7, "D"),
    np.timedelta64(10, "D"),
]

INITIALIZATION_TIME_RANGE = slice("2022-01-01T12:00:00", "2022-12-21T12:00:00")
INITIALIZATION_HOURS = [0, 12]
EXCLUDED_INITIALIZATIONS = np.array(["2022-12-07T12:00:00"], dtype="datetime64[ns]")

EXPECTED_INITIALIZATIONS = 708
EXPECTED_VALID_TIMES = 727

COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

GRID_CONFIGS = {
    "240x121": {
        "expected_longitudes": 240,
        "expected_latitudes": 121,
        "forecast_chunks": {
            "time": -1,
            "prediction_timedelta": 1,
            "longitude": 8,
            "latitude": 11,
        },
        "target_chunks": {
            "time": -1,
            "longitude": 8,
            "latitude": 11,
        },
        "hres_store": "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr",
        "pangu_store": "gs://weatherbench2/datasets/pangu_hres_init/2022_0012_240x121_equiangular_with_poles_conservative.zarr",
        "graphcast_store": "gs://weatherbench2/datasets/graphcast_hres_init_v2/2022-240x121_equiangular_with_poles_conservative.zarr",
        "aurora_store": "gs://weatherbench2/datasets/aurora/2022-240x121_equiangular_with_poles_conservative.zarr",
        "hres_t0_store": "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    },
    "0p25": {
        "expected_longitudes": 1440,
        "expected_latitudes": 720,
        "forecast_chunks": {
            "time": -1,
            "prediction_timedelta": 1,
            "longitude": 60,
            "latitude": 60,
        },
        "target_chunks": {
            "time": -1,
            "longitude": 60,
            "latitude": 60,
        },
        "hres_store": "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        "pangu_store": "gs://weatherbench2/datasets/pangu_hres_init/2022_0012_0p25.zarr",
        "graphcast_store": "gs://weatherbench2/datasets/graphcast_hres_init_v2/2022-1440x721.zarr",
        "aurora_store": "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr",
        "hres_t0_store": "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
    },
}

def standardize_spatial_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    """Rename lat/lon dimensions to latitude/longitude."""
    rename = {}

    if "latitude" not in dataset.dims and "lat" in dataset.dims:
        rename["lat"] = "latitude"

    if "longitude" not in dataset.dims and "lon" in dataset.dims:
        rename["lon"] = "longitude"

    return dataset.rename(rename)


def select_00_12_utc(dataset: xr.Dataset) -> xr.Dataset:
    """Retain only times at 00 and 12 UTC."""
    time_mask = dataset.time.dt.hour.isin(INITIALIZATION_HOURS).values
    return dataset.isel(time=time_mask)

def open_forecasts(store: str) -> xr.Dataset:
    """Open and lazily select the required forecast data."""
    dataset = xr.open_zarr(
        store=store,
        storage_options={"token": "anon"},
        decode_timedelta=True,
    )
    dataset = standardize_spatial_coordinates(dataset)
    dataset = dataset.sel(time=INITIALIZATION_TIME_RANGE, prediction_timedelta=LEAD_TIMES)[VARIABLES]
    dataset = select_00_12_utc(dataset)
    dataset = dataset.drop_sel(time=EXCLUDED_INITIALIZATIONS)
    dataset = dataset.drop_vars("lead_time_secs", errors="ignore")
    return dataset

def find_common_coordinate(datasets: list[xr.Dataset], coordinate: str) -> np.ndarray:
    """Find coordinate values shared by every dataset."""
    common_values = datasets[0][coordinate].values

    for dataset in datasets[1:]:
        common_values = np.intersect1d(common_values, dataset[coordinate].values)

    return common_values

def prepare_forecast(
    dataset: xr.Dataset,
    common_times: np.ndarray,
    common_latitude: np.ndarray,
    common_longitude: np.ndarray,
    chunks: dict,
) -> xr.Dataset:
    """Prepare one forecast dataset for local storage."""
    dataset = dataset.sel(time=common_times, latitude=common_latitude, longitude=common_longitude)
    dataset = dataset.assign_coords(latitude=common_latitude, longitude=common_longitude)
    dataset = dataset.astype({variable: np.float32 for variable in VARIABLES})
    return dataset.chunk(chunks)

def open_hres_t0(
    store: str,
    required_valid_times: np.ndarray,
    common_latitude: np.ndarray,
    common_longitude: np.ndarray,
    chunks: dict,
) -> xr.Dataset:
    """Open IFS analyses for every required forecast verification time."""
    dataset = xr.open_zarr(
        store=store,
        storage_options={"token": "anon"},
        decode_timedelta=True,
    )
    dataset = standardize_spatial_coordinates(dataset)
    dataset = dataset.sel(time=required_valid_times, latitude=common_latitude, longitude=common_longitude)[VARIABLES]
    dataset = dataset.assign_coords(latitude=common_latitude, longitude=common_longitude)
    dataset = dataset.astype({variable: np.float32 for variable in VARIABLES})
    return dataset.chunk(chunks)

def remove_source_encoding(dataset: xr.Dataset) -> xr.Dataset:
    """Remove inherited source encodings before writing."""
    dataset = dataset.copy()

    for variable in dataset.variables:
        dataset[variable].encoding = {}

    return dataset

def create_output_encoding(dataset: xr.Dataset) -> dict:
    """Create a consistent encoding for the local Zarr store."""
    encoding = {variable: {"compressor": COMPRESSOR} for variable in dataset.variables}

    for variable in VARIABLES:
        if variable in dataset:
            encoding[variable]["dtype"] = "float32"
            encoding[variable]["_FillValue"] = np.nan

    return encoding

def check_forecast_times(
    dataset: xr.Dataset,
    name: str,
    reference_times: np.ndarray | None = None,
) -> None:
    """Check that a forecast dataset has the intended common time grid."""
    times = dataset.time.values
    hours = np.unique(dataset.time.dt.hour.values)

    if len(times) != EXPECTED_INITIALIZATIONS:
        raise ValueError(f"{name} has {len(times)} times; expected {EXPECTED_INITIALIZATIONS}.")
    if not np.array_equal(hours, np.array(INITIALIZATION_HOURS)):
        raise ValueError(f"{name} has initialization hours {hours}.")
    if reference_times is not None and not np.array_equal(times, reference_times):
        raise ValueError(f"{name} does not have the common initialization times.")

    print(f"{name}: {len(times)} initializations from {times[0]} to {times[-1]}")

def check_common_grid(
    common_latitude: np.ndarray,
    common_longitude: np.ndarray,
    grid_name: str,
    config: dict,
) -> None:
    """Check the size of the common spatial grid."""
    if len(common_latitude) != config["expected_latitudes"]:
        raise ValueError(f"{grid_name} has {len(common_latitude)} common latitudes; expected {config['expected_latitudes']}.")
    if len(common_longitude) != config["expected_longitudes"]:
        raise ValueError(f"{grid_name} has {len(common_longitude)} common longitudes; expected {config['expected_longitudes']}.")

    print(f"{grid_name} common grid: {len(common_longitude)} longitudes × {len(common_latitude)} latitudes")

def save_dataset(dataset: xr.Dataset, output_path: Path) -> None:
    """Write one dataset to a consolidated local Zarr store."""
    dataset = remove_source_encoding(dataset)
    encoding = create_output_encoding(dataset)

    print(f"Saving {output_path}")

    with ProgressBar():
        dataset.to_zarr(output_path, mode="w", consolidated=True, encoding=encoding)

def process_grid(grid_name: str, config: dict) -> None:
    """Prepare and save all datasets for one spatial resolution."""
    print(f"\nOpening {grid_name} forecast datasets")

    forecasts = {
        "hres": open_forecasts(config["hres_store"]),
        "pangu": open_forecasts(config["pangu_store"]),
        "graphcast": open_forecasts(config["graphcast_store"]),
        "aurora": open_forecasts(config["aurora_store"]),
    }

    reference_times = forecasts["graphcast"].time.values

    check_forecast_times(forecasts["graphcast"], "GraphCast operational")
    check_forecast_times(forecasts["hres"], "HRES", reference_times)
    check_forecast_times(forecasts["pangu"], "Pangu operational", reference_times)
    check_forecast_times(forecasts["aurora"], "Aurora", reference_times)

    forecast_list = list(forecasts.values())
    common_latitude = find_common_coordinate(forecast_list, "latitude").astype(np.float64)
    common_longitude = find_common_coordinate(forecast_list, "longitude").astype(np.float64)

    check_common_grid(common_latitude, common_longitude, grid_name, config)

    for name, dataset in forecasts.items():
        forecasts[name] = prepare_forecast(
            dataset,
            reference_times,
            common_latitude,
            common_longitude,
            config["forecast_chunks"],
        )

    required_valid_times = np.unique(
        reference_times[:, np.newaxis] + np.asarray(LEAD_TIMES)[np.newaxis, :]
    )

    if len(required_valid_times) != EXPECTED_VALID_TIMES:
        raise ValueError(f"{grid_name} requires {len(required_valid_times)} valid times; expected {EXPECTED_VALID_TIMES}.")

    print(f"Opening {grid_name} IFS analysis data")

    hres_t0 = open_hres_t0(
        config["hres_t0_store"],
        required_valid_times,
        common_latitude,
        common_longitude,
        config["target_chunks"],
    )

    if not np.array_equal(hres_t0.time.values, required_valid_times):
        raise ValueError(f"{grid_name} IFS analysis does not contain the exact required valid times.")

    print(f"IFS analysis: {hres_t0.sizes['time']} times from {hres_t0.time.values[0]} to {hres_t0.time.values[-1]}")

    outputs = [
        (forecasts["hres"], OUTPUT_DIR / f"hres_{grid_name}_2022.zarr"),
        (forecasts["pangu"], OUTPUT_DIR / f"pangu_operational_{grid_name}_2022.zarr"),
        (forecasts["graphcast"], OUTPUT_DIR / f"graphcast_operational_{grid_name}_2022.zarr"),
        (forecasts["aurora"], OUTPUT_DIR / f"aurora_{grid_name}_2022.zarr"),
        (hres_t0, OUTPUT_DIR / f"hres_t0_{grid_name}_2022_eval_times.zarr"),
    ]

    for dataset, output_path in outputs:
        save_dataset(dataset, output_path)

def main() -> None:
    """Prepare and save all datasets required for the 2022 evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for grid_name, config in GRID_CONFIGS.items():
        process_grid(grid_name, config)

    print("\nAll datasets were saved successfully.")


if __name__ == "__main__":
    main()