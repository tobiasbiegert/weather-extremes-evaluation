"""
Compute monthly record-based threshold datasets from ERA5.
"""

import logging
import os

import dask
import xarray as xr
from dask.diagnostics import ProgressBar


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MAX_VARIABLES = [
    "2m_temperature",
    "10m_wind_speed",
]

MIN_VARIABLES = [
    "2m_temperature",
]

VARIABLES = sorted(set(MAX_VARIABLES + MIN_VARIABLES))

ERA5_INPUT_PATHS = {
    "240x121": (
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    ),
    "1440x721": (
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    ),
}

DATASETS_TO_COMPUTE = ["240x121", "1440x721"]

TIME_START = "1979-01-01"
TIME_STOP_2017 = "2017-12-31"
TIME_START_EXTRA = "2018-01-01"
TIME_STOP_2019 = "2019-12-31"


def monthly_records(ds):
    """Compute monthly record maxima and minima for selected variables."""
    record_max = (
        ds[MAX_VARIABLES]
        .groupby("time.month")
        .max(dim="time", skipna=True)
        .astype("float32")
    )

    record_min = (
        ds[MIN_VARIABLES]
        .groupby("time.month")
        .min(dim="time", skipna=True)
        .astype("float32")
    )

    return record_max, record_min


def main():
    thresholds_dir = os.environ["THRESHOLDS_DIR"]

    for resolution in DATASETS_TO_COMPUTE:
        input_path = ERA5_INPUT_PATHS[resolution]

        output_max_2017_path = f"{thresholds_dir}/era5_{resolution}_record_max_month_1979_2017.zarr"
        output_min_2017_path = f"{thresholds_dir}/era5_{resolution}_record_min_month_1979_2017.zarr"
        output_max_2019_path = f"{thresholds_dir}/era5_{resolution}_record_max_month_1979_2019.zarr"
        output_min_2019_path = f"{thresholds_dir}/era5_{resolution}_record_min_month_1979_2019.zarr"

        logging.info("Opening ERA5 dataset for %s.", resolution)
        era5 = (
            xr.open_zarr(
                store=input_path,
                storage_options={"token": "anon"},
                decode_timedelta=True,
            )
            .sel(time=slice(TIME_START, TIME_STOP_2019))[VARIABLES]
        )

        era5_1979_2017 = era5.sel(time=slice(TIME_START, TIME_STOP_2017))
        era5_2018_2019 = era5.sel(time=slice(TIME_START_EXTRA, TIME_STOP_2019))

        logging.info("Building monthly records for %s, 1979-2017.", resolution)
        record_max_2017, record_min_2017 = monthly_records(era5_1979_2017)

        logging.info("Computing monthly records for %s, 1979-2017.", resolution)
        with ProgressBar():
            record_max_2017, record_min_2017 = dask.compute(
                record_max_2017,
                record_min_2017,
            )

        logging.info("Writing %s max records 1979-2017.", resolution)
        record_max_2017.to_zarr(
            output_max_2017_path,
            mode="w",
            consolidated=True,
        )

        logging.info("Writing %s min records 1979-2017.", resolution)
        record_min_2017.to_zarr(
            output_min_2017_path,
            mode="w",
            consolidated=True,
        )

        logging.info("Building monthly records for %s, 2018-2019.", resolution)
        record_max_extra, record_min_extra = monthly_records(era5_2018_2019)

        logging.info("Computing monthly records for %s, 2018-2019.", resolution)
        with ProgressBar():
            record_max_extra, record_min_extra = dask.compute(
                record_max_extra,
                record_min_extra,
            )

        logging.info("Combining records for %s to get 1979-2019.", resolution)
        record_max_2019 = (
            xr.concat([record_max_2017, record_max_extra], dim="period_part")
            .max(dim="period_part", skipna=True)
            .astype("float32")
        )

        record_min_2019 = (
            xr.concat([record_min_2017, record_min_extra], dim="period_part")
            .min(dim="period_part", skipna=True)
            .astype("float32")
        )

        logging.info("Writing %s max records 1979-2019.", resolution)
        record_max_2019.to_zarr(
            output_max_2019_path,
            mode="w",
            consolidated=True,
        )

        logging.info("Writing %s min records 1979-2019.", resolution)
        record_min_2019.to_zarr(
            output_min_2019_path,
            mode="w",
            consolidated=True,
        )

        logging.info("Finished %s.", resolution)

    logging.info("Finished all requested datasets.")


if __name__ == "__main__":
    main()