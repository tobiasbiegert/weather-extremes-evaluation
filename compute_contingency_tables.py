"""Compute contingency statistics and twMSE contribution tables.

The script produces pooled contingency statistics and twMSE shares for the land and global domains.
It also produces latitude-weighted twMSE contributions by lead time for both domains.
"""

import os
from pathlib import Path

from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import xarray as xr
from weatherbench2.metrics import get_lat_weights

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def get_environment_path(name: str) -> Path:
    """Return a required path from an environment variable."""
    value = os.environ.get(name)

    if value is None:
        raise RuntimeError(
            f"{name} is not set. Load local_paths.env before running this script."
        )

    return Path(value)


DATA_DIR = get_environment_path("DATA_DIR")
RESULTS_DIR = get_environment_path("RESULTS_DIR")

TAGGART_RESULTS_DIR = RESULTS_DIR / "taggart" / "0p25_t2017_y2022"

GRID_REFERENCE_PATH = DATA_DIR / "hres_0p25_2022.zarr"

LAND_SEA_MASK_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------
MODEL_STEMS = {
    "HRES": "hres",
    "Pangu-Weather": "pangu_operational",
    "GraphCast": "graphcast_operational",
    "Aurora": "aurora",
}

RECORD_CONFIGS = {
    "Heat": {
        "lead_time_label": "Heat records",
        "variable": "2m_temperature",
        "threshold": "max",
        "score": "S2",
    },
    "Cold": {
        "lead_time_label": "Cold records",
        "variable": "2m_temperature",
        "threshold": "min",
        "score": "S1",
    },
    "Wind": {
        "lead_time_label": "Wind records",
        "variable": "10m_wind_speed",
        "threshold": "max",
        "score": "S2",
    },
}

LEAD_TIME_DAYS = [1, 3, 5, 7, 10]

LEAD_TIMES = [
    np.timedelta64(day, "D")
    for day in LEAD_TIME_DAYS
]

DOMAIN_KEYS = [
    "land",
    "global",
]

COUNT_COMPONENTS = [
    "HIT_COUNT",
    "MISS_COUNT",
    "FALSE_ALARM_COUNT",
]

SPATIAL_DIMS = (
    "latitude",
    "longitude",
)

POOLING_DIMS = (
    "prediction_timedelta",
    *SPATIAL_DIMS,
)

CONTINGENCY_COLUMNS = [
    "MR [%]",
    "FAR [%]",
    "FB",
    "Hit share [%]",
    "Miss share [%]",
    "False-alarm share [%]",
]

CONTINGENCY_DECIMALS = {
    "MR [%]": 2,
    "FAR [%]": 2,
    "FB": 3,
    "Hit share [%]": 2,
    "Miss share [%]": 2,
    "False-alarm share [%]": 2,
}

CONTRIBUTION_COLUMNS = [
    "Taggart score",
    "Hit contribution",
    "Miss contribution",
    "False-alarm contribution",
]

# ---------------------------------------------------------------------
# Spatial data
# ---------------------------------------------------------------------
def select_common_grid(
    obj: xr.Dataset | xr.DataArray,
    common_grid: xr.Dataset,
) -> xr.Dataset | xr.DataArray:
    """Select an xarray object on the common evaluation grid."""
    latitude = common_grid.latitude.values
    longitude = common_grid.longitude.values

    obj = obj.sortby(
        ["latitude", "longitude"]
    ).sel(
        latitude=latitude,
        longitude=longitude,
    )

    return obj.assign_coords(
        latitude=latitude,
        longitude=longitude,
    )

def load_spatial_data() -> tuple[
    xr.Dataset,
    xr.DataArray,
    xr.DataArray,
]:
    """Load the common grid, latitude weights, and domain masks."""
    grid_source = xr.open_zarr(
        GRID_REFERENCE_PATH,
        decode_timedelta=True,
    )

    common_grid = xr.Dataset(
        coords={
            "latitude": np.sort(
                grid_source.latitude.values.astype(np.float64)
            ),
            "longitude": np.sort(
                grid_source.longitude.values.astype(np.float64)
            ),
        }
    )

    grid_source.close()

    mask_source = xr.open_zarr(
        LAND_SEA_MASK_STORE,
        storage_options={"token": "anon"},
    )

    land_sea_mask = mask_source["land_sea_mask"]

    if "time" in land_sea_mask.dims:
        land_sea_mask = land_sea_mask.isel(
            time=0,
            drop=True,
        )

    land_sea_mask = select_common_grid(
        land_sea_mask,
        common_grid,
    ).load()

    mask_source.close()

    latitude_weights = (
        get_lat_weights(common_grid)
        .astype(np.float64)
        .load()
    )

    land_mask = (
        (land_sea_mask > 0.5)
        & (common_grid.latitude > -60)
    ).load()

    global_mask = xr.ones_like(
        land_mask,
        dtype=bool,
    )

    domain_masks = xr.concat(
        [
            land_mask,
            global_mask,
        ],
        dim=xr.IndexVariable(
            "domain",
            DOMAIN_KEYS,
        ),
    ).load()

    return (
        common_grid,
        latitude_weights,
        domain_masks,
    )

# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def aggregate_components(
    data: xr.DataArray,
    score_component: str,
    latitude_weights: xr.DataArray,
    domain_masks: xr.DataArray,
) -> xr.Dataset:
    """Pool counts and compute latitude-weighted score contributions."""
    count_totals = (
        data
        .sel(component=COUNT_COMPONENTS)
        .astype(np.int64)
        .where(domain_masks, other=0)
        .sum(POOLING_DIMS)
    )

    total_count = (
        data
        .sel(component="COUNT", drop=True)
        .astype(np.float64)
    )

    score_components = [
        score_component,
        f"{score_component}_HIT",
        f"{score_component}_MISS",
        f"{score_component}_FALSE_ALARM",
    ]

    scores = (
        data
        .sel(component=score_components)
        .astype(np.float64)
    )

    # The stored scores are temporal means. Multiplication by COUNT converts them back to totals over valid forecast cases.
    score_weights = (
        latitude_weights
        * total_count.where(
            domain_masks,
            other=0.0,
        )
    ).fillna(0.0)

    weighted_score_sums_by_lead = (
        scores
        * score_weights
    ).sum(
        SPATIAL_DIMS,
        skipna=True,
    )

    # Exclude the weights of missing score values from the denominator.
    valid_score_weights = score_weights.where(
        scores.notnull(),
        other=0.0,
    )

    score_weight_sums_by_lead = (
        valid_score_weights
        .sum(SPATIAL_DIMS)
    )

    weighted_score_means_by_lead = (
        weighted_score_sums_by_lead
        / score_weight_sums_by_lead
    )

    weighted_score_totals = (
        weighted_score_sums_by_lead
        .sum("prediction_timedelta")
    )

    summary = xr.Dataset({
        "hits": count_totals.sel(
            component="HIT_COUNT",
            drop=True,
        ),
        "misses": count_totals.sel(
            component="MISS_COUNT",
            drop=True,
        ),
        "false_alarms": count_totals.sel(
            component="FALSE_ALARM_COUNT",
            drop=True,
        ),
        "twmse": weighted_score_totals.sel(
            component=score_component,
            drop=True,
        ),
        "hit_contribution": weighted_score_totals.sel(
            component=f"{score_component}_HIT",
            drop=True,
        ),
        "miss_contribution": weighted_score_totals.sel(
            component=f"{score_component}_MISS",
            drop=True,
        ),
        "false_alarm_contribution": weighted_score_totals.sel(
            component=f"{score_component}_FALSE_ALARM",
            drop=True,
        ),
        "twmse_by_lead": weighted_score_means_by_lead.sel(
            component=score_component,
            drop=True,
        ),
        "hit_contribution_by_lead": weighted_score_means_by_lead.sel(
            component=f"{score_component}_HIT",
            drop=True,
        ),
        "miss_contribution_by_lead": weighted_score_means_by_lead.sel(
            component=f"{score_component}_MISS",
            drop=True,
        ),
        "false_alarm_contribution_by_lead": (
            weighted_score_means_by_lead.sel(
                component=f"{score_component}_FALSE_ALARM",
                drop=True,
            )
        ),
    })

    with ProgressBar():
        return summary.compute()

def ratio(
    numerator: float,
    denominator: float,
    scale: float = 1.0,
) -> float:
    """Return a scaled ratio, or NaN if the denominator is zero."""
    if denominator == 0:
        return np.nan

    return scale * numerator / denominator

# ---------------------------------------------------------------------
# Pooled contingency and share tables
# ---------------------------------------------------------------------
def make_contingency_row(
    summary: xr.Dataset,
    domain: str,
    record_type: str,
    model: str,
) -> dict:
    """Construct one row of a pooled contingency and share table."""
    values = summary.sel(domain=domain)

    hits = int(values["hits"].item())
    misses = int(values["misses"].item())
    false_alarms = int(values["false_alarms"].item())

    observed_records = hits + misses
    forecast_records = hits + false_alarms

    twmse = float(values["twmse"].item())
    hit_contribution = float(
        values["hit_contribution"].item()
    )
    miss_contribution = float(
        values["miss_contribution"].item()
    )
    false_alarm_contribution = float(
        values["false_alarm_contribution"].item()
    )

    return {
        "Record type": record_type,
        "Model": model,
        "MR [%]": ratio(
            misses,
            observed_records,
            scale=100.0,
        ),
        "FAR [%]": ratio(
            false_alarms,
            forecast_records,
            scale=100.0,
        ),
        "FB": ratio(
            forecast_records,
            observed_records,
        ),
        "Hit share [%]": ratio(
            hit_contribution,
            twmse,
            scale=100.0,
        ),
        "Miss share [%]": ratio(
            miss_contribution,
            twmse,
            scale=100.0,
        ),
        "False-alarm share [%]": ratio(
            false_alarm_contribution,
            twmse,
            scale=100.0,
        ),
    }

def make_contingency_table(
    rows: list[dict],
) -> pd.DataFrame:
    """Construct a pooled contingency and share table."""
    index = pd.MultiIndex.from_product(
        [
            list(RECORD_CONFIGS),
            list(MODEL_STEMS),
        ],
        names=[
            "Record type",
            "Model",
        ],
    )

    table = (
        pd.DataFrame(rows)
        .set_index([
            "Record type",
            "Model",
        ])
        .reindex(index)
    )

    return table[
        CONTINGENCY_COLUMNS
    ].round(
        CONTINGENCY_DECIMALS
    )

# ---------------------------------------------------------------------
# Lead-time contribution tables
# ---------------------------------------------------------------------
def make_contribution_rows(
    summary: xr.Dataset,
    domain: str,
    record_type: str,
    model: str,
) -> list[dict]:
    """Construct contribution rows for all evaluated lead times."""
    rows = []

    record_label = RECORD_CONFIGS[
        record_type
    ]["lead_time_label"]

    for lead_time_days, lead_time in zip(
        LEAD_TIME_DAYS,
        LEAD_TIMES,
    ):
        selection = {
            "domain": domain,
            "prediction_timedelta": lead_time,
        }

        rows.append({
            "Record type": record_label,
            "Model": model,
            "Lead time [d]": lead_time_days,
            "Taggart score": float(
                summary["twmse_by_lead"]
                .sel(selection)
                .item()
            ),
            "Hit contribution": float(
                summary["hit_contribution_by_lead"]
                .sel(selection)
                .item()
            ),
            "Miss contribution": float(
                summary["miss_contribution_by_lead"]
                .sel(selection)
                .item()
            ),
            "False-alarm contribution": float(
                summary["false_alarm_contribution_by_lead"]
                .sel(selection)
                .item()
            ),
        })

    return rows

def make_contribution_table(
    rows: list[dict],
) -> pd.DataFrame:
    """Construct an ordered lead-time contribution table."""
    record_order = [
        config["lead_time_label"]
        for config in RECORD_CONFIGS.values()
    ]

    index = pd.MultiIndex.from_product(
        [
            record_order,
            list(MODEL_STEMS),
            LEAD_TIME_DAYS,
        ],
        names=[
            "Record type",
            "Model",
            "Lead time [d]",
        ],
    )

    return (
        pd.DataFrame(rows)
        .set_index([
            "Record type",
            "Model",
            "Lead time [d]",
        ])
        .reindex(index)[CONTRIBUTION_COLUMNS]
        .reset_index()
    )

# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------
def main() -> dict[str, pd.DataFrame]:
    common_grid, latitude_weights, domain_masks = (
        load_spatial_data()
    )

    contingency_rows = {
        domain: []
        for domain in DOMAIN_KEYS
    }

    contribution_rows = {
        domain: []
        for domain in DOMAIN_KEYS
    }

    for record_type, record_config in RECORD_CONFIGS.items():
        for model, model_stem in MODEL_STEMS.items():
            filename = (
                f"{model_stem}_0p25_vs_ifs_analysis_monthly_{record_config['threshold']}_record.zarr"
            )

            result_path = (
                TAGGART_RESULTS_DIR
                / filename
            )

            if not result_path.exists():
                raise FileNotFoundError(
                    f"Missing Taggart result: {result_path}"
                )

            print(
                f"Processing {record_type}, {model}: {result_path}"
            )

            result_source = xr.open_zarr(
                result_path,
                consolidated=True,
                decode_timedelta=True,
            )

            try:
                result = result_source.sel(
                    prediction_timedelta=LEAD_TIMES,
                )

                data = select_common_grid(
                    result[record_config["variable"]],
                    common_grid,
                )

                summary = aggregate_components(
                    data=data,
                    score_component=record_config["score"],
                    latitude_weights=latitude_weights,
                    domain_masks=domain_masks,
                )
            finally:
                result_source.close()

            for domain in DOMAIN_KEYS:
                contingency_rows[domain].append(
                    make_contingency_row(
                        summary=summary,
                        domain=domain,
                        record_type=record_type,
                        model=model,
                    )
                )

                contribution_rows[domain].extend(
                    make_contribution_rows(
                        summary=summary,
                        domain=domain,
                        record_type=record_type,
                        model=model,
                    )
                )

    generated_tables = {
        "land_contingency_and_shares": make_contingency_table(contingency_rows["land"]),
        "global_contingency_and_shares": make_contingency_table(contingency_rows["global"]),
        "land_contributions_by_lead_time": make_contribution_table(contribution_rows["land"]),
        "global_contributions_by_lead_time": make_contribution_table(contribution_rows["global"]),
    }

    output_paths = {
        "land_contingency_and_shares": TAGGART_RESULTS_DIR / "record_contingency_and_twmse_shares_0p25_land_t2017_y2022.csv",
        "global_contingency_and_shares": TAGGART_RESULTS_DIR / "record_contingency_and_twmse_shares_0p25_global_t2017_y2022.csv",
        "land_contributions_by_lead_time": TAGGART_RESULTS_DIR / "taggart_score_contributions_0p25_land_by_lead_time_t2017_y2022.csv",
        "global_contributions_by_lead_time": TAGGART_RESULTS_DIR / "taggart_score_contributions_0p25_global_by_lead_time_t2017_y2022.csv",
    }

    for name in [
        "land_contingency_and_shares",
        "global_contingency_and_shares",
    ]:
        generated_tables[name].to_csv(
            output_paths[name],
            index=True,
            na_rep="NaN",
        )

    for name in [
        "land_contributions_by_lead_time",
        "global_contributions_by_lead_time",
    ]:
        generated_tables[name].to_csv(
            output_paths[name],
            index=False,
            na_rep="NaN",
        )

    print("\nLand contingency statistics and twMSE shares:")
    print(generated_tables["land_contingency_and_shares"].to_string())

    print("\nGlobal contingency statistics and twMSE shares:")
    print(generated_tables["global_contingency_and_shares"].to_string())

    for output_path in output_paths.values():
        print(f"\nSaved: {output_path}")

    return generated_tables

if __name__ == "__main__":
    generated_tables = main()