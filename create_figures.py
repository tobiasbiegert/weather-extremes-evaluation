"""Create the figures used in the paper and appendix.

The script loads the final 0.25-degree MSE, PCRPS, and twMSE results, calculates the required spatial averages,
and creates the selected figures for the land and global domains.
"""

import os
from pathlib import Path

import dask
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import stats
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

MSE_RESULTS_DIR = RESULTS_DIR / "mse"
PCRPS_RESULTS_DIR = RESULTS_DIR / "pcrps"
TAGGART_RESULTS_DIR = RESULTS_DIR / "taggart"
PLOTS_DIR = Path("plots")

RESOLUTION = "0p25"

GRID_REFERENCE_PATH = DATA_DIR / "hres_0p25_2022.zarr"

LAND_SEA_MASK_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

TAGGART_TABLE_DIR = TAGGART_RESULTS_DIR / "0p25_t2017_y2022"


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

MODEL_STEMS = {
    "HRES": "hres",
    "Pangu-Weather": "pangu_operational",
    "GraphCast": "graphcast_operational",
    "Aurora": "aurora",
}

EVENT_CONFIGS = {
    "all": {
        "suffix": "",
        "variables": [
            "2m_temperature",
            "10m_wind_speed",
        ],
    },
    "monthly_max": {
        "suffix": "_monthly_max_record",
        "variables": [
            "2m_temperature",
            "10m_wind_speed",
        ],
    },
    "monthly_min": {
        "suffix": "_monthly_min_record",
        "variables": [
            "2m_temperature",
        ],
    },
}

TAGGART_EVENT_CONFIGS = {
    "monthly_max": {
        "suffix": "_monthly_max_record",
        "variables": [
            "2m_temperature",
            "10m_wind_speed",
        ],
        "score_component": "S2",
    },
    "monthly_min": {
        "suffix": "_monthly_min_record",
        "variables": [
            "2m_temperature",
        ],
        "score_component": "S1",
    },
}

PCRPS_VARIABLE_SUFFIXES = {
    "2m_temperature": "t2m",
    "10m_wind_speed": "ws10",
}

PCRPS_READ_CHUNKS = {
    "time": -1,
    "prediction_timedelta": 1,
    "latitude": 120,
    "longitude": 120,
}

DOMAINS = (
    "global",
    "land",
)

COLORS = {
    "HRES": "black",
    "Pangu-Weather": "tomato",
    "GraphCast": "tab:blue",
    "Aurora": "olive",
}

RECORD_DECOMPOSITION_PANELS = (
    (
        "monthly_max",
        "2m_temperature",
        "Heat records",
    ),
    (
        "monthly_min",
        "2m_temperature",
        "Cold records",
    ),
    (
        "monthly_max",
        "10m_wind_speed",
        "Wind records",
    ),
)

MODEL_ORDER = list(MODEL_STEMS)

RECORD_ORDER = [
    "Heat records",
    "Cold records",
    "Wind records",
]

LEAD_TIME_DAYS = [
    1,
    3,
    5,
    7,
    10,
]

TOTAL_SCORE_COLUMN = "Taggart score"

CONTRIBUTION_STYLES = {
    "Hit contribution": {
        "label": "Hits",
        "color": "#009E73",
    },
    "Miss contribution": {
        "label": "Misses",
        "color": "#E69F00",
    },
    "False-alarm contribution": {
        "label": "False alarms",
        "color": "#CC79A7",
    },
}

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


# ---------------------------------------------------------------------
# Shared spatial processing
# ---------------------------------------------------------------------

def select_common_grid(
    obj: xr.Dataset | xr.DataArray,
    common_grid: xr.Dataset,
) -> xr.Dataset | xr.DataArray:
    """Select and standardize an object on the common evaluation grid."""
    common_latitude = common_grid.latitude.values
    common_longitude = common_grid.longitude.values

    obj = obj.sortby(
        [
            "latitude",
            "longitude",
        ]
    ).sel(
        latitude=common_latitude,
        longitude=common_longitude,
    )

    return obj.assign_coords(
        latitude=common_latitude,
        longitude=common_longitude,
    )


def load_spatial_data() -> dict:
    """Load the common grid, latitude weights, and land mask."""
    grid_source = xr.open_zarr(
        GRID_REFERENCE_PATH,
        decode_timedelta=True,
    )

    try:
        common_grid = xr.Dataset(
            coords={
                "latitude": np.sort(
                    grid_source.latitude.values.astype(
                        np.float64
                    )
                ),
                "longitude": np.sort(
                    grid_source.longitude.values.astype(
                        np.float64
                    )
                ),
            }
        )
    finally:
        grid_source.close()

    mask_source = xr.open_zarr(
        LAND_SEA_MASK_STORE,
        storage_options={
            "token": "anon",
        },
    )

    try:
        land_sea_mask = mask_source[
            "land_sea_mask"
        ]

        if "time" in land_sea_mask.dims:
            land_sea_mask = land_sea_mask.isel(
                time=0,
                drop=True,
            )

        land_sea_mask = select_common_grid(
            land_sea_mask,
            common_grid,
        ).load()
    finally:
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

    return {
        "common_grid": common_grid,
        "latitude_weights": latitude_weights,
        "land_mask": land_mask,
    }


def spatial_average_decomposition_result(
    dataset: xr.Dataset,
    score_component: str,
    spatial_data: dict,
    domain: str,
) -> xr.Dataset:
    """Spatially average one local score decomposition result."""
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain {domain!r}."
        )

    dataset = select_common_grid(
        dataset,
        spatial_data["common_grid"],
    ).astype(np.float64)

    decomposition_components = [
        "MCB",
        "DSC",
        "UNC",
        score_component,
    ]

    variance_component = (
        f"{score_component}_VAR"
    )

    required_components = set(
        decomposition_components
        + [
            variance_component,
            "COUNT",
        ]
    )

    available_components = set(
        dataset.component.values.tolist()
    )

    missing_components = (
        required_components
        - available_components
    )

    if missing_components:
        raise ValueError(
            f"Missing components: {sorted(missing_components)}"
        )

    averaged = {}

    for variable, values in dataset.data_vars.items():
        score = values.sel(
            component=score_component,
            drop=True,
        )

        score_variance = values.sel(
            component=variance_component,
            drop=True,
        )

        count = values.sel(
            component="COUNT",
            drop=True,
        )

        spatial_weight = spatial_data[
            "latitude_weights"
        ].broadcast_like(count)

        valid = (
            (count > 0)
            & np.isfinite(score)
            & np.isfinite(score_variance)
        )

        if domain == "land":
            valid = (
                valid
                & spatial_data["land_mask"]
            )

        count = count.where(
            valid,
            0,
        )

        spatial_weight = (
            spatial_weight
            .where(valid, 0)
            .fillna(0)
        )

        weights = (
            spatial_weight
            * count
        )

        averaged_components = (
            values
            .sel(
                component=decomposition_components
            )
            .weighted(weights)
            .mean(
                (
                    "latitude",
                    "longitude",
                ),
                skipna=True,
            )
        )

        total_count = count.sum(
            (
                "latitude",
                "longitude",
            )
        )

        weight_sum = weights.sum(
            (
                "latitude",
                "longitude",
            )
        )

        normalized_weight = xr.where(
            weight_sum > 0,
            (
                spatial_weight
                * total_count
                / weight_sum
            ),
            0,
        )

        weighted_second_moment = xr.where(
            total_count > 0,
            (
                count
                * normalized_weight**2
                * (
                    score_variance
                    + score**2
                )
            ).sum(
                (
                    "latitude",
                    "longitude",
                )
            )
            / total_count,
            np.nan,
        )

        spatial_score = averaged_components.sel(
            component=score_component,
            drop=True,
        )

        spatial_score_variance = (
            weighted_second_moment
            - spatial_score**2
        ).clip(
            min=0
        )

        averaged[variable] = xr.concat(
            [
                averaged_components,
                spatial_score_variance.expand_dims(
                    component=[
                        variance_component
                    ]
                ),
                total_count.expand_dims(
                    component=[
                        "COUNT"
                    ]
                ),
            ],
            dim="component",
        )

    return xr.Dataset(averaged)


def empty_spatial_results(
    events,
) -> dict:
    """Create the common domain-event-model result structure."""
    return {
        domain: {
            event: {}
            for event in events
        }
        for domain in DOMAINS
    }


# ---------------------------------------------------------------------
# MSE and owMSE results
# ---------------------------------------------------------------------

def mse_result_path(
    model_stem: str,
    event: str,
) -> Path:
    """Return one MSE decomposition result path."""
    suffix = EVENT_CONFIGS[
        event
    ]["suffix"]

    filename = (
        f"{model_stem}_{RESOLUTION}"
        f"_vs_ifs_analysis{suffix}.zarr"
    )

    return (
        MSE_RESULTS_DIR
        / f"{RESOLUTION}_t2017_y2022"
        / filename
    )


def load_mse_spatial_averages(
    spatial_data: dict,
) -> dict:
    """Load the MSE results and calculate both spatial domains."""
    results = empty_spatial_results(
        EVENT_CONFIGS
    )

    for event, event_config in EVENT_CONFIGS.items():
        for label, model_stem in MODEL_STEMS.items():
            path = mse_result_path(
                model_stem,
                event,
            )

            print(
                f"{RESOLUTION}: loading MSE "
                f"for {label}, {event}"
            )

            source = xr.open_zarr(
                path,
                decode_timedelta=True,
            )

            try:
                with ProgressBar():
                    dataset = source[
                        event_config["variables"]
                    ].load()
            finally:
                source.close()

            for domain in DOMAINS:
                results[domain][event][label] = (
                    spatial_average_decomposition_result(
                        dataset,
                        score_component="MSE",
                        spatial_data=spatial_data,
                        domain=domain,
                    )
                )

            del dataset

    return results


# ---------------------------------------------------------------------
# PCRPS and twPCRPS results
# ---------------------------------------------------------------------

def pcrps_result_path(
    model_stem: str,
    variable: str,
) -> Path:
    """Return one casewise PCRPS result path."""
    variable_suffix = (
        PCRPS_VARIABLE_SUFFIXES[
            variable
        ]
    )

    filename = (
        f"{model_stem}_{RESOLUTION}"
        f"_vs_ifs_analysis_"
        f"{variable_suffix}_temporal.zarr"
    )

    return (
        PCRPS_RESULTS_DIR
        / f"{RESOLUTION}_t2017_y2022"
        / filename
    )


def prepare_pcrps_events(
    t2m_dataset: xr.Dataset,
    ws10_dataset: xr.Dataset,
) -> dict[str, xr.Dataset]:
    """Construct the five casewise PCRPS quantities used in the figures."""
    t2m_dataset = t2m_dataset.drop_vars(
        "lead_time_secs",
        errors="ignore",
    )

    ws10_dataset = ws10_dataset.drop_vars(
        "lead_time_secs",
        errors="ignore",
    )

    t2m_pcrps = t2m_dataset[
        "2m_temperature_pcrps"
    ]

    t2m_tw_pcrps = t2m_dataset[
        "2m_temperature_tw_pcrps"
    ]

    ws10_pcrps = ws10_dataset[
        "10m_wind_speed_pcrps"
    ]

    ws10_tw_pcrps = ws10_dataset[
        "10m_wind_speed_tw_pcrps"
    ]

    return {
        "all": xr.Dataset({
            "2m_temperature": t2m_pcrps,
            "10m_wind_speed": ws10_pcrps,
        }),
        "monthly_max": xr.Dataset({
            "2m_temperature": (
                t2m_tw_pcrps.sel(
                    threshold="monthly_max",
                    drop=True,
                )
            ),
            "10m_wind_speed": (
                ws10_tw_pcrps.sel(
                    threshold="monthly_max",
                    drop=True,
                )
            ),
        }),
        "monthly_min": xr.Dataset({
            "2m_temperature": (
                t2m_tw_pcrps.sel(
                    threshold="monthly_min",
                    drop=True,
                )
            ),
        }),
    }


def summarize_casewise_scores(
    dataset: xr.Dataset,
    component: str,
    spatial_data: dict,
    domain: str,
) -> xr.Dataset:
    """Calculate a weighted mean, variance, and sample size."""
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain {domain!r}."
        )

    dataset = select_common_grid(
        dataset,
        spatial_data["common_grid"],
    ).astype(np.float64)

    reduction_dimensions = (
        "time",
        "latitude",
        "longitude",
    )

    summarized = {}

    for variable, values in dataset.data_vars.items():
        spatial_template = values.isel(
            time=0,
            prediction_timedelta=0,
            drop=True,
        )

        spatial_weights = spatial_data[
            "latitude_weights"
        ].broadcast_like(
            spatial_template
        )

        if domain == "land":
            spatial_weights = (
                spatial_weights.where(
                    spatial_data["land_mask"]
                )
            )

        spatial_weights = (
            spatial_weights
            / spatial_weights.mean(
                (
                    "latitude",
                    "longitude",
                ),
                skipna=True,
            )
        )

        weighted_values = (
            values
            * spatial_weights
        )

        mean = weighted_values.mean(
            reduction_dimensions,
            skipna=True,
        )

        variance = weighted_values.var(
            reduction_dimensions,
            skipna=True,
            ddof=0,
        )

        count = weighted_values.count(
            reduction_dimensions
        ).astype(np.float64)

        summarized[variable] = xr.concat(
            [
                mean.expand_dims(
                    component=[
                        component
                    ]
                ),
                variance.expand_dims(
                    component=[
                        f"{component}_VAR"
                    ]
                ),
                count.expand_dims(
                    component=[
                        "COUNT"
                    ]
                ),
            ],
            dim="component",
        )

    return xr.Dataset(summarized)


def load_pcrps_spatial_averages(
    spatial_data: dict,
) -> dict:
    """Load the PCRPS results and calculate both spatial domains."""
    results = empty_spatial_results(
        EVENT_CONFIGS
    )

    for label, model_stem in MODEL_STEMS.items():
        print(
            f"{RESOLUTION}: calculating PCRPS "
            f"summaries for {label}"
        )

        t2m_source = xr.open_zarr(
            pcrps_result_path(
                model_stem,
                "2m_temperature",
            ),
            decode_timedelta=True,
            chunks=PCRPS_READ_CHUNKS,
        )

        ws10_source = xr.open_zarr(
            pcrps_result_path(
                model_stem,
                "10m_wind_speed",
            ),
            decode_timedelta=True,
            chunks=PCRPS_READ_CHUNKS,
        )

        try:
            event_scores = prepare_pcrps_events(
                t2m_source,
                ws10_source,
            )

            pending = {}

            for event, scores in event_scores.items():
                for domain in DOMAINS:
                    pending[
                        (
                            domain,
                            event,
                        )
                    ] = summarize_casewise_scores(
                        scores,
                        "PCRPS",
                        spatial_data,
                        domain,
                    )

            with ProgressBar():
                computed = dask.compute(
                    *pending.values()
                )
        finally:
            t2m_source.close()
            ws10_source.close()

        for (
            domain,
            event,
        ), result in zip(
            pending,
            computed,
        ):
            results[domain][event][label] = (
                result
            )

    return results


# ---------------------------------------------------------------------
# twMSE results
# ---------------------------------------------------------------------

def taggart_result_path(
    model_stem: str,
    event: str,
) -> Path:
    """Return one twMSE decomposition result path."""
    if event not in TAGGART_EVENT_CONFIGS:
        raise ValueError(
            f"Unknown twMSE event {event!r}."
        )

    event_config = TAGGART_EVENT_CONFIGS[
        event
    ]

    filename = (
        f"{model_stem}_{RESOLUTION}"
        f"_vs_ifs_analysis"
        f"{event_config['suffix']}.zarr"
    )

    return (
        TAGGART_RESULTS_DIR
        / f"{RESOLUTION}_t2017_y2022"
        / filename
    )


def load_taggart_spatial_averages(
    spatial_data: dict,
) -> dict:
    """Load the twMSE results and calculate both spatial domains."""
    results = {
        domain: {
            event: {}
            for event in TAGGART_EVENT_CONFIGS
        }
        for domain in DOMAINS
    }

    for event, event_config in (
        TAGGART_EVENT_CONFIGS.items()
    ):
        score_component = event_config[
            "score_component"
        ]

        for label, model_stem in MODEL_STEMS.items():
            path = taggart_result_path(
                model_stem,
                event,
            )

            print(
                f"{RESOLUTION}: loading twMSE "
                f"results for {label}, {event}"
            )

            source = xr.open_zarr(
                path,
                decode_timedelta=True,
            )

            try:
                with ProgressBar():
                    dataset = source[
                        event_config["variables"]
                    ].load()
            finally:
                source.close()

            for domain in DOMAINS:
                results[domain][event][label] = (
                    spatial_average_decomposition_result(
                        dataset,
                        score_component=score_component,
                        spatial_data=spatial_data,
                        domain=domain,
                    )
                )

            del dataset

    return results


# ---------------------------------------------------------------------
# Shared score-panel plotting
# ---------------------------------------------------------------------

def plot_score_panel(
    axis,
    datasets,
    variable,
    component,
    unit_key,
    title=None,
    show_xlabel=False,
    quantile=None,
    confidence_alpha=0.35,
    zero_baseline=False,
    xticks=(1, 3, 5, 7, 10),
    title_fontsize=18,
    label_fontsize=15,
    tick_fontsize=13,
):
    """Plot one score panel using the spatial-average result format."""
    plotted = False

    for label, dataset in datasets.items():
        if variable not in dataset:
            continue

        if component not in dataset.component.values:
            raise ValueError(
                f"Component {component!r} "
                f"is missing from {label!r}."
            )

        values = dataset[
            variable
        ].sel(
            component=component
        )

        lead_time_days = (
            values.prediction_timedelta.values
            / np.timedelta64(1, "D")
        )

        color = COLORS[label]

        if quantile is not None:
            variance_component = (
                f"{component}_VAR"
            )

            if (
                variance_component
                not in dataset.component.values
            ):
                raise ValueError(
                    f"Component {variance_component!r} "
                    f"is missing from {label!r}."
                )

            if (
                "COUNT"
                not in dataset.component.values
            ):
                raise ValueError(
                    f"Component 'COUNT' "
                    f"is missing from {label!r}."
                )

            variance = dataset[
                variable
            ].sel(
                component=variance_component
            )

            count = dataset[
                variable
            ].sel(
                component="COUNT"
            )

            half_width = (
                quantile
                * np.sqrt(
                    variance
                    / count
                )
            )

            axis.fill_between(
                lead_time_days,
                (
                    values
                    - half_width
                ).values,
                (
                    values
                    + half_width
                ).values,
                color=color,
                alpha=confidence_alpha,
                linewidth=0,
                zorder=1,
            )

        axis.plot(
            lead_time_days,
            values.values,
            label=label,
            color=color,
            linewidth=1.7,
            zorder=2,
        )

        plotted = True

    if not plotted:
        raise ValueError(
            f"No data were found for {variable!r}."
        )

    if title is not None:
        axis.set_title(
            title,
            fontsize=title_fontsize,
            pad=7,
        )

    if show_xlabel:
        axis.set_xlabel(
            "Lead Time [d]",
            fontsize=label_fontsize,
        )

    axis.set_xticks(
        xticks
    )

    if zero_baseline:
        axis.set_ylim(
            0,
            None,
        )

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
    )

    axis.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(
            0,
            0,
        ),
        useMathText=True,
    )

    axis.yaxis.get_offset_text().set_fontsize(
        tick_fontsize
    )


def format_empty_panel(
    axis,
    title=None,
    show_xlabel=False,
    xticks=(1, 3, 5, 7, 10),
    title_fontsize=18,
    label_fontsize=15,
    tick_fontsize=14,
):
    """Format an empty placeholder panel."""
    if title is not None:
        axis.set_title(
            title,
            fontsize=title_fontsize,
            pad=7,
        )

    if show_xlabel:
        axis.set_xlabel(
            "Lead Time [d]",
            fontsize=label_fontsize,
        )

    axis.set_xlim(
        xticks[0],
        xticks[-1],
    )

    axis.set_xticks(
        xticks
    )

    axis.set_yticks([])

    axis.tick_params(
        axis="x",
        which="major",
        labelsize=tick_fontsize,
    )


# ---------------------------------------------------------------------
# Main score figure
# ---------------------------------------------------------------------

def plot_main_score_figure(
    mse_datasets,
    taggart_datasets,
    pcrps_datasets=None,
    confidence_bands=True,
    confidence_level=0.99,
    confidence_alpha=0.35,
    zero_baseline=False,
    xticks=(1, 3, 5, 7, 10),
    figsize=(15, 19),
    title_fontsize=20,
    label_fontsize=19,
    tick_fontsize=15,
    row_label_fontsize=18,
    legend_fontsize=22,
    save_path=None,
    save_dpi=200,
):
    """
    Plot all-data scores above record-event scores.

    Variables and record types are shown as rows, while evaluation
    measures are shown as columns. The legend occupies the upper-right
    block, and x-axis labels appear only in the bottom row.
    """
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must lie between zero and one."
        )

    quantile = (
        stats.norm.ppf((1 + confidence_level) / 2)
        if confidence_bands
        else None
    )

    figure = plt.figure(figsize=figsize)

    grid = figure.add_gridspec(
        7,
        3,
        height_ratios=(0.10, 1, 1, 0.10, 1, 1, 1),
        left=0.10,
        right=0.98,
        bottom=0.05,
        top=0.98,
        wspace=0.16,
        hspace=0.27,
    )

    panel_kwargs = {
        "quantile": quantile,
        "confidence_alpha": confidence_alpha,
        "zero_baseline": zero_baseline,
        "xticks": xticks,
        "title_fontsize": title_fontsize,
        "label_fontsize": label_fontsize,
        "tick_fontsize": tick_fontsize,
    }

    empty_kwargs = {
        "xticks": xticks,
        "title_fontsize": title_fontsize,
        "label_fontsize": label_fontsize,
        "tick_fontsize": tick_fontsize,
    }

    legend_axis = figure.add_subplot(grid[1:3, 2])
    legend_axis.axis("off")

    all_axes = np.empty((2, 2), dtype=object)

    for row in range(2):
        for column in range(2):
            all_axes[row, column] = figure.add_subplot(
                grid[row + 1, column]
            )

    all_specs = [
        ("2m_temperature", "2m temperature"),
        ("10m_wind_speed", "10m wind speed"),
    ]

    for row, (variable, row_label) in enumerate(all_specs):
        show_titles = row == 0
        show_xlabel = False

        plot_score_panel(
            all_axes[row, 0],
            mse_datasets["all"],
            variable,
            "MSE",
            "squared",
            title="MSE" if show_titles else None,
            show_xlabel=show_xlabel,
            **panel_kwargs,
        )

        if pcrps_datasets is None:
            format_empty_panel(
                all_axes[row, 1],
                title="PCRPS" if show_titles else None,
                show_xlabel=show_xlabel,
                **empty_kwargs,
            )
        else:
            plot_score_panel(
                all_axes[row, 1],
                pcrps_datasets["all"],
                variable,
                "PCRPS",
                "linear",
                title="PCRPS" if show_titles else None,
                show_xlabel=show_xlabel,
                **panel_kwargs,
            )

        all_axes[row, 0].set_ylabel(
            row_label,
            fontsize=row_label_fontsize,
            fontweight="bold",
        )

    record_axes = np.empty((3, 3), dtype=object)

    for row in range(3):
        for column in range(3):
            record_axes[row, column] = figure.add_subplot(
                grid[row + 4, column]
            )

    record_specs = [
        ("monthly_max", "2m_temperature", "Heat records", "S2"),
        ("monthly_min", "2m_temperature", "Cold records", "S1"),
        ("monthly_max", "10m_wind_speed", "Wind records", "S2"),
    ]

    for row, (
        event,
        variable,
        row_label,
        taggart_component,
    ) in enumerate(record_specs):
        show_titles = row == 0
        show_xlabel = row == len(record_specs) - 1

        plot_score_panel(
            record_axes[row, 0],
            mse_datasets[event],
            variable,
            "MSE",
            "squared",
            title="owMSE" if show_titles else None,
            show_xlabel=show_xlabel,
            **panel_kwargs,
        )

        if pcrps_datasets is None:
            format_empty_panel(
                record_axes[row, 1],
                title="twPCRPS" if show_titles else None,
                show_xlabel=show_xlabel,
                **empty_kwargs,
            )
        else:
            plot_score_panel(
                record_axes[row, 1],
                pcrps_datasets[event],
                variable,
                "PCRPS",
                "linear",
                title="twPCRPS" if show_titles else None,
                show_xlabel=show_xlabel,
                **panel_kwargs,
            )

        plot_score_panel(
            record_axes[row, 2],
            taggart_datasets[event],
            variable,
            taggart_component,
            "squared",
            title="twMSE" if show_titles else None,
            show_xlabel=show_xlabel,
            **panel_kwargs,
        )

        record_axes[row, 0].set_ylabel(
            row_label,
            fontsize=row_label_fontsize,
            fontweight="bold",
        )

    for axis in all_axes[0, :]:
        axis.title.set_fontweight("normal")

    for axis in record_axes[0, :]:
        axis.title.set_fontweight("normal")

    figure.align_ylabels(all_axes[:, 0].tolist())
    figure.align_ylabels(record_axes[:, 0].tolist())

    for heading_row, heading_text in (
        (0, "(a) All data"),
        (3, "(b) Records"),
    ):
        heading_axis = figure.add_subplot(grid[heading_row, :])
        heading_axis.axis("off")
        heading_axis.text(
            0,
            0.5,
            heading_text,
            ha="left",
            va="center",
            fontsize=title_fontsize + 2,
            fontweight="bold",
        )

    handles, labels = all_axes[0, 0].get_legend_handles_labels()

    legend = legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        fontsize=legend_fontsize,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.6,
    )

    for handle in legend.legend_handles:
        handle.set_marker("")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            save_path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        figure.savefig(
            save_path.with_suffix(".png"),
            dpi=save_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    plt.show()


# ---------------------------------------------------------------------
# MSE and twMSE decomposition figures
# ---------------------------------------------------------------------

def plot_record_decomposition_figure(
    datasets: dict[str, dict[str, xr.Dataset]],
    component_labels: dict[str, str],
    components=("MCB", "DSC"),
    zero_baseline_components=(),
    xticks=(1, 3, 5, 7, 10),
    title_fontsize=20,
    label_fontsize=17,
    tick_fontsize=14,
    legend_fontsize=18,
    figsize=(15, 8.5),
    save_path=None,
    save_dpi=200,
):
    """Plot MCB and DSC for record events with a bottom legend."""
    if len(components) != 2:
        raise ValueError("Exactly two components are required.")

    missing_labels = set(components) - set(component_labels)

    if missing_labels:
        raise ValueError(
            f"Missing display labels for {sorted(missing_labels)}."
        )

    figure, axes = plt.subplots(
        len(components),
        len(RECORD_DECOMPOSITION_PANELS),
        figsize=figsize,
        sharex="col",
    )

    for row, component in enumerate(components):
        for column, (
            event,
            variable,
            title,
        ) in enumerate(RECORD_DECOMPOSITION_PANELS):
            axis = axes[row, column]

            plot_score_panel(
                axis=axis,
                datasets=datasets[event],
                variable=variable,
                component=component,
                unit_key="squared",
                title=title if row == 0 else None,
                show_xlabel=row == len(components) - 1,
                quantile=None,
                zero_baseline=component in zero_baseline_components,
                xticks=xticks,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
            )

            if column == 0:
                axis.set_ylabel(
                    component_labels[component],
                    fontsize=label_fontsize,
                )
            else:
                axis.set_ylabel("")

            if row < len(components) - 1:
                axis.tick_params(
                    axis="x",
                    labelbottom=False,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels),
        fontsize=legend_fontsize,
        frameon=True,
    )

    figure.align_ylabels(axes[:, 0])

    figure.tight_layout(
        rect=(0, 0.08, 1, 1),
        h_pad=1.2,
        w_pad=1.0,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            save_path.with_suffix(".pdf"),
            bbox_inches="tight",
        )
        figure.savefig(
            save_path.with_suffix(".png"),
            dpi=save_dpi,
            bbox_inches="tight",
        )

    plt.show()


# ---------------------------------------------------------------------
# Relative miscalibration figure
# ---------------------------------------------------------------------

def plot_record_miscalibration_comparison(
    mse_datasets: dict[str, dict[str, xr.Dataset]],
    taggart_datasets: dict[str, dict[str, xr.Dataset]],
    taggart_score_components=None,
    share_ylim=(0, 1),
    xticks=(1, 3, 5, 7, 10),
    title_fontsize=20,
    label_fontsize=17,
    tick_fontsize=14,
    legend_fontsize=18,
    fraction_decimals=2,
    figsize=(15, 8.5),
    save_path=None,
    save_dpi=200,
):
    """Compare relative miscalibration under owMSE and twMSE."""
    if taggart_score_components is None:
        taggart_score_components = {
            "monthly_max": "S2",
            "monthly_min": "S1",
        }

    row_specs = (
        (
            mse_datasets,
            {
                "monthly_max": "MSE",
                "monthly_min": "MSE",
            },
            "MCB / owMSE",
        ),
        (
            taggart_datasets,
            taggart_score_components,
            "MCB / twMSE",
        ),
    )

    figure, axes = plt.subplots(
        2,
        len(RECORD_DECOMPOSITION_PANELS),
        figsize=figsize,
        sharex="col",
        sharey=True,
    )

    for row, (
        protocol_datasets,
        score_components,
        row_label,
    ) in enumerate(
        row_specs
    ):
        for column, (
            event,
            variable,
            title,
        ) in enumerate(
            RECORD_DECOMPOSITION_PANELS
        ):
            axis = axes[
                row,
                column,
            ]

            score_component = (
                score_components[event]
            )

            for label, dataset in (
                protocol_datasets[
                    event
                ].items()
            ):
                if variable not in dataset:
                    continue

                values = dataset[
                    variable
                ]

                available = set(
                    values.component.values.tolist()
                )

                required = {
                    "MCB",
                    score_component,
                }

                missing = (
                    required
                    - available
                )

                if missing:
                    raise ValueError(
                        f"{label!r} is missing "
                        f"{sorted(missing)} for "
                        f"{event!r}, {variable!r}."
                    )

                score = values.sel(
                    component=score_component
                )

                mcb = values.sel(
                    component="MCB"
                )

                fraction = (
                    mcb
                    / score
                )

                fraction_values = (
                    fraction.values
                )

                if (
                    not np.all(
                        np.isfinite(
                            fraction_values
                        )
                    )
                    or np.any(
                        fraction_values < 0
                    )
                    or np.any(
                        fraction_values > 1
                    )
                ):
                    raise ValueError(
                        "Invalid MCB / score values "
                        f"for {label!r}, {event!r}, "
                        f"{variable!r}."
                    )

                lead_time_days = (
                    fraction
                    .prediction_timedelta
                    .values
                    / np.timedelta64(
                        1,
                        "D",
                    )
                )

                axis.plot(
                    lead_time_days,
                    fraction_values,
                    label=label,
                    color=COLORS[label],
                    linewidth=1.7,
                )

            if row == 0:
                axis.set_title(
                    title,
                    fontsize=title_fontsize,
                    pad=7,
                )

                axis.tick_params(
                    axis="x",
                    labelbottom=False,
                )
            else:
                axis.set_xlabel(
                    "Lead Time [d]",
                    fontsize=label_fontsize,
                )

            if column == 0:
                axis.set_ylabel(
                    row_label,
                    fontsize=label_fontsize,
                )

            axis.set_xticks(
                xticks
            )

            axis.tick_params(
                axis="both",
                which="major",
                labelsize=tick_fontsize,
            )

            axis.yaxis.set_major_formatter(
                FormatStrFormatter(
                    f"%.{fraction_decimals}f"
                )
            )

            if share_ylim is not None:
                axis.set_ylim(
                    *share_ylim
                )

    handles, labels = (
        axes[
            0,
            0,
        ].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.01,
        ),
        ncol=len(labels),
        fontsize=legend_fontsize,
        frameon=True,
    )

    figure.align_ylabels(
        axes[
            :,
            0,
        ]
    )

    figure.tight_layout(
        rect=(
            0,
            0.08,
            1,
            1,
        ),
        h_pad=1.2,
        w_pad=1.0,
    )

    if save_path is not None:
        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path.with_suffix(
                ".pdf"
            ),
            bbox_inches="tight",
        )

        figure.savefig(
            save_path.with_suffix(
                ".png"
            ),
            dpi=save_dpi,
            bbox_inches="tight",
        )

    plt.show()


# ---------------------------------------------------------------------
# twMSE contribution figure
# ---------------------------------------------------------------------

def plot_weighted_mse_contributions(
    table: pd.DataFrame,
    title_fontsize=20,
    label_fontsize=17,
    tick_fontsize=15,
    legend_fontsize=20,
    figsize=(15.5, 10),
    save_path=None,
):
    """Plot twMSE contributions over land by lead time."""
    figure, axes = plt.subplots(
        nrows=len(RECORD_ORDER),
        ncols=len(MODEL_ORDER),
        figsize=figsize,
        sharex=True,
        sharey="row",
    )

    x_positions = np.asarray(
        LEAD_TIME_DAYS,
        dtype=float,
    )

    for row, record_type in enumerate(
        RECORD_ORDER
    ):
        record_data = table.loc[
            table["Record type"]
            == record_type
        ]

        row_maximum = record_data[
            TOTAL_SCORE_COLUMN
        ].max()

        for column, model in enumerate(
            MODEL_ORDER
        ):
            axis = axes[
                row,
                column,
            ]

            model_data = (
                table.loc[
                    (
                        table["Record type"]
                        == record_type
                    )
                    & (
                        table["Model"]
                        == model
                    )
                ]
                .set_index(
                    "Lead time [d]"
                )
                .reindex(
                    LEAD_TIME_DAYS
                )
            )

            bottom = np.zeros(
                len(
                    LEAD_TIME_DAYS
                )
            )

            for contribution, style in (
                CONTRIBUTION_STYLES.items()
            ):
                values = model_data[
                    contribution
                ].to_numpy(
                    dtype=float
                )

                axis.bar(
                    x_positions,
                    values,
                    width=1.25,
                    bottom=bottom,
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=2,
                )

                bottom += values

            total_score = model_data[
                TOTAL_SCORE_COLUMN
            ].to_numpy(
                dtype=float
            )

            axis.plot(
                x_positions,
                total_score,
                color="black",
                marker="o",
                markersize=4,
                linewidth=1.5,
                zorder=3,
            )

            axis.set_xticks(
                LEAD_TIME_DAYS
            )

            axis.set_xlim(
                0,
                11,
            )

            axis.tick_params(
                axis="both",
                which="major",
                labelsize=tick_fontsize,
            )

            axis.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(
                    0,
                    0,
                ),
                useMathText=True,
            )

            axis.yaxis.get_offset_text().set_fontsize(
                tick_fontsize
            )

            if row == 0:
                axis.set_title(
                    model,
                    fontsize=title_fontsize,
                )

            if column == 0:
                axis.set_ylabel(
                    record_type,
                    fontsize=label_fontsize,
                )

            if row == len(RECORD_ORDER) - 1:
                axis.set_xlabel(
                    "Lead Time [d]",
                    fontsize=label_fontsize,
                )

        axes[
            row,
            0,
        ].set_ylim(
            0,
            1.08 * row_maximum,
        )

    legend_handles = [
        Patch(
            facecolor=style["color"],
            label=style["label"],
        )
        for style in CONTRIBUTION_STYLES.values()
    ]

    legend_handles.append(
        Line2D(
            [
                0
            ],
            [
                0
            ],
            color="black",
            marker="o",
            linewidth=1.5,
            markersize=4,
            label="twMSE",
        )
    )

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.995,
        ),
        ncol=4,
        fontsize=legend_fontsize,
        frameon=False,
    )

    figure.align_ylabels(
        axes[
            :,
            0,
        ]
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    if save_path is not None:
        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path.with_suffix(
                ".pdf"
            ),
            bbox_inches="tight",
        )

        figure.savefig(
            save_path.with_suffix(
                ".png"
            ),
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# ---------------------------------------------------------------------
# Create the figures
# ---------------------------------------------------------------------

def main() -> None:
    spatial_data = load_spatial_data()

    mse_results = load_mse_spatial_averages(
        spatial_data
    )

    pcrps_results = load_pcrps_spatial_averages(
        spatial_data
    )

    taggart_results = load_taggart_spatial_averages(
        spatial_data
    )

    plot_main_score_figure(
        mse_datasets=mse_results["global"],
        taggart_datasets=taggart_results["global"],
        pcrps_datasets=pcrps_results["global"],
        confidence_bands=True,
        zero_baseline=False,
        save_path=PLOTS_DIR / "main_scores_0p25_global",
    )

    plot_main_score_figure(
        mse_datasets=mse_results["land"],
        taggart_datasets=taggart_results["land"],
        pcrps_datasets=pcrps_results["land"],
        confidence_bands=True,
        zero_baseline=False,
        save_path=PLOTS_DIR / "main_scores_0p25_land",
    )

    plot_record_decomposition_figure(
        datasets=mse_results["land"],
        component_labels={
            "MCB": "MCB",
            "DSC": "DSC",
        },
        save_path=PLOTS_DIR / "owmse_decomposition_0p25_land",
    )

    plot_record_miscalibration_comparison(
        mse_datasets=mse_results["land"],
        taggart_datasets=taggart_results["land"],
        save_path=PLOTS_DIR / "relative_miscalibration_0p25_land",
    )

    land_contributions = pd.read_csv(
        TAGGART_TABLE_DIR / "taggart_score_contributions_0p25_land_by_lead_time_t2017_y2022.csv"
    )

    plot_weighted_mse_contributions(
        table=land_contributions,
        save_path=PLOTS_DIR / "twmse_contributions_0p25_land",
    )

    plot_record_decomposition_figure(
        datasets=taggart_results["global"],
        component_labels={
            "MCB": "MCB",
            "DSC": "DSC",
        },
        save_path=PLOTS_DIR / "twmse_decomposition_0p25_global",
    )

    plot_record_decomposition_figure(
        datasets=taggart_results["land"],
        component_labels={
            "MCB": "MCB",
            "DSC": "DSC",
        },
        save_path=PLOTS_DIR / "twmse_decomposition_0p25_land",
    )


if __name__ == "__main__":
    main()