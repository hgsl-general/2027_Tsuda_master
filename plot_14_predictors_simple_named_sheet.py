from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize


ROOT = Path("/work/tsuda/GAEZ")
RESULT_DIR = (
    ROOT
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_groundwater_rural_population_target2020"
)
INPUT_TABLE = (
    RESULT_DIR
    / "integrated_final_feature_dependence_bg32_perm32"
    / "integrated_feature_values_and_shapley_by_cell.csv.gz"
)
OUTPUT_PNG = RESULT_DIR / "all_14_model_input_variables_world_sheet.png"
OUTPUT_PDF = RESULT_DIR / "all_14_model_input_variables_world_sheet.pdf"


# Order follows the existing 13-variable baseline sheet; rural population is 14.
VARIABLES = [
    ("elevation_m", "Elevation", "m", "viridis", "continuous", None),
    ("slope", "Slope class", "GAEZ class (0–10)", "YlOrBr", "continuous", None),
    ("exclusion_class", "Land-exclusion class", "class code", "Set2", "categorical", None),
    ("log_city_time_20k_min", "Travel time to city ≥20k", "log1p(minutes)", "magma_r", "continuous", None),
    ("log_port_time_any_min", "Travel time to port", "log1p(minutes)", "magma_r", "continuous", None),
    ("rainfed_calorie_top5_raw", "Rainfed top-5 calorie potential", "log10(1 + raw input)", "YlGn", "continuous", "log10p1"),
    ("irrigation_calorie_gain_top5_raw", "Potential irrigation calorie gain", "log10(1 + raw input)", "YlGnBu", "continuous", "log10p1"),
    ("wx_50km_rainfed_calorie_top5_raw", "50-km rainfed calorie context", "log10(1 + raw input)", "YlGn", "continuous", "log10p1"),
    ("log_distance_river_gt10_2020", "Distance to reliable river (>10 m³/s)", "log1p(km)", "viridis_r", "continuous", None),
    ("log_glofas_p10_2020", "Local GloFAS p10 discharge", "log1p(m³/s)", "Blues", "continuous", None),
    ("soil_group_class", "WRB soil group", "class code (1–33)", "turbo", "categorical", None),
    ("log_fan_water_table_depth_m", "Groundwater-table depth [NEW]", "log1p(m)", "cividis", "continuous", None),
    ("log_watergap_total_recharge_mm_yr", "Groundwater recharge [NEW]", "log1p(mm/year)", "PuBuGn", "continuous", None),
    ("log_rural_population_density_2020", "Rural population density [NEW]", "log1p(persons/km²)", "Purples", "continuous", None),
]


def modal_value(series: pd.Series) -> float:
    modes = series.mode(dropna=True)
    return float(modes.iloc[0]) if not modes.empty else np.nan


def aggregate_to_one_degree(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = frame.copy()
    work["lon_1deg"] = np.floor(work["lon"]).clip(-180, 179).astype(int)
    work["lat_1deg"] = np.floor(work["lat"]).clip(-60, 84).astype(int)
    grouped = work.groupby(["lat_1deg", "lon_1deg"], observed=True)

    results: dict[str, pd.DataFrame] = {}
    for feature, _, _, _, kind, transform in VARIABLES:
        if kind == "categorical":
            table = grouped[feature].agg(modal_value).reset_index(name="value")
        else:
            table = grouped[feature].median().reset_index(name="value")
        if transform == "log10p1":
            table["value"] = np.log10(1.0 + np.clip(table["value"], 0.0, None))
        results[feature] = table
    return results


def categorical_scale(values: np.ndarray, cmap_name: str):
    categories = np.sort(np.unique(values[np.isfinite(values)]).astype(int))
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(0.02, 0.98, len(categories)))
    cmap = ListedColormap(colors)
    boundaries = np.concatenate(
        ([categories[0] - 0.5], (categories[:-1] + categories[1:]) / 2, [categories[-1] + 0.5])
    )
    norm = BoundaryNorm(boundaries, cmap.N)
    return categories, cmap, norm


def main() -> None:
    columns = ["lat", "lon"] + [item[0] for item in VARIABLES]
    frame = pd.read_csv(INPUT_TABLE, usecols=columns)
    if len(frame) != 240_000:
        raise RuntimeError(f"Expected 240,000 rows, found {len(frame):,}")
    aggregated = aggregate_to_one_degree(frame)

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(26, 17), facecolor="white")
    grid = fig.add_gridspec(
        4,
        4,
        left=0.020,
        right=0.985,
        bottom=0.070,
        top=0.910,
        wspace=0.030,
        hspace=0.390,
    )
    axes = [
        fig.add_subplot(grid[row, column], projection=projection)
        for row in range(4)
        for column in range(4)
    ]

    lon_edges = np.arange(-180, 181, 1, dtype=float)
    lat_edges = np.arange(-60, 86, 1, dtype=float)

    for number, (ax, variable) in enumerate(zip(axes, VARIABLES), start=1):
        feature, label, colorbar_label, cmap_name, kind, _ = variable
        table = aggregated[feature]
        values = table["value"].to_numpy(dtype=float)

        value_grid = np.full((145, 360), np.nan, dtype=np.float32)
        row_index = table["lat_1deg"].to_numpy(dtype=int) + 60
        column_index = table["lon_1deg"].to_numpy(dtype=int) + 180
        valid_index = (
            (row_index >= 0)
            & (row_index < value_grid.shape[0])
            & (column_index >= 0)
            & (column_index < value_grid.shape[1])
        )
        value_grid[row_index[valid_index], column_index[valid_index]] = values[valid_index]

        if kind == "categorical":
            categories, cmap, norm = categorical_scale(values, cmap_name)
        else:
            finite = values[np.isfinite(values)]
            low, high = np.quantile(finite, [0.01, 0.99]).astype(float)
            if high <= low:
                low, high = float(np.min(finite)), float(np.max(finite))
            if high <= low:
                high = low + 1.0
            cmap = plt.get_cmap(cmap_name)
            norm = Normalize(vmin=low, vmax=high, clip=True)

        ax.set_extent([-180, 180, -60, 85], crs=projection)
        ax.set_facecolor("white")
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            value_grid,
            cmap=cmap,
            norm=norm,
            shading="flat",
            transform=projection,
            rasterized=True,
            zorder=1,
        )
        ax.coastlines(resolution="110m", linewidth=0.35, color="#555555", zorder=3)
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            linewidth=0.18,
            edgecolor="#888888",
            zorder=3,
        )
        ax.gridlines(
            crs=projection,
            linewidth=0.22,
            color="#999999",
            alpha=0.25,
            linestyle="-",
            xlocs=[-120, -60, 0, 60, 120],
            ylocs=[-30, 0, 30, 60],
            zorder=0,
        )
        ax.set_title(
            f"{number}. {label}\n{feature}",
            fontsize=10.5,
            fontweight="bold" if "[NEW]" in label else "normal",
            pad=7,
            linespacing=1.15,
        )

        colorbar = fig.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            fraction=0.055,
            pad=0.055,
            aspect=35,
        )
        colorbar.ax.tick_params(labelsize=6.5, length=2, pad=1)
        colorbar.set_label(colorbar_label, fontsize=7.5, labelpad=2)
        colorbar.outline.set_linewidth(0.45)
        if kind == "categorical":
            if len(categories) <= 8:
                ticks = categories
            else:
                indices = np.linspace(0, len(categories) - 1, 8).round().astype(int)
                ticks = categories[indices]
            colorbar.set_ticks(ticks)

    for ax in axes[len(VARIABLES):]:
        ax.set_axis_off()

    fig.suptitle(
        "Updated 14-variable model predictors — global spatial patterns",
        fontsize=23,
        fontweight="bold",
        y=0.972,
    )
    fig.text(
        0.5,
        0.025,
        "Square cells are 1° aggregates of the 240,000-row Spatial OOF analysis sample "
        "(median for continuous/ordinal variables; mode for categorical variables). "
        "Continuous color limits use the 1st–99th percentiles.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    fig.savefig(OUTPUT_PNG, dpi=220, facecolor="white", bbox_inches=None)
    fig.savefig(OUTPUT_PDF, dpi=220, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"Rows: {len(frame):,}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
