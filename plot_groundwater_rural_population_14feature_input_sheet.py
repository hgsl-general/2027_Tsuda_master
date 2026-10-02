from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, PowerNorm


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


FEATURE_SPECS = [
    {
        "feature": "elevation_m",
        "label": "Elevation",
        "group": "Land and soil capital",
        "unit": "m",
        "cmap": "terrain",
        "kind": "continuous",
    },
    {
        "feature": "slope",
        "label": "Slope",
        "group": "Land and soil capital",
        "unit": "model input value",
        "cmap": "magma",
        "kind": "continuous",
    },
    {
        "feature": "exclusion_class",
        "label": "Environmental exclusion class",
        "group": "Land and soil capital",
        "unit": "class code",
        "cmap": "Set1",
        "kind": "categorical",
    },
    {
        "feature": "soil_group_class",
        "label": "Soil group",
        "group": "Land and soil capital",
        "unit": "class code",
        "cmap": "turbo",
        "kind": "categorical",
    },
    {
        "feature": "rainfed_calorie_top5_raw",
        "label": "Local rainfed calorie potential",
        "group": "Climate and water capital",
        "unit": "raw input value",
        "cmap": "YlGn",
        "kind": "positive_skew",
    },
    {
        "feature": "irrigation_calorie_gain_top5_raw",
        "label": "Irrigation calorie gain",
        "group": "Climate and water capital",
        "unit": "raw input value",
        "cmap": "PuBuGn",
        "kind": "positive_skew",
    },
    {
        "feature": "wx_50km_rainfed_calorie_top5_raw",
        "label": "Rainfed potential within 50 km",
        "group": "Climate and water capital",
        "unit": "raw input value",
        "cmap": "YlGnBu",
        "kind": "positive_skew",
    },
    {
        "feature": "log_distance_river_gt10_2020",
        "label": "Distance to river, p10 flow >10 m³/s",
        "group": "Climate and water capital",
        "unit": "log1p(distance)",
        "cmap": "Blues",
        "kind": "continuous",
    },
    {
        "feature": "log_glofas_p10_2020",
        "label": "Local GloFAS p10 flow",
        "group": "Climate and water capital",
        "unit": "log1p(m³/s)",
        "cmap": "GnBu",
        "kind": "continuous",
    },
    {
        "feature": "log_fan_water_table_depth_m",
        "label": "Groundwater-table depth",
        "group": "Climate and water capital",
        "unit": "log1p(m)",
        "cmap": "cividis_r",
        "kind": "continuous",
    },
    {
        "feature": "log_watergap_total_recharge_mm_yr",
        "label": "Groundwater recharge",
        "group": "Climate and water capital",
        "unit": "log1p(mm/year)",
        "cmap": "BuGn",
        "kind": "continuous",
    },
    {
        "feature": "log_city_time_20k_min",
        "label": "Travel time to city (≥20k people)",
        "group": "Market access and infrastructure",
        "unit": "log1p(minutes)",
        "cmap": "Oranges",
        "kind": "continuous",
    },
    {
        "feature": "log_port_time_any_min",
        "label": "Travel time to port",
        "group": "Market access and infrastructure",
        "unit": "log1p(minutes)",
        "cmap": "Reds",
        "kind": "continuous",
    },
    {
        "feature": "log_rural_population_density_2020",
        "label": "Rural population density (2020)",
        "group": "Rural population",
        "unit": "log1p(persons/km²)",
        "cmap": "Purples",
        "kind": "continuous",
    },
]

GROUP_COLORS = {
    "Land and soil capital": "#8C6D31",
    "Climate and water capital": "#2B8CBE",
    "Market access and infrastructure": "#E6550D",
    "Rural population": "#756BB1",
}


def robust_limits(values: np.ndarray, kind: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    if kind == "positive_skew":
        low = max(0.0, float(np.nanmin(finite)))
        high = float(np.nanquantile(finite, 0.99))
    else:
        low, high = np.nanquantile(finite, [0.02, 0.98]).astype(float)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if high <= low:
        high = low + 1.0
    return low, high


def categorical_style(values: np.ndarray, cmap_name: str):
    categories = np.sort(np.unique(values[np.isfinite(values)]).astype(int))
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(0.03, 0.97, len(categories)))
    cmap = ListedColormap(colors)
    boundaries = np.concatenate(
        ([categories[0] - 0.5], (categories[:-1] + categories[1:]) / 2, [categories[-1] + 0.5])
    )
    norm = BoundaryNorm(boundaries, cmap.N)
    return categories, cmap, norm


def main() -> None:
    required = ["lat", "lon"] + [spec["feature"] for spec in FEATURE_SPECS]
    frame = pd.read_csv(INPUT_TABLE, usecols=required)
    if len(frame) != 240_000:
        raise RuntimeError(f"Expected 240,000 analysis cells, found {len(frame):,}")

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(29, 18.5),
        subplot_kw={"projection": projection},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "All 14 predictor variables used by the 2020 cropland Spatial OOF model",
        fontsize=23,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.958,
        "Values are shown on the exact model-input scale; each colored square is one sampled 5-arc-minute grid cell (n = 240,000)",
        ha="center",
        va="top",
        fontsize=12,
        color="#333333",
    )

    lon = frame["lon"].to_numpy(dtype=float)
    lat = frame["lat"].to_numpy(dtype=float)

    for panel_number, (ax, spec) in enumerate(zip(axes.flat, FEATURE_SPECS), start=1):
        values = frame[spec["feature"]].to_numpy(dtype=float)
        ax.set_extent([-180, 180, -60, 85], crs=projection)
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#EAF2F5", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F4F2EC", zorder=0)

        if spec["kind"] == "categorical":
            categories, cmap, norm = categorical_style(values, spec["cmap"])
            plotted = values
        else:
            low, high = robust_limits(values, spec["kind"])
            plotted = np.clip(values, low, high)
            cmap = plt.get_cmap(spec["cmap"])
            if spec["kind"] == "positive_skew":
                norm = PowerNorm(gamma=0.45, vmin=low, vmax=high)
            else:
                norm = Normalize(vmin=low, vmax=high)

        scatter = ax.scatter(
            lon,
            lat,
            c=plotted,
            s=1.15,
            marker="s",
            linewidths=0,
            cmap=cmap,
            norm=norm,
            transform=projection,
            rasterized=True,
            zorder=2,
        )
        ax.coastlines(resolution="110m", linewidth=0.35, color="#4A4A4A", zorder=3)
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            linewidth=0.22,
            edgecolor="#777777",
            zorder=3,
        )
        gridliner = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.25,
            color="#777777",
            alpha=0.35,
            linestyle=":",
            xlocs=[-120, -60, 0, 60, 120],
            ylocs=[-30, 0, 30, 60],
        )
        gridliner.top_labels = False
        gridliner.right_labels = False
        gridliner.xformatter = LONGITUDE_FORMATTER
        gridliner.yformatter = LATITUDE_FORMATTER
        gridliner.xlabel_style = {"size": 6, "color": "#555555"}
        gridliner.ylabel_style = {"size": 6, "color": "#555555"}

        ax.set_title(
            f"{chr(64 + panel_number)}. {spec['label']}\n{spec['feature']}",
            fontsize=9.5,
            fontweight="bold",
            pad=5,
            color="#222222",
        )
        ax.text(
            0.012,
            0.985,
            spec["group"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": GROUP_COLORS[spec["group"]],
                "edgecolor": "none",
                "alpha": 0.93,
            },
            zorder=5,
        )

        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            orientation="horizontal",
            fraction=0.05,
            pad=0.055,
            aspect=35,
        )
        colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
        colorbar.set_label(spec["unit"], fontsize=7, labelpad=2)
        colorbar.outline.set_linewidth(0.4)
        if spec["kind"] == "categorical":
            if len(categories) <= 8:
                ticks = categories
            else:
                tick_positions = np.linspace(0, len(categories) - 1, 7).round().astype(int)
                ticks = categories[tick_positions]
            colorbar.set_ticks(ticks)

    note_axes = [axes[3, 2], axes[3, 3]]
    for ax in note_axes:
        ax.set_axis_off()

    note_axes[0].text(
        0.04,
        0.93,
        "Factor-group structure",
        transform=note_axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    y = 0.76
    for group, features in [
        ("Land and soil capital", 4),
        ("Climate and water capital", 7),
        ("Market access and infrastructure", 2),
        ("Rural population", 1),
    ]:
        note_axes[0].text(
            0.07,
            y,
            f"{group}: {features} variable{'s' if features > 1 else ''}",
            transform=note_axes[0].transAxes,
            fontsize=11,
            color="white",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": GROUP_COLORS[group],
                "edgecolor": "none",
            },
        )
        y -= 0.17

    note_axes[1].text(
        0.04,
        0.93,
        "How to read this sheet",
        transform=note_axes[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    note_axes[1].text(
        0.04,
        0.78,
        "• Every map uses the same 240,000 Spatial OOF analysis cells.\n"
        "• Colors show predictor values, not SHAP values.\n"
        "• Each panel has its own scale because units differ.\n"
        "• Continuous scales are clipped at robust 2nd–98th percentiles.\n"
        "• Raw calorie variables use a power color normalization and 99th-percentile cap.\n"
        "• Blank land areas were not selected into the analysis sample.\n"
        "• Rural density = rural persons / grid-cell area (km²), then log1p.",
        transform=note_axes[1].transAxes,
        fontsize=10.5,
        linespacing=1.45,
        va="top",
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#F4F4F4",
            "edgecolor": "#BBBBBB",
        },
    )

    fig.text(
        0.5,
        0.012,
        "Source: completed 14-feature groundwater + rural-population model. Map values are predictors as supplied to the model, not causal effects.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    plt.subplots_adjust(left=0.025, right=0.992, top=0.925, bottom=0.045, wspace=0.08, hspace=0.30)
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Rows: {len(frame):,}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
