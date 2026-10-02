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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm


ROOT = Path("/work/tsuda/GAEZ")
RESULT_DIR = (
    ROOT
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_groundwater_rural_population_target2020"
)
GROUP_SHAP_PATH = RESULT_DIR / "soil_model_oof_local_group_shap.csv.gz"
BACKGROUND_PATH = (
    RESULT_DIR
    / "integrated_final_feature_dependence_bg32_perm32"
    / "integrated_feature_values_and_shapley_by_cell.csv.gz"
)

FACTOR_NAMES = [
    "Land and soil capital",
    "Climate and water capital",
    "Market access and infrastructure",
    "Rural population",
]

STAGE_CONFIG = {
    "presence_classifier": {
        "title": "Stage 1: Cropland presence classifier",
        "explanation": "Contribution to whether cropland is present",
        "unit": "raw score / log-odds",
        "combined_output": "signed_factor_maps_stage1_presence.png",
        "rural_output": "signed_rural_population_shap_stage1_presence.png",
    },
    "conditional_fraction_regressor": {
        "title": "Stage 2: Conditional cropland-fraction regressor",
        "explanation": "Contribution to cropland fraction among cropland-positive cells",
        "unit": "conditional cropland-fraction units",
        "combined_output": "signed_factor_maps_stage2_conditional.png",
        "rural_output": "signed_rural_population_shap_stage2_conditional.png",
    },
}

COLOR_QUANTILE = 0.99
CMAP = plt.get_cmap("RdBu_r")
PROJECTION = ccrs.PlateCarree()


def color_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise RuntimeError("No finite SHAP values were found")
    limit = float(np.quantile(finite, COLOR_QUANTILE))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.max(finite))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def configure_map(ax) -> None:
    ax.set_extent([-180, 180, -60, 85], crs=PROJECTION)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#EAF2F5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F5F3EE", zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.45, color="#4A4A4A", zorder=4)
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linewidth=0.25,
        edgecolor="#777777",
        zorder=4,
    )
    gridliner = ax.gridlines(
        crs=PROJECTION,
        draw_labels=True,
        linewidth=0.3,
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
    gridliner.xlabel_style = {"size": 7, "color": "#555555"}
    gridliner.ylabel_style = {"size": 7, "color": "#555555"}


def plot_background(ax, background: pd.DataFrame) -> None:
    ax.scatter(
        background["lon"],
        background["lat"],
        s=0.42,
        marker="s",
        c="#D4D4D4",
        alpha=0.32,
        linewidths=0,
        transform=PROJECTION,
        rasterized=True,
        zorder=1,
    )


def plot_four_factors(
    stage_frame: pd.DataFrame,
    background: pd.DataFrame,
    config: dict[str, str],
) -> None:
    group_columns = {
        factor: f"group_shap__{factor}"
        for factor in FACTOR_NAMES
    }
    missing = [column for column in group_columns.values() if column not in stage_frame]
    if missing:
        raise KeyError(f"Missing group SHAP columns: {missing}")

    combined_values = np.concatenate(
        [stage_frame[column].to_numpy(dtype=float) for column in group_columns.values()]
    )
    limit = color_limit(combined_values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(20, 12.2),
        subplot_kw={"projection": PROJECTION},
    )
    for ax, factor in zip(axes.flat, FACTOR_NAMES):
        configure_map(ax)
        plot_background(ax, background)
        values = stage_frame[group_columns[factor]].to_numpy(dtype=float)
        ax.scatter(
            stage_frame["lon"],
            stage_frame["lat"],
            c=values,
            cmap=CMAP,
            norm=norm,
            s=4.1,
            marker="s",
            alpha=0.93,
            linewidths=0,
            transform=PROJECTION,
            rasterized=True,
            zorder=3,
        )
        ax.set_title(
            f"{factor}\n"
            f"positive: {100 * np.mean(values > 0):.1f}%  |  "
            f"negative: {100 * np.mean(values < 0):.1f}%  |  "
            f"mean |SHAP|: {np.mean(np.abs(values)):.4f}",
            fontsize=12,
            fontweight="bold" if factor == "Rural population" else "normal",
            pad=7,
        )

    scalar = ScalarMappable(norm=norm, cmap=CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.075,
        shrink=0.72,
        aspect=45,
    )
    colorbar.set_label(
        f"Signed grouped OOF SHAP ({config['unit']})\n"
        "Blue lowers the prediction; red raises it. "
        f"All four panels share one scale, clipped at the {COLOR_QUANTILE:.0%} percentile.",
        fontsize=11,
    )
    fig.suptitle(
        f"{config['title']} — signed SHAP by production-factor group\n"
        f"{config['explanation']} | Rural population is shown in the lower-right panel",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Colored squares: 50,000 local-SHAP cells. Pale gray squares: full 240,000-cell analysis sample.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.14, wspace=0.035, hspace=0.17)
    output = RESULT_DIR / config["combined_output"]
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_rural_only(
    stage_frame: pd.DataFrame,
    background: pd.DataFrame,
    config: dict[str, str],
) -> None:
    column = "group_shap__Rural population"
    values = stage_frame[column].to_numpy(dtype=float)
    limit = color_limit(values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(15.5, 8.3),
        subplot_kw={"projection": PROJECTION},
    )
    configure_map(ax)
    plot_background(ax, background)
    scatter = ax.scatter(
        stage_frame["lon"],
        stage_frame["lat"],
        c=values,
        cmap=CMAP,
        norm=norm,
        s=7.0,
        marker="s",
        alpha=0.95,
        linewidths=0,
        transform=PROJECTION,
        rasterized=True,
        zorder=3,
    )
    ax.set_title(
        "Rural population factor\n"
        f"positive: {100 * np.mean(values > 0):.1f}%  |  "
        f"negative: {100 * np.mean(values < 0):.1f}%  |  "
        f"mean |SHAP|: {np.mean(np.abs(values)):.4f}",
        fontsize=14,
        pad=8,
    )
    colorbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.085, shrink=0.76, aspect=45)
    colorbar.set_label(
        f"Rural-population signed OOF SHAP ({config['unit']})\n"
        f"Rural-only scale, clipped at the {COLOR_QUANTILE:.0%} percentile",
        fontsize=11,
    )
    fig.suptitle(
        f"{config['title']} — Rural population signed SHAP\n{config['explanation']}",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Blue: rural population lowers the stage prediction. Red: rural population raises it.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(left=0.035, right=0.98, top=0.86, bottom=0.16)
    output = RESULT_DIR / config["rural_output"]
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    group_frame = pd.read_csv(GROUP_SHAP_PATH)
    background = (
        pd.read_csv(BACKGROUND_PATH, usecols=["lat", "lon"])
        .dropna()
        .drop_duplicates()
    )
    expected_group_columns = [f"group_shap__{factor}" for factor in FACTOR_NAMES]
    missing = [column for column in expected_group_columns if column not in group_frame]
    if missing:
        raise KeyError(f"Rural-inclusive group SHAP data are incomplete: {missing}")

    for stage, config in STAGE_CONFIG.items():
        stage_frame = group_frame.loc[group_frame["stage"] == stage].copy()
        if len(stage_frame) != 50_000:
            raise RuntimeError(f"Expected 50,000 {stage} rows, found {len(stage_frame):,}")
        plot_four_factors(stage_frame, background, config)
        plot_rural_only(stage_frame, background, config)


if __name__ == "__main__":
    main()
