from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path("/work/tsuda/GAEZ")
NOTEBOOK_PATH = (
    ROOT / "cropland_groundwater_13feature_regional_change_target2020.ipynb"
)


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "research",
    "language": "python",
    "name": "research",
}
notebook["metadata"]["language_info"] = {
    "name": "python",
    "version": "3.12",
}

notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        """# 地下水2変数追加前後の地域別Spatial OOF比較

旧11変数モデルと、地下水面深度・地下水涵養量を加えた新13変数モデルを、
同一のSpatial OOFセルで比較します。

対象地域:

- ナイル川河口部
- 米国西経100度周辺
- 黄河周辺

残差は `predicted − observed`、絶対誤差改善は
`|旧残差| − |新残差|` です。
"""
    ),
    code(
        """
from collections import OrderedDict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path("/work/tsuda/GAEZ")
OLD_OUTPUT_DIR = (
    ROOT
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_target2020"
)
NEW_OUTPUT_DIR = (
    ROOT
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_groundwater_target2020"
)
OUTPUT_DIR = NEW_OUTPUT_DIR / "regional_change_vs_11feature"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLD_PREDICTION_PATH = OLD_OUTPUT_DIR / "soil_comparison_oof_predictions.csv.gz"
NEW_PREDICTION_PATH = NEW_OUTPUT_DIR / "soil_comparison_oof_predictions.csv.gz"
NEW_INTEGRATED_SHAP_PATH = (
    NEW_OUTPUT_DIR / "integrated_final_all_cells_bg32_perm32.csv.gz"
)

for path in [
    OLD_PREDICTION_PATH,
    NEW_PREDICTION_PATH,
    NEW_INTEGRATED_SHAP_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)

OLD_MODEL_COLUMN = "pred_soil_group_wetland_excluded"
NEW_MODEL_COLUMN = "pred_soil_group_wetland_excluded_groundwater"
WTD_SHAP_COLUMN = "shap_feature__log_fan_water_table_depth_m"
RECHARGE_SHAP_COLUMN = "shap_feature__log_watergap_total_recharge_mm_yr"

REGIONS = OrderedDict([
    (
        "nile_delta",
        {
            "label": "Nile Delta / Lower Nile",
            "bounds": (28.0, 34.0, 28.5, 32.5),
        },
    ),
    (
        "usa_around_100w",
        {
            "label": "USA around 100°W",
            "bounds": (-115.0, -85.0, 25.0, 50.0),
            "reference_meridian": -100.0,
        },
    ),
    (
        "yellow_river_region",
        {
            "label": "Yellow River region",
            "bounds": (95.0, 125.0, 25.0, 45.0),
        },
    ),
])

TARGET_RESOLUTION = 1.0 / 12.0

print("Old predictions:", OLD_PREDICTION_PATH)
print("New predictions:", NEW_PREDICTION_PATH)
print("New integrated Shapley:", NEW_INTEGRATED_SHAP_PATH)
print("Output:", OUTPUT_DIR)
"""
    ),
    code(
        """
old_predictions = pd.read_csv(
    OLD_PREDICTION_PATH,
    compression="gzip",
    usecols=[
        "row",
        "col",
        "lat",
        "lon",
        "spatial_block",
        "cropland_fraction",
        OLD_MODEL_COLUMN,
    ],
).rename(columns={OLD_MODEL_COLUMN: "prediction_old_11feature"})

new_predictions = pd.read_csv(
    NEW_PREDICTION_PATH,
    compression="gzip",
    usecols=["row", "col", NEW_MODEL_COLUMN],
).rename(columns={NEW_MODEL_COLUMN: "prediction_new_13feature"})

new_shap = pd.read_csv(
    NEW_INTEGRATED_SHAP_PATH,
    compression="gzip",
    usecols=[
        "row",
        "col",
        "weight",
        WTD_SHAP_COLUMN,
        RECHARGE_SHAP_COLUMN,
    ],
).rename(
    columns={
        WTD_SHAP_COLUMN: "shap_groundwater_table_depth",
        RECHARGE_SHAP_COLUMN: "shap_groundwater_recharge",
    }
)

comparison = (
    old_predictions
    .merge(
        new_predictions,
        on=["row", "col"],
        how="inner",
        validate="one_to_one",
    )
    .merge(
        new_shap,
        on=["row", "col"],
        how="inner",
        validate="one_to_one",
    )
)

if len(comparison) != 240_000:
    raise RuntimeError(
        f"Expected 240,000 matched cells; found {len(comparison):,}"
    )

comparison["residual_old_predicted_minus_observed"] = (
    comparison["prediction_old_11feature"]
    - comparison["cropland_fraction"]
)
comparison["residual_new_predicted_minus_observed"] = (
    comparison["prediction_new_13feature"]
    - comparison["cropland_fraction"]
)
comparison["absolute_error_improvement_old_minus_new"] = (
    comparison["residual_old_predicted_minus_observed"].abs()
    - comparison["residual_new_predicted_minus_observed"].abs()
)
comparison["prediction_change_new_minus_old"] = (
    comparison["prediction_new_13feature"]
    - comparison["prediction_old_11feature"]
)

print("Matched cells:", f"{len(comparison):,}")
display(comparison.head())
"""
    ),
    code(
        """
def region_mask(frame, bounds):
    west, east, south, north = bounds
    return (
        frame["lon"].between(west, east)
        & frame["lat"].between(south, north)
    )


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_rmse(observed, predicted, weights):
    return float(
        np.sqrt(
            weighted_mean(
                (np.asarray(predicted) - np.asarray(observed)) ** 2,
                weights,
            )
        )
    )


def weighted_mae(observed, predicted, weights):
    return weighted_mean(
        np.abs(np.asarray(predicted) - np.asarray(observed)),
        weights,
    )


def weighted_r2(observed, predicted, weights):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    weights = np.asarray(weights, dtype=float)
    observed_mean = weighted_mean(observed, weights)
    numerator = np.sum(weights * (observed - predicted) ** 2)
    denominator = np.sum(weights * (observed - observed_mean) ** 2)
    return float(1.0 - numerator / denominator)


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(quantile, cumulative, values))


def rounded_limit(value, minimum=0.01):
    value = max(float(value), minimum)
    return float(np.ceil(value * 100) / 100)


metric_rows = []
regional_frames = []

for region_name, specification in REGIONS.items():
    selected = region_mask(comparison, specification["bounds"])
    frame = comparison.loc[selected].copy()
    frame.insert(0, "region", region_name)
    regional_frames.append(frame)

    observed = frame["cropland_fraction"].to_numpy(dtype=float)
    old_prediction = frame["prediction_old_11feature"].to_numpy(dtype=float)
    new_prediction = frame["prediction_new_13feature"].to_numpy(dtype=float)

    for weighting in ["unweighted", "area_weighted"]:
        if weighting == "area_weighted":
            weights = frame["weight"].to_numpy(dtype=float)
        else:
            weights = np.ones(len(frame), dtype=float)

        old_rmse = weighted_rmse(observed, old_prediction, weights)
        new_rmse = weighted_rmse(observed, new_prediction, weights)
        old_mae = weighted_mae(observed, old_prediction, weights)
        new_mae = weighted_mae(observed, new_prediction, weights)
        old_r2 = weighted_r2(observed, old_prediction, weights)
        new_r2 = weighted_r2(observed, new_prediction, weights)
        improved = (
            np.abs(new_prediction - observed)
            < np.abs(old_prediction - observed)
        )

        metric_rows.append({
            "region": region_name,
            "region_label": specification["label"],
            "weighting": weighting,
            "n_cells": len(frame),
            "old_rmse": old_rmse,
            "new_rmse": new_rmse,
            "rmse_improvement_old_minus_new": old_rmse - new_rmse,
            "rmse_change_pct": 100 * (new_rmse - old_rmse) / old_rmse,
            "old_mae": old_mae,
            "new_mae": new_mae,
            "mae_improvement_old_minus_new": old_mae - new_mae,
            "old_r2": old_r2,
            "new_r2": new_r2,
            "r2_change_new_minus_old": new_r2 - old_r2,
            "improved_cell_share_pct": 100 * weighted_mean(improved, weights),
            "mean_prediction_change_new_minus_old": weighted_mean(
                new_prediction - old_prediction,
                weights,
            ),
            "mean_abs_wtd_shap": weighted_mean(
                np.abs(frame["shap_groundwater_table_depth"]),
                weights,
            ),
            "mean_abs_recharge_shap": weighted_mean(
                np.abs(frame["shap_groundwater_recharge"]),
                weights,
            ),
        })

regional_metrics = pd.DataFrame(metric_rows)
regional_cells = pd.concat(regional_frames, ignore_index=True)

regional_metrics.to_csv(
    OUTPUT_DIR / "regional_11feature_vs_13feature_metrics.csv",
    index=False,
)
regional_cells.to_csv(
    OUTPUT_DIR / "regional_cell_level_model_change.csv.gz",
    index=False,
    compression="gzip",
)

focus_mask = np.zeros(len(comparison), dtype=bool)
for specification in REGIONS.values():
    focus_mask |= region_mask(comparison, specification["bounds"]).to_numpy()

focus = comparison.loc[focus_mask]
focus_weights = focus["weight"].to_numpy(dtype=float)
residual_values = np.concatenate([
    np.abs(focus["residual_old_predicted_minus_observed"].to_numpy()),
    np.abs(focus["residual_new_predicted_minus_observed"].to_numpy()),
])
residual_weights = np.tile(focus_weights, 2)

SCALES = {
    "residual": rounded_limit(
        weighted_quantile(residual_values, residual_weights, 0.99),
        minimum=0.05,
    ),
    "error_improvement": rounded_limit(
        weighted_quantile(
            np.abs(focus["absolute_error_improvement_old_minus_new"]),
            focus_weights,
            0.99,
        )
    ),
    "prediction_change": rounded_limit(
        weighted_quantile(
            np.abs(focus["prediction_change_new_minus_old"]),
            focus_weights,
            0.99,
        )
    ),
    "wtd_shap": rounded_limit(
        weighted_quantile(
            np.abs(comparison["shap_groundwater_table_depth"]),
            comparison["weight"],
            0.99,
        )
    ),
    "recharge_shap": rounded_limit(
        weighted_quantile(
            np.abs(comparison["shap_groundwater_recharge"]),
            comparison["weight"],
            0.99,
        )
    ),
}

print("Common panel scales across the three regions:")
print(json.dumps(SCALES, indent=2))
display(regional_metrics.round(6))
"""
    ),
    code(
        """
def grid_from_frame(frame, values, bounds):
    west, east, south, north = bounds
    nx = int(round((east - west) / TARGET_RESOLUTION))
    ny = int(round((north - south) / TARGET_RESOLUTION))
    x_edges = np.linspace(west, east, nx + 1)
    y_edges = np.linspace(south, north, ny + 1)
    columns = np.rint(
        (
            frame["lon"].to_numpy(dtype=float)
            - (west + TARGET_RESOLUTION / 2)
        )
        / TARGET_RESOLUTION
    ).astype(int)
    rows = np.rint(
        (
            frame["lat"].to_numpy(dtype=float)
            - (south + TARGET_RESOLUTION / 2)
        )
        / TARGET_RESOLUTION
    ).astype(int)
    values = np.asarray(values, dtype=float)
    valid = (
        (columns >= 0)
        & (columns < nx)
        & (rows >= 0)
        & (rows < ny)
        & np.isfinite(values)
    )
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    grid[rows[valid], columns[valid]] = values[valid]
    return x_edges, y_edges, np.ma.masked_invalid(grid)


def decorate_map(ax, bounds, specification):
    west, east, south, north = bounds
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    ax.set_facecolor("#EAF2F6")
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="#FAF9F2",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.OCEAN.with_scale("50m"),
        facecolor="#EAF2F6",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        edgecolor="#567785",
        linewidth=0.55,
        alpha=0.85,
        zorder=3,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="#EAF2F6",
        edgecolor="#567785",
        linewidth=0.35,
        zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="#888888",
        linewidth=0.4,
        zorder=4,
    )
    if "reference_meridian" in specification:
        meridian = specification["reference_meridian"]
        ax.plot(
            [meridian, meridian],
            [south, north],
            transform=ccrs.PlateCarree(),
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            zorder=5,
        )
        ax.text(
            meridian + 0.35,
            north - 1.1,
            "100°W",
            transform=ccrs.PlateCarree(),
            fontsize=8,
            color="#222222",
            zorder=6,
        )
        try:
            ax.add_feature(
                cfeature.STATES.with_scale("50m"),
                edgecolor="#AAAAAA",
                linewidth=0.28,
                zorder=4,
            )
        except Exception:
            pass
    ax.coastlines(
        resolution="50m",
        color="#555555",
        linewidth=0.65,
        zorder=5,
    )
    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="#7F8C91",
        alpha=0.32,
        linestyle="--",
        zorder=6,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 8}
    gridlines.ylabel_style = {"size": 8}


def draw_grid_panel(
    fig,
    ax,
    frame,
    values,
    bounds,
    specification,
    title,
    cmap,
    limit,
    colorbar_label,
    add_colorbar=True,
):
    decorate_map(ax, bounds, specification)
    x_edges, y_edges, grid = grid_from_frame(frame, values, bounds)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=mpl.colors.Normalize(-limit, limit),
        shading="flat",
        antialiased=False,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    ax.set_title("")
    if add_colorbar:
        colorbar = fig.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            pad=0.07,
            fraction=0.05,
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label(colorbar_label, fontsize=8.5, labelpad=3)
    return mesh
"""
    ),
    code(
        """
detail_paths = []

for region_name, specification in REGIONS.items():
    bounds = specification["bounds"]
    selected = region_mask(comparison, bounds)
    frame = comparison.loc[selected].copy()
    area_metrics = regional_metrics.loc[
        regional_metrics["region"].eq(region_name)
        & regional_metrics["weighting"].eq("area_weighted")
    ].iloc[0]

    rmse_change_pct = float(area_metrics["rmse_change_pct"])
    rmse_word = "worse" if rmse_change_pct > 0 else "better"
    summary_line = (
        f"Area-weighted RMSE: {area_metrics['old_rmse']:.4f} → "
        f"{area_metrics['new_rmse']:.4f} "
        f"({abs(rmse_change_pct):.1f}% {rmse_word}); "
        f"improved-cell share: {area_metrics['improved_cell_share_pct']:.1f}%"
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(21, 12.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.065,
        top=0.855,
        wspace=0.16,
        hspace=0.38,
    )

    panels = [
        (
            "residual_old_predicted_minus_observed",
            "A. Old 11-variable residual",
            "RdBu_r",
            SCALES["residual"],
            "Prediction error (predicted − observed)",
        ),
        (
            "residual_new_predicted_minus_observed",
            "B. New 13-variable residual",
            "RdBu_r",
            SCALES["residual"],
            "Prediction error (predicted − observed)",
        ),
        (
            "absolute_error_improvement_old_minus_new",
            "C. Absolute-error change\\n(old − new)",
            "RdYlGn",
            SCALES["error_improvement"],
            "Green: improved   |   Red: worsened",
        ),
        (
            "prediction_change_new_minus_old",
            "D. Prediction change\\n(new − old)",
            "RdBu_r",
            SCALES["prediction_change"],
            "Change in predicted cropland fraction",
        ),
        (
            "shap_groundwater_table_depth",
            "E. New-model SHAP\\nGroundwater-table depth",
            "RdBu_r",
            SCALES["wtd_shap"],
            "Integrated OOF Shapley",
        ),
        (
            "shap_groundwater_recharge",
            "F. New-model SHAP\\nGroundwater recharge",
            "RdBu_r",
            SCALES["recharge_shap"],
            "Integrated OOF Shapley",
        ),
    ]
    detail_captions = []

    for ax, (
        column,
        title,
        panel_cmap,
        limit,
        colorbar_label,
    ) in zip(axes.reshape(-1), panels, strict=True):
        draw_grid_panel(
            fig=fig,
            ax=ax,
            frame=frame,
            values=frame[column].to_numpy(dtype=float),
            bounds=bounds,
            specification=specification,
            title=title,
            cmap=panel_cmap,
            limit=limit,
            colorbar_label=colorbar_label,
        )
        detail_captions.append((ax, title))

    for caption_ax, caption_text in detail_captions:
        caption_box = caption_ax.get_position()
        fig.text(
            (caption_box.x0 + caption_box.x1) / 2,
            caption_box.y1 + 0.008,
            caption_text,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.5},
        )

    fig.suptitle(
        chr(10).join([
            f"{specification['label']} — effect of adding two groundwater variables",
            summary_line,
            "Residual blue = underprediction; residual red = overprediction",
        ]),
        fontsize=17,
        y=0.975,
    )

    output_path = OUTPUT_DIR / f"{region_name}_old11_vs_new13_detailed.png"
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    detail_paths.append(output_path)
    print("Saved:", output_path)

print("Detailed figures:", len(detail_paths))
"""
    ),
    code(
        """
fig, axes = plt.subplots(
    len(REGIONS),
    3,
    figsize=(20, 17),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=False,
)
fig.subplots_adjust(
    left=0.035,
    right=0.985,
    bottom=0.085,
    top=0.89,
    wspace=0.08,
    hspace=0.34,
)

summary_mesh = {}
summary_captions = []
for row_index, (region_name, specification) in enumerate(REGIONS.items()):
    bounds = specification["bounds"]
    selected = region_mask(comparison, bounds)
    frame = comparison.loc[selected].copy()
    area_metrics = regional_metrics.loc[
        regional_metrics["region"].eq(region_name)
        & regional_metrics["weighting"].eq("area_weighted")
    ].iloc[0]
    rmse_change_pct = float(area_metrics["rmse_change_pct"])
    direction = "worse" if rmse_change_pct > 0 else "better"
    region_title = chr(10).join([
        specification["label"],
        f"RMSE {area_metrics['old_rmse']:.4f} → {area_metrics['new_rmse']:.4f} "
        f"({abs(rmse_change_pct):.1f}% {direction})",
    ])

    columns = [
        (
            "residual_old_predicted_minus_observed",
            "Old 11-variable residual",
            "RdBu_r",
            SCALES["residual"],
        ),
        (
            "residual_new_predicted_minus_observed",
            "New 13-variable residual",
            "RdBu_r",
            SCALES["residual"],
        ),
        (
            "absolute_error_improvement_old_minus_new",
            "Absolute-error improvement",
            "RdYlGn",
            SCALES["error_improvement"],
        ),
    ]

    for column_index, (column, title, panel_cmap, limit) in enumerate(columns):
        panel_title = title
        if column_index == 0:
            panel_title = chr(10).join([region_title, title])
        mesh = draw_grid_panel(
            fig=fig,
            ax=axes[row_index, column_index],
            frame=frame,
            values=frame[column].to_numpy(dtype=float),
            bounds=bounds,
            specification=specification,
            title=panel_title,
            cmap=panel_cmap,
            limit=limit,
            colorbar_label="",
            add_colorbar=False,
        )
        summary_mesh[column_index] = mesh
        summary_captions.append((axes[row_index, column_index], panel_title))

for column_index, label in enumerate([
    "Residual (predicted − observed)",
    "Residual (predicted − observed)",
    "Absolute-error improvement: green better; red worse",
]):
    colorbar = fig.colorbar(
        summary_mesh[column_index],
        ax=axes[:, column_index].tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.025,
        shrink=0.86,
    )
    colorbar.set_label(label, fontsize=9)

for caption_ax, caption_text in summary_captions:
    caption_box = caption_ax.get_position()
    fig.text(
        (caption_box.x0 + caption_box.x1) / 2,
        caption_box.y1 + 0.006,
        caption_text,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="semibold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.2},
    )

fig.suptitle(
    chr(10).join([
        "Regional Spatial OOF change after adding groundwater-table depth and recharge",
        "Old 11-variable model vs new 13-variable model; common scales across all three regions",
    ]),
    fontsize=18,
    y=0.975,
)

summary_path = OUTPUT_DIR / "all_three_regions_old11_vs_new13_summary.png"
fig.savefig(summary_path, dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved:", summary_path)
"""
    ),
    code(
        """
manifest = {
    "old_model": "soil_group_wetland_excluded (11 variables)",
    "new_model": "soil_group_wetland_excluded_groundwater (13 variables)",
    "added_variables": [
        "log_fan_water_table_depth_m",
        "log_watergap_total_recharge_mm_yr",
    ],
    "regions": REGIONS,
    "shared_panel_scales": SCALES,
    "detailed_figures": [str(path) for path in detail_paths],
    "summary_figure": str(summary_path),
    "residual_definition": "predicted minus observed",
    "error_improvement_definition": "absolute old residual minus absolute new residual",
}
(OUTPUT_DIR / "regional_change_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

area_table = regional_metrics.loc[
    regional_metrics["weighting"].eq("area_weighted")
].copy()

readme_lines = [
    "# Regional change after adding groundwater variables",
    "",
    "The old 11-variable and new 13-variable models are compared on identical Spatial OOF cells.",
    "Residual = predicted minus observed.",
    "Absolute-error improvement = absolute old residual minus absolute new residual.",
    "Positive/green improvement values indicate lower error in the new model.",
    "",
    "## Area-weighted regional results",
    "",
]
for row in area_table.itertuples(index=False):
    direction = "worse" if row.rmse_change_pct > 0 else "better"
    readme_lines.append(
        f"- {row.region_label}: RMSE {row.old_rmse:.4f} to {row.new_rmse:.4f} "
        f"({abs(row.rmse_change_pct):.1f}% {direction}); "
        f"improved-cell share {row.improved_cell_share_pct:.1f}%"
    )
readme_lines.extend([
    "",
    "The groundwater SHAP panels describe predictive attribution inside the new model; they are not causal effects.",
])
(OUTPUT_DIR / "README.md").write_text(
    chr(10).join(readme_lines) + chr(10),
    encoding="utf-8",
)

result_root = NEW_OUTPUT_DIR
result_files = sorted(
    str(path.relative_to(result_root))
    for path in result_root.rglob("*")
    if path.is_file()
)
(result_root / "result_file_index.txt").write_text(
    chr(10).join(result_files) + chr(10),
    encoding="utf-8",
)

print("Regional comparison completed:", OUTPUT_DIR)
print(area_table[[
    "region_label",
    "n_cells",
    "old_rmse",
    "new_rmse",
    "rmse_change_pct",
    "improved_cell_share_pct",
]].to_string(index=False))
"""
    ),
]

nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
