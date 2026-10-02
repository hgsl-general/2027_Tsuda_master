from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path("/work/tsuda/GAEZ")
NOTEBOOK_PATH = (
    ROOT
    / "cropland_groundwater_13feature_spatial_shap_maps_target2020.ipynb"
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
        """# 13変数の地域別・統合OOF Shapley地図

地下水2変数を加えた13変数モデルについて、最終耕作地割合
`f(X) = p(X) × q(X)` に対する統合OOF Shapleyを地図化します。

- 赤: 最終予測を押し上げる
- 青: 最終予測を押し下げる
- 各色付きピクセル: 1つの5分グリッド分析セル
- SHAPの再計算やモデルの再学習は行わず、保存済み240,000セルを使用
"""
    ),
    code(
        """
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path("/work/tsuda/GAEZ")
MODEL_OUTPUT_DIR = (
    ROOT
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_groundwater_target2020"
)
INPUT_PATH = (
    MODEL_OUTPUT_DIR
    / "integrated_final_all_cells_bg32_perm32.csv.gz"
)
OUTPUT_DIR = (
    MODEL_OUTPUT_DIR
    / "integrated_feature_spatial_maps_bg32_perm32"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

FEATURES = [
    "elevation_m",
    "slope",
    "exclusion_class",
    "log_city_time_20k_min",
    "log_port_time_any_min",
    "rainfed_calorie_top5_raw",
    "irrigation_calorie_gain_top5_raw",
    "wx_50km_rainfed_calorie_top5_raw",
    "log_distance_river_gt10_2020",
    "log_glofas_p10_2020",
    "soil_group_class",
    "log_fan_water_table_depth_m",
    "log_watergap_total_recharge_mm_yr",
]

FEATURE_LABELS = {
    "elevation_m": "Elevation",
    "slope": "Slope class",
    "exclusion_class": "Environmental exclusion class",
    "log_city_time_20k_min": "Travel time to city",
    "log_port_time_any_min": "Travel time to port",
    "rainfed_calorie_top5_raw": "Local rainfed calorie potential",
    "irrigation_calorie_gain_top5_raw": "Potential calorie gain from irrigation",
    "wx_50km_rainfed_calorie_top5_raw": "50-km rainfed calorie potential",
    "log_distance_river_gt10_2020": "Distance to river with p10 flow >10 m³/s",
    "log_glofas_p10_2020": "Local low-flow river discharge (GloFAS p10)",
    "soil_group_class": "Soil group",
    "log_fan_water_table_depth_m": "Groundwater-table depth (Fan)",
    "log_watergap_total_recharge_mm_yr": "Groundwater recharge (WaterGAP)",
}

FEATURE_GROUPS = {
    "elevation_m": "Land and soil capital",
    "slope": "Land and soil capital",
    "exclusion_class": "Land and soil capital",
    "soil_group_class": "Land and soil capital",
    "rainfed_calorie_top5_raw": "Climate and water capital",
    "irrigation_calorie_gain_top5_raw": "Climate and water capital",
    "wx_50km_rainfed_calorie_top5_raw": "Climate and water capital",
    "log_distance_river_gt10_2020": "Climate and water capital",
    "log_glofas_p10_2020": "Climate and water capital",
    "log_fan_water_table_depth_m": "Climate and water capital",
    "log_watergap_total_recharge_mm_yr": "Climate and water capital",
    "log_city_time_20k_min": "Market access and infrastructure",
    "log_port_time_any_min": "Market access and infrastructure",
}

SHAP_COLUMNS = {
    feature: f"shap_feature__{feature}"
    for feature in FEATURES
}

usecols = (
    ["row", "col", "lat", "lon", "weight"]
    + list(SHAP_COLUMNS.values())
)
shap_cells = pd.read_csv(
    INPUT_PATH,
    compression="gzip",
    usecols=usecols,
)

print("Input:", INPUT_PATH)
print("Output:", OUTPUT_DIR)
print("Cells:", f"{len(shap_cells):,}")
print("Features:", len(FEATURES))
"""
    ),
    code(
        """
N_ROWS_GLOBAL = 2160
N_COLS_GLOBAL = 4320
MAP_EXTENT = (-180, 180, -60, 85)
ROBUST_QUANTILE = 0.99

cmap = mpl.colormaps["RdBu_r"].copy()
cmap.set_bad((0, 0, 0, 0))


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(quantile, cumulative, values))


def feature_limit(feature):
    values = pd.to_numeric(
        shap_cells[SHAP_COLUMNS[feature]],
        errors="coerce",
    ).to_numpy(dtype=float)
    weights = pd.to_numeric(
        shap_cells["weight"],
        errors="coerce",
    ).to_numpy(dtype=float)
    limit = weighted_quantile(
        np.abs(values),
        weights,
        ROBUST_QUANTILE,
    )
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(values)))
    return limit


def make_raster(feature):
    raster = np.full(
        (N_ROWS_GLOBAL, N_COLS_GLOBAL),
        np.nan,
        dtype=np.float32,
    )
    rows = shap_cells["row"].to_numpy(dtype=np.int32)
    cols = shap_cells["col"].to_numpy(dtype=np.int32)
    values = pd.to_numeric(
        shap_cells[SHAP_COLUMNS[feature]],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    raster[rows, cols] = values
    return raster


def add_map_background(ax):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("#EAF2F6")
    ax.add_feature(
        cfeature.LAND,
        facecolor="#F2F0E8",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#EAF2F6",
        edgecolor="none",
        zorder=0,
    )


def add_boundaries(ax):
    ax.coastlines(
        resolution="110m",
        linewidth=0.45,
        color="#4A4A4A",
        zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.25,
        edgecolor="#777777",
        zorder=3,
    )
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="#777777",
        alpha=0.28,
        linestyle="--",
        zorder=4,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


feature_limits = {
    feature: feature_limit(feature)
    for feature in FEATURES
}

scale_rows = []
weights = shap_cells["weight"].to_numpy(dtype=float)
for feature in FEATURES:
    values = shap_cells[SHAP_COLUMNS[feature]].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    scale_rows.append({
        "feature": feature,
        "label": FEATURE_LABELS[feature],
        "group": FEATURE_GROUPS[feature],
        "n_cells": int(np.isfinite(values).sum()),
        "area_weighted_mean_abs_shapley": float(
            np.average(np.abs(values[valid]), weights=weights[valid])
        ),
        "area_weighted_mean_signed_shapley": float(
            np.average(values[valid], weights=weights[valid])
        ),
        "map_abs_limit_weighted_p99": feature_limits[feature],
    })

scale_summary = pd.DataFrame(scale_rows).sort_values(
    "area_weighted_mean_abs_shapley",
    ascending=False,
).reset_index(drop=True)
scale_summary.insert(0, "importance_rank", np.arange(1, len(scale_summary) + 1))
scale_summary.to_csv(
    OUTPUT_DIR / "integrated_feature_spatial_map_scale_summary.csv",
    index=False,
)
display(scale_summary)
"""
    ),
    code(
        """
individual_paths = []

for number, feature in enumerate(FEATURES, start=1):
    raster = make_raster(feature)
    limit = feature_limits[feature]

    fig = plt.figure(figsize=(18, 9.5))
    ax = fig.add_axes(
        [0.025, 0.20, 0.95, 0.64],
        projection=ccrs.PlateCarree(),
    )
    add_map_background(ax)

    image = ax.imshow(
        raster,
        origin="upper",
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree(),
        interpolation="none",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        zorder=2,
        rasterized=True,
    )
    add_boundaries(ax)

    colorbar_axis = fig.add_axes([0.20, 0.095, 0.60, 0.032])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(
        "Integrated OOF Shapley (final cropland-fraction units): "
        "blue lowers prediction; red raises prediction",
        fontsize=10.5,
        labelpad=7,
    )

    fig.suptitle(
        f"{FEATURE_LABELS[feature]}\\n"
        f"{feature}  |  {FEATURE_GROUPS[feature]}\\n"
        "Spatial integrated OOF Shapley for final cropland fraction  "
        r"$f(X)=p(X)\\times q(X)$",
        fontsize=15.5,
        y=0.99,
    )
    fig.text(
        0.5,
        0.018,
        "Each colored raster pixel is one sampled 5-arc-minute grid cell. "
        f"Colors are clipped at the area-weighted 99th percentile of |SHAP| "
        f"(±{limit:.4f}). Predictive attribution, not a causal effect.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    output_path = (
        OUTPUT_DIR
        / f"{number:02d}_{feature}_integrated_oof_shap_map.png"
    )
    fig.savefig(
        output_path,
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    individual_paths.append(output_path)
    print(f"[{number:02d}/{len(FEATURES)}] saved: {output_path.name}")

print("Individual maps:", len(individual_paths))
"""
    ),
    code(
        """
# 一覧図：各変数の空間パターンが見えるよう、変数ごとのp99で正規化。
fig, axes = plt.subplots(
    5,
    3,
    figsize=(19, 17),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True,
)
axes = np.asarray(axes).reshape(-1)

for ax, feature in zip(axes, FEATURES):
    add_map_background(ax)
    raster = make_raster(feature)
    limit = feature_limits[feature]
    normalized = np.clip(raster / limit, -1, 1)
    ax.imshow(
        normalized,
        origin="upper",
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree(),
        interpolation="none",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        zorder=2,
        rasterized=True,
    )
    ax.coastlines(
        resolution="110m",
        linewidth=0.30,
        color="#4A4A4A",
        zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.18,
        edgecolor="#777777",
        zorder=3,
    )
    ax.set_title(
        f"{FEATURE_LABELS[feature]}\\n"
        f"scale: ±{limit:.4f}",
        fontsize=10,
    )

for ax in axes[len(FEATURES):]:
    ax.set_visible(False)

scalar = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=-1, vmax=1),
    cmap=cmap,
)
colorbar = fig.colorbar(
    scalar,
    ax=axes[:len(FEATURES)].tolist(),
    orientation="horizontal",
    fraction=0.025,
    pad=0.025,
    shrink=0.65,
)
colorbar.set_label(
    "Within-variable normalized integrated Shapley "
    "(−1 = blue p99 limit; +1 = red p99 limit)"
)
fig.suptitle(
    "Spatial pattern of integrated OOF Shapley by feature\\n"
    r"Final cropland fraction $f(X)=p(X)\\times q(X)$; "
    "variable-specific scales",
    fontsize=18,
)

overview_path = (
    OUTPUT_DIR
    / "00_integrated_feature_shap_maps_overview_variable_specific_scales.png"
)
fig.savefig(
    overview_path,
    dpi=250,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)
print("Saved:", overview_path)
"""
    ),
    code(
        """
# 一覧図：全変数で同じ絶対SHAPスケール。変数間の寄与量を比較できる。
# 各変数の面積加重p99の最大値を、見やすい0.01単位へ切り上げる。
# これにより全13図を同じ±0.20スケールで比較できる。
common_limit_raw = max(feature_limits.values())
common_limit = float(np.ceil(common_limit_raw * 100) / 100)

fig, axes = plt.subplots(
    5,
    3,
    figsize=(19, 17),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True,
)
axes = np.asarray(axes).reshape(-1)

for ax, feature in zip(axes, FEATURES):
    add_map_background(ax)
    raster = make_raster(feature)
    image = ax.imshow(
        raster,
        origin="upper",
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree(),
        interpolation="none",
        cmap=cmap,
        vmin=-common_limit,
        vmax=common_limit,
        zorder=2,
        rasterized=True,
    )
    ax.coastlines(
        resolution="110m",
        linewidth=0.30,
        color="#4A4A4A",
        zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.18,
        edgecolor="#777777",
        zorder=3,
    )
    ax.set_title(FEATURE_LABELS[feature], fontsize=10)

for ax in axes[len(FEATURES):]:
    ax.set_visible(False)

colorbar = fig.colorbar(
    image,
    ax=axes[:len(FEATURES)].tolist(),
    orientation="horizontal",
    fraction=0.025,
    pad=0.025,
    shrink=0.65,
)
colorbar.set_label(
    "Integrated OOF Shapley (final cropland-fraction units); "
    f"common scale ±{common_limit:.4f}"
)
fig.suptitle(
    "Spatial integrated OOF Shapley by feature\\n"
    "Common absolute scale for comparison across variables",
    fontsize=18,
)

common_overview_path = (
    OUTPUT_DIR
    / "00_integrated_feature_shap_maps_overview_common_scale.png"
)
fig.savefig(
    common_overview_path,
    dpi=250,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)
print("Saved:", common_overview_path)
print("Common scale:", common_limit)
"""
    ),
    code(
        """
COMMON_SCALE_DIR = OUTPUT_DIR / "common_scale"
COMMON_SCALE_DIR.mkdir(parents=True, exist_ok=True)
common_scale_paths = []

for number, feature in enumerate(FEATURES, start=1):
    raster = make_raster(feature)

    fig = plt.figure(figsize=(18, 9.5))
    ax = fig.add_axes(
        [0.025, 0.20, 0.95, 0.64],
        projection=ccrs.PlateCarree(),
    )
    add_map_background(ax)

    image = ax.imshow(
        raster,
        origin="upper",
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree(),
        interpolation="none",
        cmap=cmap,
        vmin=-common_limit,
        vmax=common_limit,
        zorder=2,
        rasterized=True,
    )
    add_boundaries(ax)

    colorbar_axis = fig.add_axes([0.20, 0.095, 0.60, 0.032])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=np.linspace(-common_limit, common_limit, 9),
    )
    colorbar.set_label(
        "Integrated OOF Shapley (final cropland-fraction units): "
        "blue lowers prediction; red raises prediction",
        fontsize=10.5,
        labelpad=7,
    )

    fig.suptitle(
        f"{FEATURE_LABELS[feature]}\\n"
        f"{feature}  |  {FEATURE_GROUPS[feature]}\\n"
        "Spatial integrated OOF Shapley; "
        f"COMMON SCALE ±{common_limit:.2f} across all 13 variables",
        fontsize=15.5,
        y=0.99,
    )
    fig.text(
        0.5,
        0.018,
        "Each colored raster pixel is one sampled 5-arc-minute grid cell. "
        "All 13 individual maps use exactly the same color limits. "
        "Predictive attribution, not a causal effect.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    output_path = (
        COMMON_SCALE_DIR
        / f"{number:02d}_{feature}_integrated_oof_shap_map_common_scale.png"
    )
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    common_scale_paths.append(output_path)
    print(f"[{number:02d}/{len(FEATURES)}] saved: {output_path.name}")

common_scale_manifest = {
    "shared_vmin": -common_limit,
    "shared_vmax": common_limit,
    "scale_definition": "maximum feature-specific area-weighted p99, rounded up to 0.01",
    "n_cells": int(len(shap_cells)),
    "features": FEATURES,
    "maps": [str(path) for path in common_scale_paths],
}
(COMMON_SCALE_DIR / "common_scale_manifest.json").write_text(
    json.dumps(common_scale_manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(COMMON_SCALE_DIR / "README.txt").write_text(
    "All 13 maps use the identical integrated OOF Shapley color scale.\\n"
    f"Shared limits: {-common_limit:.2f} to {common_limit:.2f}.\\n"
    "Blue lowers and red raises the final predicted cropland fraction.\\n",
    encoding="utf-8",
)

print("Common-scale individual maps:", len(common_scale_paths))
print("Shared color limits:", -common_limit, common_limit)
"""
    ),
    code(
        """
readme_lines = [
    "# Integrated feature spatial Shapley maps",
    "",
    "Input: `integrated_final_all_cells_bg32_perm32.csv.gz`",
    "",
    "These maps use all 240,000 saved Spatial OOF analysis cells. ",
    "Red raises the final predicted cropland fraction and blue lowers it. ",
    "Each colored raster pixel is one sampled 5-arc-minute grid cell.",
    "",
    "## Overview",
    "",
    "- `00_integrated_feature_shap_maps_overview_variable_specific_scales.png`: spatial pattern, separate scale per feature",
    "- `00_integrated_feature_shap_maps_overview_common_scale.png`: common scale, comparable magnitudes",
    "",
    "## Individual maps",
    "",
]
for path in individual_paths:
    readme_lines.append(f"- `{path.name}`")
readme_lines.extend([
    "",
    "Map limits are the area-weighted 99th percentile of absolute Shapley values.",
    "Extreme values are clipped only in the visualization, not in the saved Shapley data.",
    "SHAP is predictive attribution and should not be interpreted as a causal effect.",
])

readme_path = OUTPUT_DIR / "README.md"
readme_path.write_text("\\n".join(readme_lines) + "\\n", encoding="utf-8")

manifest = {
    "input": str(INPUT_PATH),
    "output_directory": str(OUTPUT_DIR),
    "n_cells": int(len(shap_cells)),
    "n_features": len(FEATURES),
    "features": FEATURES,
    "shapley_definition": "integrated OOF Shapley for final fraction p(X) * q(X)",
    "background_draws": 32,
    "feature_permutations": 32,
    "individual_map_limit": "area-weighted 99th percentile of absolute Shapley",
    "individual_maps": [str(path) for path in individual_paths],
    "overview_variable_specific": str(overview_path),
    "overview_common_scale": str(common_overview_path),
}
(OUTPUT_DIR / "map_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Completed spatial feature maps:", OUTPUT_DIR)
print("Files:", len(list(OUTPUT_DIR.glob("*"))))
"""
    ),
]

nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
