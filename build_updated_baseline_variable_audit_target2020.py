from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm


ROOT = Path("/work/tsuda/GAEZ")
SOURCE_SCRIPT = ROOT / "run_water_irrigation_regional_sensitivity_target2020.py"
OUTPUT_DIR = (
    ROOT
    / "CroplandRegression"
    / "updated_baseline_variable_audit_target2020"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    {
        "order": 1,
        "feature": "elevation_m",
        "label": "Elevation",
        "label_ja": "標高",
        "group": "Land and soil",
        "status": "existing",
        "type": "continuous",
        "unit": "m",
        "model_input": "raw value",
        "definition": "Elevation of the 5-arc-minute grid cell.",
        "source": "GAEZ elevation layer / elevation_5min.npy",
        "missing_handling": "No missing values in the analysis sample.",
        "rationale": "Controls terrain, temperature gradients and land suitability.",
        "cmap": "terrain",
    },
    {
        "order": 2,
        "feature": "slope",
        "label": "Slope class",
        "label_ja": "傾斜区分",
        "group": "Land and soil",
        "status": "existing",
        "type": "ordinal",
        "unit": "GAEZ class (0–10)",
        "model_input": "class value treated as ordinal numeric",
        "definition": "GAEZ slope class aggregated to the 5-arc-minute grid.",
        "source": "GAEZ slope layer / slope_5min.npy",
        "missing_handling": "No missing values in the analysis sample.",
        "rationale": "Represents mechanization, erosion and cultivation constraints.",
        "cmap": "YlOrBr",
    },
    {
        "order": 3,
        "feature": "exclusion_class",
        "label": "Land-exclusion class",
        "label_ja": "土地除外区分",
        "group": "Land and soil",
        "status": "existing",
        "type": "categorical",
        "unit": "class code",
        "model_input": "LightGBM categorical",
        "definition": "1 none; 2 WDPA IUCN; 3 WDPA other; 4 KBA; 5 permanent wetland; 6 tree cover >80%.",
        "source": "GAEZ v5 land-exclusion layer / gaez_v5_exclusion_5min_mode.npy",
        "missing_handling": "Class 5 is excluded by the model mask; no missing values remain.",
        "rationale": "Separates protected or otherwise constrained land from unconstrained land.",
        "cmap": "Set2",
    },
    {
        "order": 11,
        "feature": "soil_group_class",
        "label": "WRB soil group",
        "label_ja": "WRB土壌グループ",
        "group": "Land and soil",
        "status": "existing",
        "type": "categorical",
        "unit": "class code (1–33)",
        "model_input": "LightGBM categorical",
        "definition": "Modal GAEZ v5 WRB soil group in each 5-arc-minute grid cell.",
        "source": "GAEZ v5 WRB soil group / gaez_v5_wrb_soil_group_5min_mode.npy",
        "missing_handling": "Only valid classes 1–33 are retained.",
        "rationale": "Captures soil properties without imposing an arbitrary linear ordering.",
        "cmap": "gist_ncar",
    },
    {
        "order": 4,
        "feature": "log_city_time_20k_min",
        "label": "Travel time to city ≥20k",
        "label_ja": "人口2万人以上の都市への移動時間",
        "group": "Market access",
        "status": "existing",
        "type": "continuous",
        "unit": "log1p(minutes)",
        "model_input": "log1p(raw minutes)",
        "definition": "Travel time to the nearest city with at least 20,000 people.",
        "source": "Existing accessibility raster / cities_10_1_12deg_min.npy",
        "missing_handling": "Non-negative finite values; no missing values in sample.",
        "rationale": "Represents local market and service access.",
        "cmap": "magma_r",
    },
    {
        "order": 5,
        "feature": "log_port_time_any_min",
        "label": "Travel time to port",
        "label_ja": "港への移動時間",
        "group": "Market access",
        "status": "existing",
        "type": "continuous",
        "unit": "log1p(minutes)",
        "model_input": "log1p(raw minutes)",
        "definition": "Travel time to the nearest port represented by the port-access raster.",
        "source": "Existing accessibility raster / ports_05_1_12deg_min.npy",
        "missing_handling": "Non-negative finite values; no missing values in sample.",
        "rationale": "Represents access to trade and transport infrastructure.",
        "cmap": "magma_r",
    },
    {
        "order": 6,
        "feature": "rainfed_calorie_top5_raw",
        "label": "Rainfed top-5 calorie potential",
        "label_ja": "天水上位5作物カロリー潜在量",
        "group": "Agro-climatic potential",
        "status": "existing",
        "type": "continuous",
        "unit": "kcal/ha",
        "model_input": "non-negative raw value",
        "definition": "Sum of the five highest rainfed calorie potentials among 36 checked crops.",
        "source": "GAEZ crop-potential layers and checked calorie conversion table",
        "missing_handling": "Invalid or negative potential is set to zero by positive_raw().",
        "rationale": "Measures rainfed agricultural suitability and attainable calorie potential.",
        "cmap": "YlGn",
    },
    {
        "order": 7,
        "feature": "irrigation_calorie_gain_top5_raw",
        "label": "Potential irrigation calorie gain",
        "label_ja": "灌漑による潜在カロリー増分",
        "group": "Agro-climatic potential",
        "status": "existing",
        "type": "continuous",
        "unit": "kcal/ha",
        "model_input": "max(irrigated potential − rainfed potential, 0)",
        "definition": "Counterfactual GAEZ yield-potential gain from irrigation; not observed irrigation area.",
        "source": "GAEZ irrigated and rainfed crop-potential layers",
        "missing_handling": "Invalid pairs are set to zero; negative gains are clipped to zero.",
        "rationale": "Captures the agronomic value of water while avoiding observed-cropland leakage.",
        "cmap": "YlGnBu",
    },
    {
        "order": 8,
        "feature": "wx_50km_rainfed_calorie_top5_raw",
        "label": "50-km rainfed calorie context",
        "label_ja": "周囲50kmの天水カロリー潜在量",
        "group": "Agro-climatic potential",
        "status": "existing",
        "type": "continuous",
        "unit": "kcal/ha",
        "model_input": "latitude-aware 50-km box mean of raw values",
        "definition": "Neighbourhood mean of positive rainfed top-5 calorie potential.",
        "source": "Derived from rainfed calorie potential / wx_50km_rainfed_calorie_top5_raw.npy",
        "missing_handling": "Derived from finite non-negative potential.",
        "rationale": "Represents the surrounding agro-climatic production environment.",
        "cmap": "YlGn",
    },
    {
        "order": 9,
        "feature": "log_distance_river_gt10_2020",
        "label": "Distance to reliable river (>10 m³/s)",
        "label_ja": "信頼可能河川（10 m³/s超）までの距離",
        "group": "Surface water",
        "status": "existing / retained",
        "type": "continuous",
        "unit": "log1p(km)",
        "model_input": "log1p(distance in km)",
        "definition": "Distance to nearest cell with 2020 GloFAS p10 discharge above 10 m³/s.",
        "source": "GloFAS 2020 derived river-distance raster",
        "missing_handling": "Non-negative finite values; no missing values in sample.",
        "rationale": "Global factorial OOF tests favored retaining the 10 m³/s threshold.",
        "cmap": "viridis_r",
    },
    {
        "order": 10,
        "feature": "log_glofas_p10_2020",
        "label": "Local GloFAS p10 discharge",
        "label_ja": "当該セルのGloFAS p10流量",
        "group": "Surface water",
        "status": "existing / retained",
        "type": "continuous",
        "unit": "log1p(m³/s)",
        "model_input": "log1p(local 2020 p10 discharge)",
        "definition": "Maximum 2020 p10 discharge represented within the 5-arc-minute cell.",
        "source": "GloFAS 2020 / p10_discharge_max_5min_2020.npy",
        "missing_handling": "Non-negative finite values; no missing values in sample.",
        "rationale": "Retains direct local low-flow information; 50-km flow was not selected.",
        "cmap": "Blues",
    },
    {
        "order": 12,
        "feature": "log_fan_water_table_depth_m",
        "label": "Groundwater-table depth",
        "label_ja": "地下水面深度",
        "group": "Groundwater",
        "status": "NEW",
        "type": "continuous",
        "unit": "log1p(m)",
        "model_input": "median imputation, then log1p(depth in m)",
        "definition": "Depth from land surface to the groundwater table.",
        "source": "Fan et al. global groundwater-table depth map",
        "missing_handling": "Invalid cells imputed with sample median before log1p.",
        "rationale": "Directly captures groundwater accessibility; fixed for the updated baseline.",
        "cmap": "cividis",
    },
    {
        "order": 13,
        "feature": "log_watergap_total_recharge_mm_yr",
        "label": "Groundwater recharge",
        "label_ja": "地下水涵養量",
        "group": "Groundwater",
        "status": "NEW",
        "type": "continuous",
        "unit": "log1p(mm/year)",
        "model_input": "median imputation, then log1p(2000–2010 mean)",
        "definition": "Mean total groundwater recharge over 2000–2010.",
        "source": "WaterGAP 2.2d total groundwater recharge",
        "missing_handling": "Invalid cells imputed with sample median before log1p.",
        "rationale": "Captures renewable groundwater supply; fixed for the updated baseline.",
        "cmap": "PuBuGn",
    },
]

EXCLUDED_CANDIDATES = [
    {
        "feature": "gmia_irrigation_share_50km",
        "decision": "EXCLUDED",
        "reason": "Observed irrigation area can mechanically reveal that cropland exists (target leakage/endogeneity risk).",
    },
    {
        "feature": "log_distance_river_gt1_2020",
        "decision": "NOT SELECTED",
        "reason": "The relaxed 1 m³/s threshold did not improve global Spatial OOF performance over 10 m³/s.",
    },
    {
        "feature": "log_distance_river_gt0p1_2020",
        "decision": "NOT SELECTED",
        "reason": "The relaxed 0.1 m³/s threshold did not improve global Spatial OOF performance over 10 m³/s.",
    },
    {
        "feature": "wx_50km_log_glofas_p10_2020",
        "decision": "NOT SELECTED",
        "reason": "The 50-km surrounding-flow feature did not add global Spatial OOF improvement.",
    },
]


def load_source_module():
    spec = importlib.util.spec_from_file_location("water_analysis", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def feature_frame():
    return pd.DataFrame(FEATURES).sort_values("order").reset_index(drop=True)


def save_catalog_and_spec(catalog, sample, source_metadata):
    catalog.drop(columns="cmap").to_csv(
        OUTPUT_DIR / "updated_baseline_variable_catalog.csv", index=False
    )
    pd.DataFrame(EXCLUDED_CANDIDATES).to_csv(
        OUTPUT_DIR / "excluded_candidate_variables.csv", index=False
    )

    specification = {
        "target_year": 2020,
        "n_features": len(catalog),
        "feature_order": catalog["feature"].tolist(),
        "categorical_features": catalog.loc[
            catalog["type"].eq("categorical"), "feature"
        ].tolist(),
        "new_features": catalog.loc[
            catalog["status"].eq("NEW"), "feature"
        ].tolist(),
        "source_analysis_sample": str(source_metadata["source_sample"]),
        "n_analysis_rows": int(len(sample)),
        "baseline_retrained_in_this_step": False,
        "excluded_candidates": EXCLUDED_CANDIDATES,
    }
    (OUTPUT_DIR / "updated_baseline_feature_specification.json").write_text(
        json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_summary(catalog, sample):
    rows = []
    for meta in catalog.to_dict("records"):
        name = meta["feature"]
        values = pd.to_numeric(sample[name].astype(object), errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "order": meta["order"],
                "feature": name,
                "type": meta["type"],
                "n_rows": len(values),
                "n_missing_after_preprocessing": int(values.isna().sum()),
                "n_unique": int(values.nunique(dropna=True)),
                "min": float(finite.min()),
                "p01": float(finite.quantile(0.01)),
                "median": float(finite.median()),
                "mean": float(finite.mean()),
                "p99": float(finite.quantile(0.99)),
                "max": float(finite.max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "updated_baseline_sample_summary.csv", index=False)
    return summary


def save_overview(catalog):
    group_colors = {
        "Land and soil": "#7A6A53",
        "Market access": "#A75D5D",
        "Agro-climatic potential": "#4C956C",
        "Surface water": "#277DA1",
        "Groundwater": "#5A4E9C",
    }
    groups = list(group_colors)
    fig, axes = plt.subplots(1, len(groups), figsize=(22, 8), constrained_layout=True)
    fig.suptitle(
        "Updated 2020 baseline: 13 selected predictors\n"
        "11 retained predictors + 2 fixed groundwater additions",
        fontsize=22,
        fontweight="bold",
    )
    for ax, group in zip(axes, groups):
        part = catalog[catalog["group"].eq(group)]
        ax.set_facecolor("#F7F7F7")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(len(part), 1) + 1.2)
        ax.axis("off")
        ax.text(
            0.5,
            max(len(part), 1) + 0.8,
            group,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=group_colors[group],
        )
        for i, row in enumerate(part.itertuples(index=False)):
            y = len(part) - i
            face = "#E8E1FF" if row.status == "NEW" else "white"
            ax.text(
                0.5,
                y,
                f"{row.order}. {row.label}\n{row.feature}"
                + ("\nNEW" if row.status == "NEW" else ""),
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold" if row.status == "NEW" else "normal",
                bbox={
                    "boxstyle": "round,pad=0.55",
                    "facecolor": face,
                    "edgecolor": group_colors[group],
                    "linewidth": 1.8 if row.status == "NEW" else 1.0,
                },
            )
    fig.text(
        0.5,
        0.012,
        "Observed irrigation area is excluded. River threshold remains 10 m³/s; local GloFAS p10 remains; 50-km GloFAS is not added.",
        ha="center",
        fontsize=13,
        color="#333333",
    )
    fig.savefig(OUTPUT_DIR / "updated_baseline_variable_overview.png", dpi=220)
    plt.close(fig)


def numeric_values(sample, name):
    return pd.to_numeric(sample[name].astype(object), errors="coerce").to_numpy(float)


def display_values(sample, meta):
    values = numeric_values(sample, meta["feature"])
    if meta["feature"] in {
        "rainfed_calorie_top5_raw",
        "irrigation_calorie_gain_top5_raw",
        "wx_50km_rainfed_calorie_top5_raw",
    }:
        return np.log10(1.0 + np.clip(values, 0, None)), "log10(1 + kcal/ha)"
    return values, meta["unit"]


def one_degree_grid(sample, values, categorical):
    west, east, south, north, step = -180.0, 180.0, -60.0, 85.0, 1.0
    nx = int((east - west) / step)
    ny = int((north - south) / step)
    lon = sample["lon"].to_numpy(float)
    lat = sample["lat"].to_numpy(float)
    ix = np.floor((lon - west) / step).astype(int)
    iy = np.floor((lat - south) / step).astype(int)
    valid = (
        np.isfinite(values)
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )
    work = pd.DataFrame(
        {"cell": iy[valid] * nx + ix[valid], "value": values[valid]}
    )
    if categorical:
        aggregated = work.groupby("cell", sort=False)["value"].agg(
            lambda x: x.value_counts().index[0]
        )
    else:
        aggregated = work.groupby("cell", sort=False)["value"].median()
    grid = np.full(ny * nx, np.nan, dtype=np.float32)
    grid[aggregated.index.to_numpy(int)] = aggregated.to_numpy(np.float32)
    grid = grid.reshape(ny, nx)
    x_edges = np.linspace(west, east, nx + 1)
    y_edges = np.linspace(south, north, ny + 1)
    return x_edges, y_edges, grid


def save_maps(catalog, sample):
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(25, 18),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    fig.suptitle(
        "Updated baseline predictors — global spatial patterns",
        fontsize=24,
        fontweight="bold",
    )
    for ax, meta in zip(axes.flat, catalog.to_dict("records")):
        values, display_unit = display_values(sample, meta)
        categorical = meta["type"] == "categorical"
        x_edges, y_edges, grid = one_degree_grid(sample, values, categorical)
        finite = grid[np.isfinite(grid)]
        kwargs = {"cmap": meta["cmap"]}
        if categorical:
            classes = np.unique(finite).astype(int)
            boundaries = np.arange(classes.min() - 0.5, classes.max() + 1.5)
            kwargs["norm"] = BoundaryNorm(boundaries, plt.get_cmap(meta["cmap"]).N)
        else:
            lower, upper = np.nanquantile(finite, [0.01, 0.99])
            if lower == upper:
                lower, upper = np.nanmin(finite), np.nanmax(finite)
            kwargs.update(vmin=lower, vmax=upper)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            shading="flat",
            transform=projection,
            rasterized=True,
            **kwargs,
        )
        ax.coastlines(resolution="110m", linewidth=0.45, color="#333333")
        ax.set_extent([-180, 180, -60, 85], crs=projection)
        ax.gridlines(draw_labels=False, linewidth=0.25, alpha=0.35)
        status = " [NEW]" if meta["status"] == "NEW" else ""
        ax.set_title(
            f"{meta['order']}. {meta['label']}{status}\n{meta['feature']}",
            fontsize=10.5,
            fontweight="bold" if status else "normal",
        )
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.025, shrink=0.88)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(display_unit, fontsize=8)
        if categorical and len(classes) <= 10:
            cbar.set_ticks(classes)
    for ax in axes.flat[len(catalog) :]:
        ax.set_visible(False)
    fig.text(
        0.5,
        0.003,
        "Square cells are 1° aggregates of the 240,000-row Spatial OOF analysis sample (median for continuous/ordinal variables; mode for categorical variables). Color limits use the 1st–99th percentiles.",
        ha="center",
        fontsize=11,
    )
    fig.savefig(OUTPUT_DIR / "updated_baseline_variable_maps.png", dpi=220)
    plt.close(fig)


def save_distributions(catalog, sample):
    fig, axes = plt.subplots(4, 4, figsize=(22, 17), constrained_layout=True)
    fig.suptitle(
        "Updated baseline predictors — model-input distributions",
        fontsize=23,
        fontweight="bold",
    )
    fig.get_layout_engine().set(rect=(0.015, 0.05, 0.985, 0.96))
    for ax, meta in zip(axes.flat, catalog.to_dict("records")):
        values, display_unit = display_values(sample, meta)
        values = values[np.isfinite(values)]
        if meta["type"] in {"categorical", "ordinal"}:
            codes, counts = np.unique(values.astype(int), return_counts=True)
            ax.bar(codes, counts / counts.sum() * 100, color=meta["cmap"] if False else "#4C78A8")
            ax.set_ylabel("Share of sample (%)")
            ax.set_xticks(codes if len(codes) <= 12 else codes[::4])
        else:
            lower, upper = np.quantile(values, [0.005, 0.995])
            shown = values[(values >= lower) & (values <= upper)]
            ax.hist(shown, bins=55, color="#4C78A8", edgecolor="white", linewidth=0.2)
            ax.set_ylabel("OOF sample cells")
            ax.text(
                0.98,
                0.94,
                "display: 0.5–99.5%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )
        status = " [NEW]" if meta["status"] == "NEW" else ""
        ax.set_title(
            f"{meta['order']}. {meta['label']}{status}\n{meta['feature']}",
            fontsize=10.5,
            fontweight="bold" if status else "normal",
        )
        ax.set_xlabel(display_unit, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.flat[len(catalog) :]:
        ax.set_visible(False)
    fig.text(
        0.5,
        0.003,
        "Distributions describe the balanced 240,000-row Spatial OOF analysis sample; they are not global area shares. Calorie-potential axes use log10(1 + kcal/ha) only for display; the model receives raw non-negative values.",
        ha="center",
        fontsize=11,
    )
    fig.savefig(OUTPUT_DIR / "updated_baseline_variable_distributions.png", dpi=220)
    plt.close(fig)


def save_correlation(catalog, sample):
    selected = catalog.loc[
        ~catalog["type"].eq("categorical"), ["feature", "label"]
    ]
    matrix = pd.DataFrame(
        {name: numeric_values(sample, name) for name in selected["feature"]}
    ).corr(method="spearman")
    matrix.to_csv(OUTPUT_DIR / "updated_baseline_numeric_spearman_correlation.csv")
    labels = selected["label"].tolist()
    fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=48, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix.iloc[i, j]
            color = "white" if abs(value) > 0.55 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, shrink=0.78)
    cbar.set_label("Spearman correlation")
    ax.set_title(
        "Updated baseline: Spearman correlations among numeric/ordinal predictors\n"
        "Categorical land-exclusion and soil-group variables are intentionally omitted",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "updated_baseline_numeric_spearman_correlation.png", dpi=220)
    plt.close(fig)


def save_readme(catalog, summary, source_metadata):
    lines = [
        "# Updated baseline variable audit (target year 2020)",
        "",
        "This folder freezes the variable specification **before** retraining the baseline. No model was trained by this script.",
        "",
        "## Final decision",
        "",
        "- Use 13 predictors: the existing 11 plus groundwater-table depth and groundwater recharge.",
        "- Exclude observed irrigation-area variables because they can mechanically reveal cropland presence.",
        "- Retain the 10 m³/s river-distance threshold and local GloFAS p10 discharge.",
        "- Do not add the 1 or 0.1 m³/s river-distance variants or 50-km GloFAS flow.",
        "- Keep `irrigation_calorie_gain_top5_raw`: it is counterfactual GAEZ agroecological potential, not observed irrigation area.",
        "",
        "## Selected variables",
        "",
        "| # | Group | Variable | Meaning | Model input | Status |",
        "|---:|---|---|---|---|---|",
    ]
    for row in catalog.itertuples(index=False):
        lines.append(
            f"| {row.order} | {row.group} | `{row.feature}` | {row.label_ja} | {row.model_input} | {row.status} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `updated_baseline_variable_catalog.csv`: definitions, units, transformations, sources, missing handling and rationale.",
        "- `updated_baseline_sample_summary.csv`: observed model-input statistics in the 240,000-row OOF sample.",
        "- `updated_baseline_variable_overview.png`: selected features grouped by concept.",
        "- `updated_baseline_variable_maps.png`: square-cell global maps (1° aggregates of the OOF sample).",
        "- `updated_baseline_variable_distributions.png`: model-input distributions.",
        "- `updated_baseline_numeric_spearman_correlation.png` and `.csv`: numeric/ordinal correlation matrix.",
        "- `excluded_candidate_variables.csv`: explicitly rejected water/irrigation candidates and reasons.",
        "- `updated_baseline_feature_specification.json`: machine-readable feature order for the future baseline update.",
        "",
        "## Groundwater missing-value treatment in this sample",
        "",
        f"- Fan groundwater-table depth: {source_metadata['fan_wtd_imputed_share']:.2%} imputed with {source_metadata['fan_wtd_imputation_median_m']:.3f} m before `log1p`.",
        f"- WaterGAP recharge: {source_metadata['watergap_imputed_share']:.4%} imputed with {source_metadata['watergap_imputation_median_mm_yr']:.3f} mm/year before `log1p`.",
        "",
        "## Interpretation note",
        "",
        "The maps and histograms describe the saved, balanced Spatial OOF analysis sample. They are diagnostics for the exact model-ready inputs, not estimates of global land-area shares.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("Load exact model-ready analysis sample.", flush=True)
    analysis = load_source_module()
    sample, old_features, _, _ = analysis.load_analysis_sample()
    catalog = feature_frame()
    expected_old = catalog.loc[catalog["status"].ne("NEW"), "feature"].tolist()
    if old_features != expected_old:
        raise RuntimeError(
            "Existing feature order differs from the source model:\n"
            f"source={old_features}\nexpected={expected_old}"
        )
    final_features = catalog["feature"].tolist()
    missing = [name for name in final_features if name not in sample.columns]
    if missing:
        raise KeyError(f"Missing final features: {missing}")

    alignment_path = analysis.OUTPUT_DIR / "feature_alignment_metadata.json"
    source_metadata = json.loads(alignment_path.read_text(encoding="utf-8"))
    source_metadata["source_sample"] = str(analysis.BASIC_OOF_PATH)

    summary = save_summary(catalog, sample)
    save_catalog_and_spec(catalog, sample, source_metadata)
    print("Create overview.", flush=True)
    save_overview(catalog)
    print("Create square-cell maps.", flush=True)
    save_maps(catalog, sample)
    print("Create distributions.", flush=True)
    save_distributions(catalog, sample)
    print("Create correlation matrix.", flush=True)
    save_correlation(catalog, sample)
    save_readme(catalog, summary, source_metadata)
    print(f"Saved variable audit to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
