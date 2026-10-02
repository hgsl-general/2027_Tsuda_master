from __future__ import annotations

import ast
import json
import uuid
from pathlib import Path

ROOT = Path("/work/tsuda/GAEZ")
SOURCE = ROOT / "cropland_soil_group_wetland_excluded_groundwater_oof_shap_target2020.ipynb"
DESTINATION = ROOT / "cropland_soil_group_wetland_excluded_groundwater_rural_population_oof_shap_target2020.ipynb"

OLD_OUTPUT_NAME = "soil_group_calorie_only_50km_groundwater_target2020"
NEW_OUTPUT_NAME = "soil_group_calorie_only_50km_groundwater_rural_population_target2020"
OLD_SOIL_MODEL = "soil_group_wetland_excluded_groundwater"
NEW_SOIL_MODEL = "soil_group_wetland_excluded_groundwater_rural_population"
OLD_NO_SOIL_MODEL = "no_soil_wetland_excluded_groundwater"
NEW_NO_SOIL_MODEL = "no_soil_wetland_excluded_groundwater_rural_population"

RURAL_FEATURE = "log_rural_population_density_2020"
RURAL_GROUP = "Rural population"


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        cell["source"][-1] += "\n"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Required patch target not found: {label}")
    return text.replace(old, new, 1)


def add_rural_group(text: str, label: str) -> str:
    old = """    "Market access and infrastructure": [
        "log_city_time_20k_min",
        "log_port_time_any_min",
    ],
}"""
    new = """    "Market access and infrastructure": [
        "log_city_time_20k_min",
        "log_port_time_any_min",
    ],
    "Rural population": [
        "log_rural_population_density_2020",
    ],
}"""
    return replace_once(text, old, new, label)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    source = "".join(cell.get("source", []))
    source = source.replace(OLD_OUTPUT_NAME, NEW_OUTPUT_NAME)
    source = source.replace(OLD_NO_SOIL_MODEL, NEW_NO_SOIL_MODEL)
    source = source.replace(OLD_SOIL_MODEL, NEW_SOIL_MODEL)
    set_source(cell, source)

set_source(
    notebook["cells"][0],
    """# 耕作地割合：地下水2変数＋非都市人口密度を加えた14変数 Spatial OOF LightGBM / SHAP

元の13変数版は変更せず、非都市人口密度を追加した独立ノートブックです。
非都市人口密度は、2020年の非都市人口（人/5分セル）をセル面積（km²）で割って
ノートブック内で再計算し、モデルには log1p(人/km²) を入力します。
""",
)
set_source(
    notebook["cells"][1],
    """## 分析仕様

- 13変数版（既存11変数＋地下水面深度＋地下水涵養量）を維持
- log_rural_population_density_2020 を14番目の変数として追加
- 元データから rural population / grid-cell area を再計算
- 保存済み人口密度配列との数値一致を検証
- Rural population を単独のfactor groupとして扱う
- 240,000セル・同じ10度空間ブロック・5-fold Spatial OOF
- 個別SHAP、4-factor統合Shapley、全セルSHAP、Permutation Importanceを実行
- 旧13変数版との直接比較を保存
""",
)

cell = "".join(notebook["cells"][3]["source"])
cell = replace_once(
    cell,
    """)

DERIVED_MASK_DIR.mkdir(parents=True, exist_ok=True)""",
    """)
RURAL_POPULATION_DIR = (
    GAEZ_DIR / "PopulationData" / "rural_population_2020_5min"
)
RURAL_POPULATION_COUNT_PATH = (
    RURAL_POPULATION_DIR / "rural_population_2020_5min.npy"
)
RURAL_POPULATION_CELL_AREA_PATH = (
    RURAL_POPULATION_DIR / "cell_area_km2_5min.npy"
)
RURAL_POPULATION_DENSITY_CHECK_PATH = (
    RURAL_POPULATION_DIR / "rural_population_density_2020_5min.npy"
)

DERIVED_MASK_DIR.mkdir(parents=True, exist_ok=True)""",
    "rural population paths",
)
cell = replace_once(
    cell,
    """    WATERGAP_RECHARGE_PATH,
]:""",
    """    WATERGAP_RECHARGE_PATH,
    RURAL_POPULATION_COUNT_PATH,
    RURAL_POPULATION_CELL_AREA_PATH,
    RURAL_POPULATION_DENSITY_CHECK_PATH,
]:""",
    "required rural population inputs",
)
set_source(notebook["cells"][3], cell)

cell = "".join(notebook["cells"][7]["source"])
fan_marker = """fan_water_table_depth_m = np.load(
    FAN_WTD_5MIN_PATH,
    mmap_mode="r",
)

"""
rural_load = fan_marker + r"""rural_population_count_2020 = np.load(
    RURAL_POPULATION_COUNT_PATH,
    mmap_mode="r",
)
rural_population_cell_area_km2 = np.load(
    RURAL_POPULATION_CELL_AREA_PATH,
    mmap_mode="r",
)
rural_population_density_saved_2020 = np.load(
    RURAL_POPULATION_DENSITY_CHECK_PATH,
    mmap_mode="r",
)

rural_population_density_2020 = np.full(
    shape,
    np.nan,
    dtype=np.float32,
)
rural_density_source_valid = (
    np.isfinite(rural_population_count_2020)
    & (rural_population_count_2020 >= 0)
    & np.isfinite(rural_population_cell_area_km2)
    & (rural_population_cell_area_km2 > 0)
)
np.divide(
    rural_population_count_2020,
    rural_population_cell_area_km2,
    out=rural_population_density_2020,
    where=rural_density_source_valid,
)

rural_density_check_valid = (
    np.isfinite(rural_population_density_2020)
    & np.isfinite(rural_population_density_saved_2020)
)
if not rural_density_check_valid.any():
    raise RuntimeError("No valid cells for rural-population density validation.")

rural_density_validation_max_abs = float(
    np.max(
        np.abs(
            rural_population_density_2020[rural_density_check_valid]
            - rural_population_density_saved_2020[rural_density_check_valid]
        )
    )
)
if not np.allclose(
    rural_population_density_2020[rural_density_check_valid],
    rural_population_density_saved_2020[rural_density_check_valid],
    rtol=1e-5,
    atol=1e-4,
):
    raise RuntimeError(
        "Recalculated rural-population density does not match the saved array. "
        f"Maximum absolute difference: {rural_density_validation_max_abs}"
    )

print(
    "rural population density validation max abs difference:",
    rural_density_validation_max_abs,
)

"""
cell = replace_once(cell, fan_marker, rural_load, "rural density calculation")
cell = replace_once(
    cell,
    '    "fan_water_table_depth_m": fan_water_table_depth_m,\n',
    '    "fan_water_table_depth_m": fan_water_table_depth_m,\n'
    '    "rural_population_count_2020": rural_population_count_2020,\n'
    '    "rural_population_cell_area_km2": rural_population_cell_area_km2,\n'
    '    "rural_population_density_2020": rural_population_density_2020,\n'
    '    "rural_population_density_saved_2020": rural_population_density_saved_2020,\n',
    "rural raster alignment",
)
set_source(notebook["cells"][7], cell)

cell = "".join(notebook["cells"][9]["source"])
cell = replace_once(
    cell,
    "\n\nsample = pd.DataFrame({",
    r"""

rural_population_density_sample = take(
    rural_population_density_2020
).astype(np.float32)
rural_population_density_valid = (
    np.isfinite(rural_population_density_sample)
    & (rural_population_density_sample >= 0)
)
rural_population_density_sample = np.where(
    rural_population_density_valid,
    rural_population_density_sample,
    0.0,
).astype(np.float32)


sample = pd.DataFrame({""",
    "sample rural density",
)
cell = replace_once(
    cell,
    """    "log_watergap_total_recharge_mm_yr": safe_log1p(
        watergap_recharge_sample
    ),
})""",
    """    "log_watergap_total_recharge_mm_yr": safe_log1p(
        watergap_recharge_sample
    ),
    "log_rural_population_density_2020": safe_log1p(
        rural_population_density_sample
    ),
})""",
    "rural DataFrame column",
)
cell = replace_once(
    cell,
    '    "log_watergap_total_recharge_mm_yr",\n]',
    '    "log_watergap_total_recharge_mm_yr",\n'
    '    "log_rural_population_density_2020",\n]',
    "rural base feature",
)
cell = replace_once(
    cell,
    """SOIL_FEATURES = (
    BASE_FEATURES[:-2]
    + ["soil_group_class"]
    + BASE_FEATURES[-2:]
)""",
    """SOIL_FEATURES = (
    BASE_FEATURES[:-3]
    + ["soil_group_class"]
    + BASE_FEATURES[-3:]
)""",
    "14-feature ordering",
)
cell = cell.replace('print("final 13 features:", SOIL_FEATURES)', 'print("final 14 features:", SOIL_FEATURES)')
cell = replace_once(
    cell,
    'print("WaterGAP imputation median (mm/yr):", watergap_imputation_mm_yr)\n',
    'print("WaterGAP imputation median (mm/yr):", watergap_imputation_mm_yr)\n'
    'print("Rural-density invalid sample share:", float(1.0 - rural_population_density_valid.mean()))\n'
    'print("Rural-density range (persons/km2):", float(rural_population_density_sample.min()), float(rural_population_density_sample.max()))\n',
    "rural report",
)
set_source(notebook["cells"][9], cell)

for index in [21, 24]:
    cell = "".join(notebook["cells"][index]["source"])
    cell = add_rural_group(cell, f"cell {index} group")
    if index == 21:
        cell = replace_once(
            cell,
            '    "Market access and infrastructure": "#E6550D",  # 橙\n',
            '    "Market access and infrastructure": "#E6550D",  # 橙\n'
            '    "Rural population": "#756BB1",                  # 紫\n',
            "cell 21 color",
        )
        cell = replace_once(
            cell,
            'FEATURE_LABELS = {\n',
            'FEATURE_LABELS = {\n'
            '    "log_rural_population_density_2020":\n'
            '        "Rural population density",\n\n',
            "cell 21 label",
        )
    else:
        cell = replace_once(
            cell,
            '    "Market access and infrastructure": "#E6550D",\n',
            '    "Market access and infrastructure": "#E6550D",\n'
            '    "Rural population": "#756BB1",\n',
            "cell 24 color",
        )
    set_source(notebook["cells"][index], cell)

for index in [28, 30]:
    cell = "".join(notebook["cells"][index]["source"])
    cell = add_rural_group(cell, f"cell {index} group")
    cell = replace_once(
        cell,
        '    "Market access and infrastructure": "market_infrastructure",\n',
        '    "Market access and infrastructure": "market_infrastructure",\n'
        '    "Rural population": "rural_population",\n',
        f"cell {index} slug",
    )
    cell = replace_once(
        cell,
        '    "Market access and infrastructure": "#f04b06",\n',
        '    "Market access and infrastructure": "#f04b06",\n'
        '    "Rural population": "#756BB1",\n',
        f"cell {index} color",
    )
    cell = cell.replace("3グループ", "4グループ")
    cell = cell.replace("2^3=8", "2^4=16")
    cell = cell.replace("8組合せ", "16組合せ")
    cell = cell.replace("8通り", "16通り")
    set_source(notebook["cells"][index], cell)

cell = "".join(notebook["cells"][32]["source"])
cell = replace_once(
    cell,
    '    "Land and soil capital": {\n',
    r'''    "Rural population": {
        "candidates": [
            "shap__rural_population",
            "group_shap__rural_population",
            "shap_group__rural_population",
            "integrated_shap__rural_population",
            "shapley__rural_population",
        ],
        "tokens": [
            ("rural", "population"),
        ],
        "color": "#756BB1",
    },

    "Land and soil capital": {
''',
    "dominant map rural factor",
)
cell = cell.replace("3グループ", "4グループ")
set_source(notebook["cells"][32], cell)

cell = "".join(notebook["cells"][33]["source"])
cell = replace_once(
    cell,
    '    "Market access and infrastructure": ["log_city_time_20k_min", "log_port_time_any_min"],\n}',
    '    "Market access and infrastructure": ["log_city_time_20k_min", "log_port_time_any_min"],\n'
    '    "Rural population": ["log_rural_population_density_2020"],\n}',
    "stage rural group",
)
cell = replace_once(
    cell,
    '    "Market access and infrastructure": "#F2550A",\n',
    '    "Market access and infrastructure": "#F2550A",\n'
    '    "Rural population": "#756BB1",\n',
    "stage rural color",
)
set_source(notebook["cells"][33], cell)

cell = "".join(notebook["cells"][35]["source"])
cell = replace_once(
    cell,
    '    "Market access and infrastructure",\n]',
    '    "Market access and infrastructure",\n'
    '    "Rural population",\n]',
    "signed factor names",
)
set_source(notebook["cells"][35], cell)

cell = "".join(notebook["cells"][40]["source"])
cell = add_rural_group(cell, "full-cell rural group")
cell = replace_once(
    cell,
    '    "Market access and infrastructure": "market_infrastructure",\n',
    '    "Market access and infrastructure": "market_infrastructure",\n'
    '    "Rural population": "rural_population",\n',
    "full-cell rural slug",
)
cell = cell.replace("Exact 3-factor", "Exact 4-factor")
cell = cell.replace("Approximate 11-feature", "Approximate 14-feature")
cell = cell.replace("3 factors produce 2^3 = 8 coalitions", "4 factors produce 2^4 = 16 coalitions")
set_source(notebook["cells"][40], cell)

cell = "".join(notebook["cells"][41]["source"])
cell = replace_once(
    cell,
    'GROUP_ORDER = [\n',
    'GROUP_ORDER = [\n    "Rural population",\n',
    "summary group order",
)
cell = replace_once(
    cell,
    '    "Land and soil capital": "#9A7831",\n',
    '    "Land and soil capital": "#9A7831",\n'
    '    "Rural population": "#756BB1",\n',
    "summary group color",
)
cell = replace_once(
    cell,
    'FEATURE_TO_GROUP = {\n',
    'FEATURE_TO_GROUP = {\n'
    '    # Rural population\n'
    '    "log_rural_population_density_2020":\n'
    '        "Rural population",\n\n',
    "summary feature group",
)
cell = replace_once(
    cell,
    'FEATURE_LABELS = {\n',
    'FEATURE_LABELS = {\n'
    '    "log_rural_population_density_2020":\n'
    '        "Rural population density",\n',
    "summary feature label",
)
cell = cell.replace("ncol=3,", "ncol=4,")
set_source(notebook["cells"][41], cell)

cell = "".join(notebook["cells"][43]["source"])
cell = replace_once(
    cell,
    '    "market_infrastructure":\n        "Market access and infrastructure",\n',
    '    "market_infrastructure":\n        "Market access and infrastructure",\n'
    '    "rural_population":\n        "Rural population",\n',
    "archive group label",
)
cell = replace_once(
    cell,
    'FEATURE_TO_GROUP = {\n',
    'FEATURE_TO_GROUP = {\n'
    '    "log_rural_population_density_2020":\n'
    '        "rural_population",\n\n',
    "archive feature group",
)
set_source(notebook["cells"][43], cell)

for index in [47, 48]:
    cell = "".join(notebook["cells"][index]["source"])
    cell = replace_once(
        cell,
        '    "log_watergap_total_recharge_mm_yr",\n',
        '    "log_watergap_total_recharge_mm_yr",\n'
        '    "log_rural_population_density_2020",\n',
        f"cell {index} feature",
    )
    label_text = (
        '    "log_rural_population_density_2020":\n'
        '        "Rural population density (log scale)",\n\n'
        if index == 47
        else
        '    "log_rural_population_density_2020":\n'
        '        "Rural population density\\n(log scale)",\n\n'
    )
    cell = replace_once(
        cell,
        '    "log_city_time_20k_min":\n',
        label_text + '    "log_city_time_20k_min":\n',
        f"cell {index} label",
    )
    set_source(notebook["cells"][index], cell)

cell = "".join(notebook["cells"][51]["source"])
cell = add_rural_group(cell, "group permutation rural group")
cell = replace_once(
    cell,
    '    "Market access and infrastructure": "#F04B06",\n',
    '    "Market access and infrastructure": "#F04B06",\n'
    '    "Rural population": "#756BB1",\n',
    "group permutation color",
)
set_source(notebook["cells"][51], cell)

set_source(
    notebook["cells"][52],
    "## 23. 14変数版の実行マニフェスト\n",
)
set_source(
    notebook["cells"][53],
    r'''# Save a compact manifest for this executed 14-feature run.
import json
from datetime import datetime, timezone

if "global_metrics" not in globals():
    global_metrics = pd.read_csv(
        Path(OUTPUT_DIR) / "soil_comparison_global_metrics.csv"
    )
if "SOIL_MODEL_NAME" not in globals():
    SOIL_MODEL_NAME = "soil_group_wetland_excluded_groundwater_rural_population"

run_manifest = {
    "notebook": str(Path(
        r"/work/tsuda/GAEZ/cropland_soil_group_wetland_excluded_groundwater_rural_population_oof_shap_target2020.ipynb"
    )),
    "output_directory": str(Path(OUTPUT_DIR)),
    "target_year": int(YEAR),
    "n_analysis_rows": int(len(sample)),
    "n_features": int(len(SOIL_FEATURES)),
    "features": list(SOIL_FEATURES),
    "final_model": SOIL_MODEL_NAME,
    "spatial_oof_folds": int(N_SPLITS),
    "rural_population_feature": "log_rural_population_density_2020",
    "rural_population_definition": "2020 rural persons per cell divided by cell area in km2, then log1p",
    "rural_population_factor_group": "Rural population",
    "rural_density_validation_max_abs": float(rural_density_validation_max_abs),
    "rural_density_invalid_sample_share": float(
        1.0 - rural_population_density_valid.mean()
    ),
    "fan_wtd_imputed_share": float(1.0 - fan_wtd_valid.mean()),
    "fan_wtd_imputation_median_m": float(fan_wtd_imputation_m),
    "watergap_imputed_share": float(1.0 - watergap_valid.mean()),
    "watergap_imputation_median_mm_yr": float(watergap_imputation_mm_yr),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "global_metrics": global_metrics.to_dict(orient="records"),
}

manifest_path = (
    Path(OUTPUT_DIR)
    / "groundwater_rural_population_14feature_run_manifest.json"
)
manifest_path.write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("14-feature run manifest:", manifest_path)
''',
)

comparison_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 24. 旧13変数版との直接比較\n"],
}
comparison_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [],
}
set_source(
    comparison_code,
    r'''from sklearn.metrics import r2_score

BASELINE_13_DIR = (
    GAEZ_DIR
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_groundwater_target2020"
)
BASELINE_13_PREDICTION_PATH = (
    BASELINE_13_DIR / "soil_comparison_oof_predictions.csv.gz"
)
NEW_14_PREDICTION_PATH = (
    Path(OUTPUT_DIR) / "soil_comparison_oof_predictions.csv.gz"
)

old_prediction_column = "pred_soil_group_wetland_excluded_groundwater"
new_prediction_column = (
    "pred_soil_group_wetland_excluded_groundwater_rural_population"
)
keys = ["row", "col"]

old_predictions = pd.read_csv(
    BASELINE_13_PREDICTION_PATH,
    compression="gzip",
    usecols=keys + ["cropland_fraction", old_prediction_column],
)
new_predictions = pd.read_csv(
    NEW_14_PREDICTION_PATH,
    compression="gzip",
    usecols=keys + ["cropland_fraction", new_prediction_column],
)

comparison = old_predictions.merge(
    new_predictions,
    on=keys,
    how="inner",
    suffixes=("_old13", "_new14"),
    validate="one_to_one",
)
if len(comparison) != len(old_predictions) or len(comparison) != len(new_predictions):
    raise RuntimeError(
        "The 13-feature and 14-feature OOF cell sets do not match: "
        f"{len(old_predictions):,}, {len(new_predictions):,}, {len(comparison):,}"
    )

observed_old = comparison["cropland_fraction_old13"].to_numpy(float)
observed_new = comparison["cropland_fraction_new14"].to_numpy(float)
if not np.allclose(observed_old, observed_new, rtol=0, atol=1e-7):
    raise RuntimeError("Observed cropland fractions differ between model runs.")

weight_frame = sample[["row", "col"]].copy()
weight_frame["area_weight"] = (
    np.ones(len(sample), dtype=float)
    if area_weight is None
    else np.asarray(area_weight, dtype=float)
)
comparison = comparison.merge(
    weight_frame,
    on=keys,
    how="left",
    validate="one_to_one",
)
if comparison["area_weight"].isna().any():
    raise RuntimeError("Area weights could not be joined to all comparison cells.")

observed = observed_new
old_pred = comparison[old_prediction_column].to_numpy(float)
new_pred = comparison[new_prediction_column].to_numpy(float)


def direct_metrics(observed_values, predicted_values, weights):
    residual = predicted_values - observed_values
    if weights is None:
        mse = float(np.mean(residual ** 2))
        mae = float(np.mean(np.abs(residual)))
        r2 = float(r2_score(observed_values, predicted_values))
    else:
        weights = np.asarray(weights, dtype=float)
        valid = np.isfinite(weights) & (weights > 0)
        weights = weights[valid]
        obs = observed_values[valid]
        pred = predicted_values[valid]
        residual = pred - obs
        mse = float(np.average(residual ** 2, weights=weights))
        mae = float(np.average(np.abs(residual), weights=weights))
        r2 = float(r2_score(obs, pred, sample_weight=weights))
    return {"r2": r2, "rmse": float(np.sqrt(mse)), "mae": mae}


comparison_rows = []
for weighting, weights in [
    ("unweighted", None),
    ("area_weighted", comparison["area_weight"].to_numpy(float)),
]:
    old_metrics = direct_metrics(observed, old_pred, weights)
    new_metrics = direct_metrics(observed, new_pred, weights)
    comparison_rows.append({
        "weighting": weighting,
        "n_cells": int(len(comparison)),
        "old_13_r2": old_metrics["r2"],
        "new_14_r2": new_metrics["r2"],
        "r2_change_new_minus_old": new_metrics["r2"] - old_metrics["r2"],
        "old_13_rmse": old_metrics["rmse"],
        "new_14_rmse": new_metrics["rmse"],
        "rmse_improvement_pct": (
            100.0
            * (old_metrics["rmse"] - new_metrics["rmse"])
            / old_metrics["rmse"]
        ),
        "old_13_mae": old_metrics["mae"],
        "new_14_mae": new_metrics["mae"],
        "mae_improvement_pct": (
            100.0
            * (old_metrics["mae"] - new_metrics["mae"])
            / old_metrics["mae"]
        ),
    })

rural_comparison_metrics = pd.DataFrame(comparison_rows)
comparison_metrics_path = (
    Path(OUTPUT_DIR)
    / "rural_population_14feature_vs_13feature_metrics.csv"
)
rural_comparison_metrics.to_csv(comparison_metrics_path, index=False)
display(rural_comparison_metrics.round(6))

area_row = rural_comparison_metrics.loc[
    rural_comparison_metrics["weighting"].eq("area_weighted")
].iloc[0]

group_summary_path = (
    Path(OUTPUT_DIR)
    / "integrated_shapley_archive_bg32_perm32"
    / "integrated_group_shapley_global_summary.csv"
)
group_summary = pd.read_csv(group_summary_path)
group_summary = group_summary.sort_values("importance_share_pct")

group_color_lookup = {
    "Land and soil capital": "#9A7831",
    "Climate and water capital": "#3496C2",
    "Market access and infrastructure": "#F4510B",
    "Rural population": "#756BB1",
}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
metric_names = ["RMSE", "MAE"]
old_values = [area_row["old_13_rmse"], area_row["old_13_mae"]]
new_values = [area_row["new_14_rmse"], area_row["new_14_mae"]]
x_positions = np.arange(len(metric_names))
bar_width = 0.36
axes[0].bar(
    x_positions - bar_width / 2,
    old_values,
    bar_width,
    color="#9E9E9E",
    label="Old 13-variable",
)
axes[0].bar(
    x_positions + bar_width / 2,
    new_values,
    bar_width,
    color="#756BB1",
    label="New 14-variable",
)
axes[0].set_xticks(x_positions, metric_names)
axes[0].set_title("Area-weighted error")
axes[0].legend(frameon=False)
axes[0].grid(axis="y", alpha=0.2)

axes[1].bar(
    ["Old 13-variable", "New 14-variable"],
    [area_row["old_13_r2"], area_row["new_14_r2"]],
    color=["#9E9E9E", "#756BB1"],
)
axes[1].set_title("Area-weighted R²")
axes[1].grid(axis="y", alpha=0.2)
axes[1].tick_params(axis="x", rotation=12)

axes[2].barh(
    group_summary["group"],
    group_summary["importance_share_pct"],
    color=[
        group_color_lookup.get(group, "#777777")
        for group in group_summary["group"]
    ],
)
axes[2].set_xlabel("Importance share (%)")
axes[2].set_title("Integrated grouped OOF Shapley")
axes[2].grid(axis="x", alpha=0.2)

direction = (
    "improved"
    if area_row["rmse_improvement_pct"] > 0
    else "worsened"
)
fig.suptitle(
    "Effect of adding rural-population density as a separate factor\n"
    f"Area-weighted RMSE {area_row['old_13_rmse']:.4f} → "
    f"{area_row['new_14_rmse']:.4f} "
    f"({abs(area_row['rmse_improvement_pct']):.2f}% {direction})",
    fontsize=15,
)
fig.tight_layout(rect=(0, 0, 1, 0.90))

evaluation_figure_path = (
    Path(OUTPUT_DIR)
    / "rural_population_14feature_evaluation_summary.png"
)
fig.savefig(evaluation_figure_path, dpi=300, bbox_inches="tight")
plt.show()

run_manifest["comparison_to_13feature"] = (
    rural_comparison_metrics.to_dict(orient="records")
)
run_manifest["comparison_metrics_file"] = str(comparison_metrics_path)
run_manifest["evaluation_summary_figure"] = str(evaluation_figure_path)
manifest_path.write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

readme_lines = [
    "# Groundwater + rural-population 14-feature model",
    "",
    "Rural population density is recalculated as 2020 rural persons per cell divided by cell area (km2), then transformed with log1p.",
    "Rural population is treated as its own production-factor group.",
    "",
    "## Direct comparison with the 13-feature groundwater model",
    "",
]
for row in rural_comparison_metrics.itertuples(index=False):
    readme_lines.append(
        f"- {row.weighting}: R2 {row.old_13_r2:.6f} to {row.new_14_r2:.6f}; "
        f"RMSE {row.old_13_rmse:.6f} to {row.new_14_rmse:.6f} "
        f"({row.rmse_improvement_pct:+.3f}% improvement); "
        f"MAE {row.old_13_mae:.6f} to {row.new_14_mae:.6f} "
        f"({row.mae_improvement_pct:+.3f}% improvement)"
    )
readme_lines.extend([
    "",
    "SHAP and permutation importance describe predictive attribution, not causal effects.",
])
(Path(OUTPUT_DIR) / "README_results.md").write_text(
    chr(10).join(readme_lines) + chr(10),
    encoding="utf-8",
)

result_files = sorted(
    str(path.relative_to(OUTPUT_DIR))
    for path in Path(OUTPUT_DIR).rglob("*")
    if path.is_file()
)
(Path(OUTPUT_DIR) / "result_file_index.txt").write_text(
    chr(10).join(result_files) + chr(10),
    encoding="utf-8",
)

print("Saved:", comparison_metrics_path)
print("Saved:", evaluation_figure_path)
print("Result files:", len(result_files))
''',
)
notebook["cells"].extend([comparison_markdown, comparison_code])

notebook.setdefault("metadata", {})["source_notebook"] = str(SOURCE)
notebook["metadata"]["analysis_variant"] = (
    "14 features: 13-feature groundwater model plus rural-population density"
)
notebook["metadata"]["result_directory"] = str(
    ROOT / "CroplandRegression" / NEW_OUTPUT_NAME
)

all_code_cells = [
    "".join(cell.get("source", []))
    for cell in notebook["cells"]
    if cell["cell_type"] == "code"
]
for index, code in enumerate(all_code_cells):
    ast.parse(code, filename=f"generated_code_cell_{index}")

combined_code = "\n".join(all_code_cells)
if combined_code.count(RURAL_FEATURE) < 18:
    raise RuntimeError("Rural-population feature was not propagated.")
if combined_code.count(RURAL_GROUP) < 18:
    raise RuntimeError("Rural-population group was not propagated.")

for cell in notebook["cells"]:
    cell.setdefault("id", uuid.uuid4().hex[:8])

DESTINATION.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(DESTINATION)
