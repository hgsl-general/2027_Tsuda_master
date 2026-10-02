from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/work/tsuda/GAEZ")
SOURCE = ROOT / "cropland_soil_group_wetland_excluded_oof_shap_target2020.ipynb"
DESTINATION = ROOT / "cropland_soil_group_wetland_excluded_groundwater_oof_shap_target2020.ipynb"

OLD_OUTPUT_NAME = "soil_group_calorie_only_50km_target2020"
NEW_OUTPUT_NAME = "soil_group_calorie_only_50km_groundwater_target2020"
OLD_SOIL_MODEL = "soil_group_wetland_excluded"
NEW_SOIL_MODEL = "soil_group_wetland_excluded_groundwater"
OLD_NO_SOIL_MODEL = "no_soil_wetland_excluded"
NEW_NO_SOIL_MODEL = "no_soil_wetland_excluded_groundwater"

GROUNDWATER_FEATURES = [
    "log_fan_water_table_depth_m",
    "log_watergap_total_recharge_mm_yr",
]


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        cell["source"][-1] += "\n"


def require_replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Required patch target not found: {label}")
    return text.replace(old, new)


def add_groundwater_after_glofas_list_entry(text: str) -> str:
    result = []
    for line in text.splitlines(keepends=True):
        result.append(line)
        if line.strip() == '"log_glofas_p10_2020",':
            indent = line[: len(line) - len(line.lstrip())]
            result.append(f'{indent}"log_fan_water_table_depth_m",\n')
            result.append(f'{indent}"log_watergap_total_recharge_mm_yr",\n')
    return "".join(result)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    source = "".join(cell.get("source", []))
    source = source.replace(OLD_OUTPUT_NAME, NEW_OUTPUT_NAME)
    source = source.replace(OLD_NO_SOIL_MODEL, NEW_NO_SOIL_MODEL)
    source = source.replace(OLD_SOIL_MODEL, NEW_SOIL_MODEL)
    if cell["cell_type"] == "code":
        source = add_groundwater_after_glofas_list_entry(source)
    set_source(cell, source)


# Introductory markdown.
set_source(
    notebook["cells"][0],
    """# 耕作地割合：地下水2変数を加えた13変数 Spatial OOF LightGBM / SHAP\n\n"
    "元の `cropland_soil_group_wetland_excluded_oof_shap_target2020.ipynb` は変更せず、\n"
    "同じ分析を、地下水面深度と地下水涵養量を加えた13変数で再実行する独立ノートブックです。\n"
    """,
)
set_source(
    notebook["cells"][1],
    """## 分析仕様\n\n"
    "- 既存11変数は維持\n"
    "- `log_fan_water_table_depth_m`（Fan地下水面深度）を追加\n"
    "- `log_watergap_total_recharge_mm_yr`（WaterGAP地下水涵養量）を追加\n"
    "- 河川距離閾値は10 m³/s、当該セルのGloFAS p10流量を維持\n"
    "- 観測された灌漑面積は使用しない\n"
    "- 240,000セル・同じ10度空間ブロック・5-fold Spatial OOF\n"
    "- 出力は `CroplandRegression/soil_group_calorie_only_50km_groundwater_target2020/` に分離\n"
    """,
)


# Cell 3: imports, paths and separate output directory.
cell = "".join(notebook["cells"][3]["source"])
cell = require_replace(
    cell,
    "import pandas as pd\n",
    "import pandas as pd\nimport xarray as xr\n",
    "xarray import",
)
output_line = (
    'OUTPUT_DIR = GAEZ_DIR / "CroplandRegression" '
    f'/ "{NEW_OUTPUT_NAME}"\n'
)
cell = require_replace(
    cell,
    output_line,
    output_line
    + 'GROUNDWATER_SOURCE_DIR = GAEZ_DIR / "WaterData" / "groundwater"\n'
    + "FAN_WTD_5MIN_PATH = (\n"
    + "    GAEZ_DIR / \"CroplandRegression\"\n"
    + "    / \"water_irrigation_regional_sensitivity_target2020\"\n"
    + "    / \"derived_features\"\n"
    + "    / \"fan_water_table_depth_m_5min_global.npy\"\n"
    + ")\n"
    + "WATERGAP_RECHARGE_PATH = (\n"
    + "    GROUNDWATER_SOURCE_DIR / \"watergap_2_2d\" / \"processed\"\n"
    + "    / \"watergap_total_recharge_mm_yr_2000_2010.nc\"\n"
    + ")\n",
    "groundwater paths",
)
cell = require_replace(
    cell,
    "for required_path in [SOIL_TIF, EXCLUSION_TIF]:\n",
    "for required_path in [\n"
    "    SOIL_TIF,\n"
    "    EXCLUSION_TIF,\n"
    "    FAN_WTD_5MIN_PATH,\n"
    "    WATERGAP_RECHARGE_PATH,\n"
    "]:\n",
    "required groundwater inputs",
)
set_source(notebook["cells"][3], cell)


# Cell 7: load the prepared 5-arc-minute Fan WTD raster and validate alignment.
cell = "".join(notebook["cells"][7]["source"])
cell = require_replace(
    cell,
    "wx_50km_rainfed_calorie = np.load(\n"
    "    WX_50KM_CALORIE_CACHE,\n"
    "    mmap_mode=\"r\",\n"
    ")\n\n",
    "wx_50km_rainfed_calorie = np.load(\n"
    "    WX_50KM_CALORIE_CACHE,\n"
    "    mmap_mode=\"r\",\n"
    ")\n\n"
    "fan_water_table_depth_m = np.load(\n"
    "    FAN_WTD_5MIN_PATH,\n"
    "    mmap_mode=\"r\",\n"
    ")\n\n",
    "Fan WTD load",
)
cell = require_replace(
    cell,
    '    "wx_50km_rainfed_calorie": wx_50km_rainfed_calorie,\n',
    '    "wx_50km_rainfed_calorie": wx_50km_rainfed_calorie,\n'
    '    "fan_water_table_depth_m": fan_water_table_depth_m,\n',
    "Fan WTD shape check",
)
set_source(notebook["cells"][7], cell)


# Cell 9: sample WaterGAP, impute both groundwater variables, and add columns.
cell = "".join(notebook["cells"][9]["source"])
marker = ").astype(np.float32)\n\nsample = pd.DataFrame({"
groundwater_preparation = """
).astype(np.float32)


fan_wtd_sample = take(fan_water_table_depth_m).astype(np.float32)
fan_wtd_valid = np.isfinite(fan_wtd_sample) & (fan_wtd_sample >= 0)
fan_wtd_imputation_m = float(np.median(fan_wtd_sample[fan_wtd_valid]))
fan_wtd_sample = np.where(
    fan_wtd_valid,
    fan_wtd_sample,
    fan_wtd_imputation_m,
).astype(np.float32)


def sample_watergap_recharge(sample_lat, sample_lon):
    variable_name = "groundwater_recharge_total_mm_yr"
    with xr.open_dataset(WATERGAP_RECHARGE_PATH, decode_times=False) as dataset:
        values = np.asarray(dataset[variable_name].values, dtype=np.float32)
        source_lat = np.asarray(dataset["lat"].values, dtype=float)
        source_lon = np.asarray(dataset["lon"].values, dtype=float)

    lat_step = float(np.median(np.diff(source_lat)))
    lon_step = float(np.median(np.diff(source_lon)))
    lat_index = np.rint((sample_lat - source_lat[0]) / lat_step).astype(int)
    lon_index = np.rint((sample_lon - source_lon[0]) / lon_step).astype(int)
    lat_index = np.clip(lat_index, 0, len(source_lat) - 1)
    lon_index = np.clip(lon_index, 0, len(source_lon) - 1)
    return values[lat_index, lon_index]


watergap_recharge_sample = sample_watergap_recharge(
    lat[rows].astype(float),
    lon[cols].astype(float),
).astype(np.float32)
watergap_valid = (
    np.isfinite(watergap_recharge_sample)
    & (watergap_recharge_sample >= 0)
)
watergap_imputation_mm_yr = float(
    np.median(watergap_recharge_sample[watergap_valid])
)
watergap_recharge_sample = np.where(
    watergap_valid,
    watergap_recharge_sample,
    watergap_imputation_mm_yr,
).astype(np.float32)


sample = pd.DataFrame({"""
cell = require_replace(
    cell,
    marker,
    groundwater_preparation,
    "groundwater sample preparation",
)
cell = require_replace(
    cell,
    "    \"wx_50km_rainfed_calorie_top5_raw\": take(\n"
    "        wx_50km_rainfed_calorie\n"
    "    ).astype(np.float32),\n"
    "})\n",
    "    \"wx_50km_rainfed_calorie_top5_raw\": take(\n"
    "        wx_50km_rainfed_calorie\n"
    "    ).astype(np.float32),\n"
    "    \"log_fan_water_table_depth_m\": safe_log1p(fan_wtd_sample),\n"
    "    \"log_watergap_total_recharge_mm_yr\": safe_log1p(\n"
    "        watergap_recharge_sample\n"
    "    ),\n"
    "})\n",
    "groundwater DataFrame columns",
)
cell = require_replace(
    cell,
    'SOIL_FEATURES = BASE_FEATURES + ["soil_group_class"]\n',
    "SOIL_FEATURES = (\n"
    "    BASE_FEATURES[:-2]\n"
    "    + [\"soil_group_class\"]\n"
    "    + BASE_FEATURES[-2:]\n"
    ")\n",
    "13-feature order",
)
cell = require_replace(
    cell,
    'print("calorie-only features:", SOIL_FEATURES)\n',
    'print("final 13 features:", SOIL_FEATURES)\n'
    'print("Fan WTD imputed share:", float(1.0 - fan_wtd_valid.mean()))\n'
    'print("Fan WTD imputation median (m):", fan_wtd_imputation_m)\n'
    'print("WaterGAP imputed share:", float(1.0 - watergap_valid.mean()))\n'
    'print("WaterGAP imputation median (mm/yr):", watergap_imputation_mm_yr)\n',
    "groundwater preprocessing report",
)
set_source(notebook["cells"][9], cell)


# Display labels and group mappings for the two new variables.
cell = "".join(notebook["cells"][21]["source"])
cell = require_replace(
    cell,
    '    "log_city_time_20k_min":\n',
    '    "log_fan_water_table_depth_m":\n'
    '        "Groundwater-table depth",\n\n'
    '    "log_watergap_total_recharge_mm_yr":\n'
    '        "Groundwater recharge",\n\n'
    '    "log_city_time_20k_min":\n',
    "cell 21 groundwater labels",
)
set_source(notebook["cells"][21], cell)

cell = "".join(notebook["cells"][41]["source"])
cell = require_replace(
    cell,
    '    "log_glofas_p10_2020":\n        "Climate and water capital",\n',
    '    "log_glofas_p10_2020":\n        "Climate and water capital",\n'
    '    "log_fan_water_table_depth_m":\n        "Climate and water capital",\n'
    '    "log_watergap_total_recharge_mm_yr":\n        "Climate and water capital",\n',
    "cell 41 groundwater groups",
)
cell = require_replace(
    cell,
    '    "log_glofas_p10_2020":\n        "Low-flow river discharge",\n',
    '    "log_glofas_p10_2020":\n        "Low-flow river discharge",\n'
    '    "log_fan_water_table_depth_m":\n        "Groundwater-table depth",\n'
    '    "log_watergap_total_recharge_mm_yr":\n        "Groundwater recharge",\n',
    "cell 41 groundwater labels",
)
set_source(notebook["cells"][41], cell)

cell = "".join(notebook["cells"][43]["source"])
cell = require_replace(
    cell,
    '    "log_glofas_p10_2020":\n        "climate_water",\n',
    '    "log_glofas_p10_2020":\n        "climate_water",\n'
    '    "log_fan_water_table_depth_m":\n        "climate_water",\n'
    '    "log_watergap_total_recharge_mm_yr":\n        "climate_water",\n',
    "cell 43 groundwater archive groups",
)
set_source(notebook["cells"][43], cell)

cell = "".join(notebook["cells"][47]["source"])
old_dependence_loader = """# dependence_dfがメモリにない場合は保存ファイルから読む
if "dependence_df" not in globals():

    if not DEPENDENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            "説明変数値と統合Shapley値の保存ファイルがありません。\\n"
            "先に前のコードのデータ作成部分を実行してください。\\n"
            f"{DEPENDENCE_DATA_PATH}"
        )

    dependence_df = pd.read_csv(
        DEPENDENCE_DATA_PATH,
        compression="gzip",
    )
"""
new_dependence_loader = """# 保存済み統合Shapleyと、この実行で作成した全セル説明変数から自動生成する。
if "dependence_df" not in globals():

    if DEPENDENCE_DATA_PATH.exists():
        dependence_df = pd.read_csv(
            DEPENDENCE_DATA_PATH,
            compression="gzip",
        )
    else:
        integrated_path = (
            BASE_DIR
            / "integrated_final_all_cells_bg32_perm32.csv.gz"
        )
        if not integrated_path.exists():
            raise FileNotFoundError(
                "統合Shapleyファイルがありません:\\n"
                f"{integrated_path}"
            )

        dependence_features = list(SOIL_FEATURES)
        integrated_columns = (
            [
                "fold",
                "sample_index",
                "row",
                "col",
                "lat",
                "lon",
                "weight",
            ]
            + [
                f"shap_feature__{feature}"
                for feature in dependence_features
            ]
        )
        integrated_values = pd.read_csv(
            integrated_path,
            compression="gzip",
            usecols=integrated_columns,
        ).rename(
            columns={
                f"shap_feature__{feature}": f"shap__{feature}"
                for feature in dependence_features
            }
        )

        feature_values = (
            sample.reset_index()
            .rename(columns={"index": "sample_index"})
        )
        feature_values = feature_values[
            ["sample_index"] + dependence_features
        ].drop_duplicates("sample_index")

        dependence_df = integrated_values.merge(
            feature_values,
            on="sample_index",
            how="left",
            validate="one_to_one",
        )

        missing_feature_rows = dependence_df[
            dependence_features
        ].isna().all(axis=1).sum()
        if missing_feature_rows:
            raise RuntimeError(
                "統合Shapleyに説明変数値を結合できない行があります: "
                f"{missing_feature_rows:,}"
            )

        dependence_df.to_csv(
            DEPENDENCE_DATA_PATH,
            index=False,
            compression="gzip",
        )
        print("created:", DEPENDENCE_DATA_PATH)
"""
cell = require_replace(
    cell,
    old_dependence_loader,
    new_dependence_loader,
    "cell 47 automatic dependence data creation",
)
cell = require_replace(
    cell,
    '    "log_city_time_20k_min":\n',
    '    "log_fan_water_table_depth_m":\n'
    '        "Groundwater-table depth (log scale)",\n\n'
    '    "log_watergap_total_recharge_mm_yr":\n'
    '        "Groundwater recharge (log scale)",\n\n'
    '    "log_city_time_20k_min":\n',
    "cell 47 groundwater labels",
)
set_source(notebook["cells"][47], cell)

cell = "".join(notebook["cells"][48]["source"])
cell = require_replace(
    cell,
    '    "log_city_time_20k_min":\n',
    '    "log_fan_water_table_depth_m":\n'
    '        "Groundwater-table depth\\n(log scale)",\n\n'
    '    "log_watergap_total_recharge_mm_yr":\n'
    '        "Groundwater recharge\\n(log scale)",\n\n'
    '    "log_city_time_20k_min":\n',
    "cell 48 groundwater labels",
)
set_source(notebook["cells"][48], cell)


# Make all major figures persistent and keep late-stage reads in the new folder.
cell = "".join(notebook["cells"][32]["source"])
cell = cell.replace("SAVE_FIGURE = False", "SAVE_FIGURE = True")
cell = require_replace(
    cell,
    "FIGURE_PATH = (\n"
    "    r\"/work/tsuda/GAEZ\"\n"
    "    r\"/final_cropland_dominant_factor_map.png\"\n"
    ")",
    'FIGURE_PATH = str(Path(OUTPUT_DIR) / "final_cropland_dominant_factor_map.png")',
    "dominant map path",
)
set_source(notebook["cells"][32], cell)

cell = "".join(notebook["cells"][35]["source"])
cell = cell.replace("SAVE_SIGNED_MAPS = False", "SAVE_SIGNED_MAPS = True")
cell = require_replace(
    cell,
    "SIGNED_MAP_OUTPUT_DIR = Path(\n    r\"/work/tsuda/GAEZ\"\n)",
    "SIGNED_MAP_OUTPUT_DIR = Path(OUTPUT_DIR)",
    "signed map directory",
)
set_source(notebook["cells"][35], cell)

cell = "".join(notebook["cells"][37]["source"])
cell = cell.replace("SAVE_FIGURE = False", "SAVE_FIGURE = True")
set_source(notebook["cells"][37], cell)

for cell_index in [41, 43, 47]:
    cell = "".join(notebook["cells"][cell_index]["source"])
    cell = require_replace(
        cell,
        "BASE_DIR = Path(\n    r\"/work/tsuda/GAEZ\"\n)",
        "BASE_DIR = Path(OUTPUT_DIR)",
        f"cell {cell_index} base directory",
    )
    set_source(notebook["cells"][cell_index], cell)

cell = "".join(notebook["cells"][48]["source"])
cell = require_replace(
    cell,
    "BASE_DIR = Path(\n"
    "    r\"/work/tsuda/GAEZ\"\n"
    ")\n\n"
    "MODEL_OUTPUT_DIR = (\n"
    "    BASE_DIR\n"
    "    / \"CroplandRegression\"\n"
    f"    / \"{NEW_OUTPUT_NAME}\"\n"
    ")",
    "BASE_DIR = Path(OUTPUT_DIR)\n\nMODEL_OUTPUT_DIR = Path(OUTPUT_DIR)",
    "cell 48 model output directory",
)
set_source(notebook["cells"][48], cell)


# Final machine-readable run manifest.
manifest_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [],
}
set_source(
    manifest_cell,
    """# Save a compact manifest for this executed 13-feature run.
import json
from datetime import datetime, timezone

run_manifest = {
    "notebook": str(Path(r"/work/tsuda/GAEZ/cropland_soil_group_wetland_excluded_groundwater_oof_shap_target2020.ipynb")),
    "output_directory": str(Path(OUTPUT_DIR)),
    "target_year": int(YEAR),
    "n_analysis_rows": int(len(sample)),
    "n_features": int(len(SOIL_FEATURES)),
    "features": list(SOIL_FEATURES),
    "final_model": SOIL_MODEL_NAME,
    "spatial_oof_folds": int(N_SPLITS),
    "fan_wtd_imputed_share": float(1.0 - fan_wtd_valid.mean()),
    "fan_wtd_imputation_median_m": float(fan_wtd_imputation_m),
    "watergap_imputed_share": float(1.0 - watergap_valid.mean()),
    "watergap_imputation_median_mm_yr": float(watergap_imputation_mm_yr),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "global_metrics": global_metrics.to_dict(orient="records"),
}

manifest_path = Path(OUTPUT_DIR) / "groundwater_13feature_run_manifest.json"
manifest_path.write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

result_files = sorted(
    str(path.relative_to(OUTPUT_DIR))
    for path in Path(OUTPUT_DIR).rglob("*")
    if path.is_file()
)
(Path(OUTPUT_DIR) / "result_file_index.txt").write_text(
    "\\n".join(result_files) + "\\n",
    encoding="utf-8",
)

print("13-feature run manifest:", manifest_path)
print("Result files:", len(result_files))
""",
)
notebook["cells"].append(
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 23. 13変数版の実行マニフェスト\n"],
    }
)
notebook["cells"].append(manifest_cell)


notebook.setdefault("metadata", {})["source_notebook"] = str(SOURCE)
notebook["metadata"]["analysis_variant"] = "13 features with Fan WTD and WaterGAP recharge"
notebook["metadata"]["result_directory"] = str(
    ROOT / "CroplandRegression" / NEW_OUTPUT_NAME
)


# Static validation before writing.
code = "\n".join(
    "".join(cell.get("source", []))
    for cell in notebook["cells"]
    if cell["cell_type"] == "code"
)
if OLD_OUTPUT_NAME in code:
    raise RuntimeError("Old output directory remains in generated notebook")
if f'"{OLD_SOIL_MODEL}"' in code or f'"{OLD_NO_SOIL_MODEL}"' in code:
    raise RuntimeError("Old model identifier remains in generated notebook")
for feature in GROUNDWATER_FEATURES:
    if code.count(feature) < 8:
        raise RuntimeError(f"Groundwater feature is not propagated: {feature}")

DESTINATION.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(DESTINATION)
