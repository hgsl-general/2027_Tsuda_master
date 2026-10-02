from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path


ROOT = Path("/work/tsuda/GAEZ")
GDP_FEATURE = "log_gdp_pc_2020_ppp2017usd"
GDP_GROUP = "Economic prosperity"
GDP_COLOR = "#31A354"
GDP_SLUG = "economic_prosperity"
GDP_PATH = (
    ROOT
    / "CroplandRegression"
    / "prior_study_inputs"
    / "derived_5arcmin_target2020"
    / "gdp_pc_2020_ppp2017usd_5arcmin.npy"
)


VARIANTS = [
    {
        "source": ROOT / "cropland_soil_group_wetland_excluded_groundwater_oof_shap_target2020.ipynb",
        "destination": ROOT / "cropland_soil_group_wetland_excluded_groundwater_gdp_oof_shap_target2020.ipynb",
        "old_output": "soil_group_calorie_only_50km_groundwater_target2020",
        "new_output": "soil_group_calorie_only_50km_groundwater_gdp_target2020",
        "old_soil_model": "soil_group_wetland_excluded_groundwater",
        "new_soil_model": "soil_group_wetland_excluded_groundwater_gdp",
        "old_no_soil_model": "no_soil_wetland_excluded_groundwater",
        "new_no_soil_model": "no_soil_wetland_excluded_groundwater_gdp",
        "base_feature_count": 13,
        "new_feature_count": 14,
        "base_group_count": 3,
        "new_group_count": 4,
        "has_rural": False,
        "comparison_csv": "gdp_14feature_vs_13feature_metrics.csv",
        "comparison_figure": "gdp_14feature_vs_13feature_summary.png",
        "manifest": "groundwater_gdp_14feature_run_manifest.json",
    },
    {
        "source": ROOT / "cropland_soil_group_wetland_excluded_groundwater_rural_population_oof_shap_target2020.ipynb",
        "destination": ROOT / "cropland_soil_group_wetland_excluded_groundwater_rural_population_gdp_oof_shap_target2020.ipynb",
        "old_output": "soil_group_calorie_only_50km_groundwater_rural_population_target2020",
        "new_output": "soil_group_calorie_only_50km_groundwater_rural_population_gdp_target2020",
        "old_soil_model": "soil_group_wetland_excluded_groundwater_rural_population",
        "new_soil_model": "soil_group_wetland_excluded_groundwater_rural_population_gdp",
        "old_no_soil_model": "no_soil_wetland_excluded_groundwater_rural_population",
        "new_no_soil_model": "no_soil_wetland_excluded_groundwater_rural_population_gdp",
        "base_feature_count": 14,
        "new_feature_count": 15,
        "base_group_count": 4,
        "new_group_count": 5,
        "has_rural": True,
        "comparison_csv": "gdp_15feature_vs_14feature_metrics.csv",
        "comparison_figure": "gdp_15feature_vs_14feature_summary.png",
        "manifest": "groundwater_rural_population_gdp_15feature_run_manifest.json",
    },
]


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        cell["source"][-1] += "\n"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def literal_mapping(node: ast.AST) -> dict | None:
    value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
    try:
        result = ast.literal_eval(value)
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def inject_group_support(source: str) -> str:
    """Add GDP to every independent group/label/color definition in a cell."""
    tree = ast.parse(source)
    insertions: dict[int, list[str]] = {}

    group_feature_dicts = {
        "FACTOR_GROUPS",
        "FINAL_FACTOR_GROUPS",
        "STAGE_FACTOR_GROUPS",
        "GROUPS",
    }
    color_dicts = {
        "FACTOR_COLORS",
        "GROUP_COLORS",
        "STAGE_GROUP_COLORS",
        "group_colors",
        "group_color_lookup",
    }
    slug_dicts = {"GROUP_SLUG", "FACTOR_SLUGS"}
    feature_lists = {"DEPENDENCE_FEATURES", "MAIN_PLOT_FEATURES", "FEATURES"}
    group_lists = {"FACTOR_NAMES", "GROUP_ORDER"}

    for node in tree.body:
        name = assigned_name(node)
        if name is None or node.end_lineno is None:
            continue
        statements: list[str] = []

        if name in group_feature_dicts:
            statements.append(
                f'{name}.setdefault("{GDP_GROUP}", [GDP_FEATURE])'
            )
        elif name in color_dicts and (
            name != "group_colors"
            or isinstance(getattr(node, "value", None), ast.Dict)
        ):
            statements.append(
                f'{name}.setdefault("{GDP_GROUP}", "{GDP_COLOR}")'
            )
        elif name in slug_dicts:
            statements.append(
                f'{name}.setdefault("{GDP_GROUP}", "{GDP_SLUG}")'
            )
        elif name == "FEATURE_LABELS":
            statements.append(
                f'{name}.setdefault(GDP_FEATURE, "GDP per capita")'
            )
        elif name in {"FEATURE_TO_FACTOR", "FEATURE_TO_GROUP"}:
            mapping = literal_mapping(node)
            mapping_values = set(mapping.values()) if mapping else set()
            value = (
                GDP_SLUG
                if any(
                    item in mapping_values
                    for item in {"land_soil", "climate_water", "rural_population"}
                )
                else GDP_GROUP
            )
            statements.append(f'{name}.setdefault(GDP_FEATURE, "{value}")')
        elif name == "GROUP_LABELS":
            statements.append(
                f'{name}.setdefault("{GDP_SLUG}", "{GDP_GROUP}")'
            )
        elif name == "FACTOR_SPECS":
            statements.append(
                f'''{name}.setdefault(
    "{GDP_GROUP}",
    {{
        "candidates": [
            "shap__{GDP_SLUG}",
            "group_shap__{GDP_SLUG}",
            "shap_group__{GDP_SLUG}",
            "integrated_shap__{GDP_SLUG}",
            "shapley__{GDP_SLUG}",
        ],
        "tokens": [("economic", "prosperity"), ("gdp",)],
        "color": "{GDP_COLOR}",
    }},
)'''
            )
        elif name in feature_lists:
            statements.append(
                f'''if GDP_FEATURE not in {name}:
    {name}.append(GDP_FEATURE)'''
            )
        elif name in group_lists:
            statements.append(
                f'''if "{GDP_GROUP}" not in {name}:
    {name}.append("{GDP_GROUP}")'''
            )

        if statements:
            insertions.setdefault(node.end_lineno, []).extend(statements)

    lines = source.splitlines()
    for line_number in sorted(insertions, reverse=True):
        block = []
        for statement in insertions[line_number]:
            block.extend(["", *statement.splitlines()])
        lines[line_number:line_number] = block
    result = "\n".join(lines)
    if source.endswith("\n"):
        result += "\n"
    ast.parse(result)
    return result


def patch_settings_cell(source: str) -> str:
    gdp_path_block = f'''GDP_PC_2020_PATH = Path(
    r"{GDP_PATH}"
)

'''
    source = replace_once(
        source,
        "DERIVED_MASK_DIR.mkdir(parents=True, exist_ok=True)",
        gdp_path_block + "DERIVED_MASK_DIR.mkdir(parents=True, exist_ok=True)",
        "GDP path",
    )
    source = replace_once(
        source,
        "    WATERGAP_RECHARGE_PATH,\n",
        "    WATERGAP_RECHARGE_PATH,\n    GDP_PC_2020_PATH,\n",
        "GDP required path",
    )
    return source


def patch_loading_cell(source: str) -> str:
    marker = "for name, raster in {"
    load_block = '''gdp_pc_2020_ppp2017usd = np.load(
    GDP_PC_2020_PATH,
    mmap_mode="r",
)

'''
    source = replace_once(source, marker, load_block + marker, "GDP raster load")
    source = replace_once(
        source,
        '    "fan_water_table_depth_m": fan_water_table_depth_m,\n',
        '    "fan_water_table_depth_m": fan_water_table_depth_m,\n'
        '    "gdp_pc_2020_ppp2017usd": gdp_pc_2020_ppp2017usd,\n',
        "GDP shape validation",
    )
    return source


def patch_sample_cell(source: str) -> str:
    gdp_sample_block = '''gdp_pc_sample = take(
    gdp_pc_2020_ppp2017usd
).astype(np.float32)
gdp_pc_valid = (
    np.isfinite(gdp_pc_sample)
    & (gdp_pc_sample >= 0)
)
if not gdp_pc_valid.any():
    raise RuntimeError("No valid GDP-per-capita values in the analysis sample.")
gdp_pc_imputation_ppp2017usd = float(
    np.median(gdp_pc_sample[gdp_pc_valid])
)
gdp_pc_sample = np.where(
    gdp_pc_valid,
    gdp_pc_sample,
    gdp_pc_imputation_ppp2017usd,
).astype(np.float32)

'''
    source = replace_once(
        source,
        "sample = pd.DataFrame({",
        gdp_sample_block + "sample = pd.DataFrame({",
        "GDP sample and imputation",
    )
    source = replace_once(
        source,
        'sample = pd.DataFrame({\n',
        'sample = pd.DataFrame({\n'
        '    "log_gdp_pc_2020_ppp2017usd": safe_log1p(gdp_pc_sample),\n',
        "GDP DataFrame column",
    )

    tree = ast.parse(source)
    soil_node = next(
        node
        for node in tree.body
        if assigned_name(node) == "SOIL_FEATURES"
    )
    lines = source.splitlines()
    insertion = [
        "",
        f'GDP_FEATURE = "{GDP_FEATURE}"',
        "if GDP_FEATURE not in BASE_FEATURES:",
        "    BASE_FEATURES.append(GDP_FEATURE)",
        "if GDP_FEATURE not in SOIL_FEATURES:",
        "    SOIL_FEATURES.append(GDP_FEATURE)",
        "",
    ]
    lines[soil_node.end_lineno:soil_node.end_lineno] = insertion
    source = "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    # The source notebooks hard-code the pre-GDP feature count in this label.
    # Keep the output accurate after appending GDP per capita.
    source = re.sub(
        r'print\("final \d+ features:", SOIL_FEATURES\)',
        'print("final features:", SOIL_FEATURES)',
        source,
    )

    source += '''
print("GDP per-capita valid share:", float(gdp_pc_valid.mean()))
print("GDP per-capita imputed cells:", int((~gdp_pc_valid).sum()))
print(
    "GDP per-capita imputation median (PPP 2017 international USD/person):",
    gdp_pc_imputation_ppp2017usd,
)
'''
    ast.parse(source)
    return source


def patch_signed_map_layout(source: str, group_count: int) -> str:
    if "def plot_signed_factor_maps(" not in source:
        return source

    start = source.index("    fig, axes = plt.subplots(")
    loop_marker = "\n\n    for ax, factor_name in zip("
    loop_start = source.index(loop_marker, start)
    old_block = source[start:loop_start]

    ncols = 2 if group_count <= 4 else 3
    nrows = int(np_ceil(group_count / ncols))
    new_block = f'''    fig, axes = plt.subplots(
        {nrows},
        {ncols},
        figsize=({10 * ncols}, {6 * nrows}),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)'''
    source = source[:start] + new_block + source[loop_start:]

    ylabel_marker = '    axes[0].set_ylabel("Latitude")'
    hide_block = '''    for unused_ax in axes[len(FACTOR_NAMES):]:
        unused_ax.set_visible(False)

'''
    source = replace_once(
        source,
        ylabel_marker,
        hide_block + ylabel_marker,
        "hide unused signed-map axes",
    )
    source = source.replace("    axes[2].legend(\n", "    axes[len(FACTOR_NAMES) - 1].legend(\n")
    source = source.replace("    axes[-1].legend(\n", "    axes[len(FACTOR_NAMES) - 1].legend(\n")
    source = source.replace("3生産要素", f"{group_count}生産要素")
    source = source.replace("# 3要素", f"# {group_count}要素")
    source = source.replace("# 3パネル", f"# {group_count}パネル")
    ast.parse(source)
    return source


def np_ceil(value: float) -> int:
    return int(-(-value // 1))


def comparison_code(variant: dict) -> str:
    old_dir = ROOT / "CroplandRegression" / variant["old_output"]
    old_model = variant["old_soil_model"]
    new_model = variant["new_soil_model"]
    return f'''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

BASELINE_OUTPUT_DIR = Path(r"{old_dir}")
CURRENT_OUTPUT_DIR = Path(OUTPUT_DIR)
baseline_path = BASELINE_OUTPUT_DIR / "soil_comparison_oof_predictions.csv.gz"
current_path = CURRENT_OUTPUT_DIR / "soil_comparison_oof_predictions.csv.gz"

baseline_oof = pd.read_csv(
    baseline_path,
    usecols=["row", "col", "cropland_fraction", "pred_{old_model}"],
).rename(columns={{"pred_{old_model}": "prediction_before_gdp"}})
current_oof = pd.read_csv(
    current_path,
    usecols=["row", "col", "cropland_fraction", "pred_{new_model}"],
).rename(columns={{"pred_{new_model}": "prediction_after_gdp"}})

comparison_cells = baseline_oof.merge(
    current_oof,
    on=["row", "col"],
    how="inner",
    suffixes=("_baseline", "_current"),
    validate="one_to_one",
)
if len(comparison_cells) != len(sample):
    raise RuntimeError(
        f"GDP comparison must use the same analysis cells: "
        f"{{len(comparison_cells):,}} != {{len(sample):,}}"
    )
if not np.allclose(
    comparison_cells["cropland_fraction_baseline"],
    comparison_cells["cropland_fraction_current"],
):
    raise RuntimeError("Observed cropland fractions differ between runs.")

weight_lookup = sample[["row", "col"]].copy()
if area_weight is None:
    weight_lookup["area_weight"] = 1.0
else:
    weight_lookup["area_weight"] = np.asarray(area_weight, dtype=float)
comparison_cells = comparison_cells.merge(
    weight_lookup,
    on=["row", "col"],
    how="left",
    validate="one_to_one",
)

def metric_row(weighting, weights):
    observed = comparison_cells["cropland_fraction_current"].to_numpy(float)
    before = comparison_cells["prediction_before_gdp"].to_numpy(float)
    after = comparison_cells["prediction_after_gdp"].to_numpy(float)
    if weights is None:
        weights = np.ones(len(observed), dtype=float)
    weights = np.asarray(weights, dtype=float)

    def metrics(prediction):
        error = prediction - observed
        return {{
            "r2": r2_score(observed, prediction, sample_weight=weights),
            "rmse": float(np.sqrt(np.average(error ** 2, weights=weights))),
            "mae": float(np.average(np.abs(error), weights=weights)),
        }}

    old = metrics(before)
    new = metrics(after)
    return {{
        "weighting": weighting,
        "n_cells": len(observed),
        "before_gdp_r2": old["r2"],
        "after_gdp_r2": new["r2"],
        "r2_change": new["r2"] - old["r2"],
        "before_gdp_rmse": old["rmse"],
        "after_gdp_rmse": new["rmse"],
        "rmse_improvement_pct": 100 * (old["rmse"] - new["rmse"]) / old["rmse"],
        "before_gdp_mae": old["mae"],
        "after_gdp_mae": new["mae"],
        "mae_improvement_pct": 100 * (old["mae"] - new["mae"]) / old["mae"],
    }}

gdp_comparison_metrics = pd.DataFrame([
    metric_row("unweighted", None),
    metric_row("area_weighted", comparison_cells["area_weight"].to_numpy(float)),
])
comparison_path = CURRENT_OUTPUT_DIR / "{variant['comparison_csv']}"
gdp_comparison_metrics.to_csv(comparison_path, index=False)
display(gdp_comparison_metrics)

area_result = gdp_comparison_metrics.loc[
    gdp_comparison_metrics["weighting"].eq("area_weighted")
].iloc[0]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, metric, label in [
    (axes[0], "rmse", "RMSE (lower is better)"),
    (axes[1], "mae", "MAE (lower is better)"),
]:
    values = [area_result[f"before_gdp_{{metric}}"], area_result[f"after_gdp_{{metric}}"]]
    bars = ax.bar(["Before GDP", "After GDP"], values, color=["#8C8C8C", "{GDP_COLOR}"])
    ax.set_title(label)
    ax.bar_label(bars, fmt="%.4f")
axes[2].bar(["Before GDP", "After GDP"], [area_result["before_gdp_r2"], area_result["after_gdp_r2"]], color=["#8C8C8C", "{GDP_COLOR}"])
axes[2].set_title("R² (higher is better)")
fig.suptitle("Direct Spatial OOF comparison: baseline vs + GDP per capita", fontsize=16)
fig.tight_layout()
comparison_figure_path = CURRENT_OUTPUT_DIR / "{variant['comparison_figure']}"
fig.savefig(comparison_figure_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", comparison_path)
print("Saved:", comparison_figure_path)
'''


def manifest_code(variant: dict) -> str:
    rural_lines = ""
    if variant["has_rural"]:
        rural_lines = '''    "rural_population_feature": "log_rural_population_density_2020",
    "rural_population_factor_group": "Rural population",
'''
    return f'''# Save a compact manifest for this GDP-augmented run.
import json
from datetime import datetime, timezone

run_manifest = {{
    "notebook": str(Path(r"{variant['destination']}")),
    "output_directory": str(Path(OUTPUT_DIR)),
    "target_year": int(YEAR),
    "n_analysis_rows": int(len(sample)),
    "n_features": int(len(SOIL_FEATURES)),
    "features": list(SOIL_FEATURES),
    "final_model": SOIL_MODEL_NAME,
    "spatial_oof_folds": int(N_SPLITS),
    "gdp_feature": GDP_FEATURE,
    "gdp_definition": "2020 GDP per capita, PPP 2017 international USD/person, then log1p",
    "gdp_source_file": str(GDP_PC_2020_PATH),
    "gdp_factor_group": "{GDP_GROUP}",
    "gdp_valid_sample_share": float(gdp_pc_valid.mean()),
    "gdp_imputed_cells": int((~gdp_pc_valid).sum()),
    "gdp_imputation_median_ppp2017usd": float(gdp_pc_imputation_ppp2017usd),
{rural_lines}    "fan_wtd_imputed_share": float(1.0 - fan_wtd_valid.mean()),
    "fan_wtd_imputation_median_m": float(fan_wtd_imputation_m),
    "watergap_imputed_share": float(1.0 - watergap_valid.mean()),
    "watergap_imputation_median_mm_yr": float(watergap_imputation_mm_yr),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "global_metrics": global_metrics.to_dict(orient="records"),
}}

manifest_path = Path(OUTPUT_DIR) / "{variant['manifest']}"
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
print("GDP run manifest:", manifest_path)
print("Result files:", len(result_files))
'''


def build_variant(variant: dict) -> None:
    if not variant["source"].is_file():
        raise FileNotFoundError(variant["source"])
    notebook = json.loads(variant["source"].read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        source = "".join(cell.get("source", []))
        source = source.replace(variant["old_output"], variant["new_output"])
        source = source.replace(variant["old_no_soil_model"], variant["new_no_soil_model"])
        source = source.replace(variant["old_soil_model"], variant["new_soil_model"])
        source = source.replace(str(variant["source"]), str(variant["destination"]))
        set_source(cell, source)

    title = (
        f"# 耕作地割合：地下水2変数"
        + ("＋非都市人口密度" if variant["has_rural"] else "")
        + "＋一人当たりGDP "
        + f"{variant['new_feature_count']}変数 Spatial OOF LightGBM / SHAP\n\n"
        + "2020年のPPP一人当たりGDPをlog1p変換して追加した独立Notebookです。\n"
        + "元Notebookと出力フォルダは変更しません。\n"
    )
    set_source(notebook["cells"][0], title)
    set_source(
        notebook["cells"][1],
        f'''## 分析仕様

- 元の{variant['base_feature_count']}変数モデルを維持
- `log_gdp_pc_2020_ppp2017usd`を追加し、合計{variant['new_feature_count']}変数
- GDPは2020年、PPP 2017国際ドル/人、5分グリッド
- GDP欠損は同一分析標本内の中央値で補完し、標本を変えない
- GDPを単独の`Economic prosperity`因子グループとして扱う
- 同じ240,000セル・10度空間ブロック・5-fold Spatial OOF
- 段階別TreeSHAP、統合Shapley、全セルSHAP、Permutation importanceを実行
- GDP追加前モデルとの直接比較を保存
''',
    )

    set_source(notebook["cells"][3], patch_settings_cell("".join(notebook["cells"][3]["source"])))
    set_source(notebook["cells"][7], patch_loading_cell("".join(notebook["cells"][7]["source"])))
    set_source(notebook["cells"][9], patch_sample_cell("".join(notebook["cells"][9]["source"])))

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code" or index in {3, 7, 9}:
            continue
        source = "".join(cell.get("source", []))
        source = inject_group_support(source)
        source = patch_signed_map_layout(source, variant["new_group_count"])
        set_source(cell, source)

    # Replace the old manifest and any old comparison cells with GDP-specific versions.
    manifest_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
        and "run_manifest =" in "".join(cell.get("source", []))
    )
    notebook["cells"] = notebook["cells"][:manifest_index]
    notebook["cells"].extend(
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## 実行マニフェスト（{variant['new_feature_count']}変数GDP追加版）\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": manifest_code(variant).splitlines(keepends=True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## GDP追加前{variant['base_feature_count']}変数版との直接比較\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": comparison_code(variant).splitlines(keepends=True),
            },
        ]
    )

    notebook.setdefault("metadata", {})["analysis_variant"] = (
        f"{variant['new_feature_count']} features with GDP per capita; "
        f"derived from {variant['base_feature_count']}-feature model"
    )
    notebook["metadata"]["source_notebook"] = str(variant["source"])
    notebook["metadata"]["result_directory"] = str(
        ROOT / "CroplandRegression" / variant["new_output"]
    )
    notebook["metadata"]["gdp_source"] = str(GDP_PATH)

    for cell in notebook["cells"]:
        cell["id"] = uuid.uuid4().hex[:8]

    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, code in enumerate(code_cells):
        ast.parse(code, filename=f"{variant['destination'].name}:cell{index}")

    combined = "\n".join(code_cells)
    required_tokens = [
        GDP_FEATURE,
        GDP_GROUP,
        str(GDP_PATH),
        variant["new_soil_model"],
        variant["new_no_soil_model"],
        variant["new_output"],
    ]
    for token in required_tokens:
        if token not in combined:
            raise RuntimeError(f"Generated notebook is missing: {token}")
    old_model_literal = f'"{variant["old_soil_model"]}"'
    if old_model_literal in combined:
        raise RuntimeError(
            f"Old model literal remains in generated code: {old_model_literal}"
        )

    variant["destination"].write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("Created:", variant["destination"])
    print("Code cells:", len(code_cells))


def main() -> None:
    if not GDP_PATH.is_file():
        raise FileNotFoundError(GDP_PATH)
    for variant in VARIANTS:
        build_variant(variant)


if __name__ == "__main__":
    main()
