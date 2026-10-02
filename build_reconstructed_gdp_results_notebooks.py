from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path("/work/tsuda/GAEZ")

VARIANTS = [
    {
        "title": "Groundwater + GDP per capita — reconstructed full results",
        "subtitle": "14-predictor Spatial OOF / SHAP analysis",
        "result_dir": ROOT
        / "CroplandRegression"
        / "soil_group_calorie_only_50km_groundwater_gdp_target2020",
        "destination": ROOT
        / "cropland_soil_group_wetland_excluded_groundwater_gdp_oof_shap_target2020_results_full_executed.ipynb",
        "gdp_comparison": "gdp_14feature_vs_13feature_metrics.csv",
        "gdp_figure": "gdp_14feature_vs_13feature_summary.png",
        "manifest": "groundwater_gdp_14feature_run_manifest.json",
        "individual_permutation": (
            "permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_gdp_"
            "spatial_oof_permutation_importance.csv"
        ),
        "individual_permutation_figure": (
            "permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_gdp_"
            "spatial_oof_permutation_importance.png"
        ),
        "group_permutation": (
            "group_permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_gdp_"
            "spatial_oof_group_permutation_importance.csv"
        ),
        "group_permutation_figure": (
            "group_permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_gdp_"
            "spatial_oof_group_permutation_importance.png"
        ),
    },
    {
        "title": (
            "Groundwater + rural population + GDP per capita — "
            "reconstructed full results"
        ),
        "subtitle": "15-predictor Spatial OOF / SHAP analysis",
        "result_dir": ROOT
        / "CroplandRegression"
        / "soil_group_calorie_only_50km_groundwater_rural_population_gdp_target2020",
        "destination": ROOT
        / "cropland_soil_group_wetland_excluded_groundwater_rural_population_gdp_oof_shap_target2020_results_full_executed.ipynb",
        "gdp_comparison": "gdp_15feature_vs_14feature_metrics.csv",
        "gdp_figure": "gdp_15feature_vs_14feature_summary.png",
        "manifest": "groundwater_rural_population_gdp_15feature_run_manifest.json",
        "individual_permutation": (
            "permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_rural_population_gdp_"
            "spatial_oof_permutation_importance.csv"
        ),
        "individual_permutation_figure": (
            "permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_rural_population_gdp_"
            "spatial_oof_permutation_importance.png"
        ),
        "group_permutation": (
            "group_permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_rural_population_gdp_"
            "spatial_oof_group_permutation_importance.csv"
        ),
        "group_permutation_figure": (
            "group_permutation_importance_final_oof/"
            "soil_group_wetland_excluded_groundwater_rural_population_gdp_"
            "spatial_oof_group_permutation_importance.png"
        ),
    },
]


TABLES = [
    ("soil_comparison_global_metrics.csv", "土壌群なし／ありモデルの全球Spatial OOF指標"),
    ("soil_comparison_fold_metrics.csv", "fold別Spatial OOF指標"),
    ("soil_model_metric_deltas.csv", "土壌群追加による指標差"),
    ("soil_comparison_residual_morans_i.csv", "OOF残差のMoran's I"),
    ("soil_model_oof_shap_summary.csv", "個別変数のOOF SHAP重要度"),
    ("soil_model_oof_shap_fold_summary.csv", "個別変数のfold別OOF SHAP"),
    ("soil_model_oof_shap_factor_groups_corrected.csv", "生産要素群別OOF SHAP"),
    ("soil_group_class_oof_shap_summary.csv", "WRB土壌群別のsigned OOF SHAP"),
    ("integrated_final_cropland_group_shap_global.csv", "最終耕作地割合の統合グループShapley"),
    ("integrated_final_cropland_group_shap_by_fold.csv", "統合グループShapleyのfold別集計"),
    ("integrated_final_group_summary_bg32_perm32.csv", "全セル統合グループShapley集計"),
    ("integrated_final_feature_summary_bg32_perm32.csv", "全セル統合個別変数Shapley集計"),
    (
        "integrated_shapley_archive_bg32_perm32/integrated_group_shapley_global_summary.csv",
        "統合グループShapleyアーカイブ集計",
    ),
    (
        "integrated_shapley_archive_bg32_perm32/integrated_feature_shapley_global_summary.csv",
        "統合個別変数Shapleyアーカイブ集計",
    ),
]


LARGE_DATA_SAMPLES = [
    ("soil_comparison_oof_predictions.csv.gz", "Spatial OOF予測"),
    ("soil_model_oof_local_feature_shap.csv.gz", "段階別ローカルOOF SHAP"),
    ("soil_model_oof_local_group_shap.csv.gz", "要素群別ローカルOOF SHAP"),
    ("soil_group_oof_local_shap.csv.gz", "土壌群モデルのローカルSHAP"),
    ("integrated_final_cropland_group_shap_local.csv.gz", "統合グループShapley（セル別）"),
    ("integrated_final_all_cells_bg32_perm32.csv.gz", "全セル統合Shapley"),
]


FIGURE_SECTIONS = [
    (
        "モデル精度とGDP追加効果",
        [
            ("model_input_variable_maps.png", "モデルが使用した全説明変数の全球分布"),
            ("soil_ablation_metrics.png", "土壌群なし／ありモデルのSpatial OOF精度"),
            ("spatial_oof_predictions_residuals.png", "Spatial OOF予測と残差の空間分布"),
            ("final_spatial_oof_model_diagnostics.png", "最終モデルのSpatial OOF診断"),
        ],
    ),
    (
        "OOF SHAPと統合Shapley",
        [
            ("individual_feature_oof_shap_factor_colored.png", "個別変数OOF SHAP重要度"),
            ("factor_group_oof_shap_corrected.png", "生産要素群別OOF SHAP"),
            ("factor_group_oof_shap_old_vs_corrected.png", "旧集計と修正集計の比較"),
            ("soil_group_signed_oof_shap.png", "土壌群別signed OOF SHAP"),
            ("integrated_final_shapley_summary_bg32_perm32_fixed.png", "最終耕作地割合の統合Shapley"),
            ("final_cropland_dominant_factor_map.png", "各グリッドの支配的生産要素"),
        ],
    ),
    (
        "段階別signed SHAPとBeeswarm",
        [
            ("signed_factor_maps_stage1_presence.png", "Stage 1：耕作地存在分類の要素群別signed SHAP"),
            ("signed_factor_maps_stage2_conditional.png", "Stage 2：条件付き耕作地割合の要素群別signed SHAP"),
            ("saved_oof_shap_beeswarm/stage1_presence_oof_shap_beeswarm.png", "Stage 1 OOF SHAP beeswarm"),
            ("saved_oof_shap_beeswarm/stage2_conditional_fraction_oof_shap_beeswarm.png", "Stage 2 OOF SHAP beeswarm"),
            ("saved_oof_shap_beeswarm/integrated_final_fraction_oof_shap_beeswarm.png", "最終耕作地割合の統合SHAP beeswarm"),
        ],
    ),
    (
        "変数値とSHAPの関係",
        [
            ("feature_shap_dependence/presence_classifier_dependence_page_1.png", "Stage 1 dependence plots — page 1"),
            ("feature_shap_dependence/presence_classifier_dependence_page_2.png", "Stage 1 dependence plots — page 2"),
            ("feature_shap_dependence/conditional_fraction_regressor_dependence_page_1.png", "Stage 2 dependence plots — page 1"),
            ("feature_shap_dependence/conditional_fraction_regressor_dependence_page_2.png", "Stage 2 dependence plots — page 2"),
            ("integrated_final_feature_dependence_bg32_perm32/integrated_feature_dependence_old_style.png", "統合個別変数Shapley dependence"),
            ("integrated_final_feature_dependence_bg32_perm32/integrated_soil_group_dependence_old_style.png", "統合土壌群Shapley dependence"),
        ],
    ),
]


def code_cell(source: str):
    return nbformat.v4.new_code_cell(source=source.strip() + "\n")


def markdown_cell(source: str):
    return nbformat.v4.new_markdown_cell(source=source.strip() + "\n")


def repr_rows(rows) -> str:
    return repr(rows)


def build_notebook(variant: dict):
    result_dir = variant["result_dir"]
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)

    table_rows = TABLES + [
        (variant["gdp_comparison"], "GDP追加前後のSpatial OOF比較"),
        (variant["individual_permutation"], "個別変数Permutation importance"),
        (variant["group_permutation"], "要素群Permutation importance"),
    ]
    figure_sections = [
        (
            FIGURE_SECTIONS[0][0],
            FIGURE_SECTIONS[0][1]
            + [(variant["gdp_figure"], "GDP追加前後のSpatial OOF比較")],
        ),
        *FIGURE_SECTIONS[1:],
        (
            "Permutation importance",
            [
                (variant["individual_permutation_figure"], "個別変数Permutation importance"),
                (variant["group_permutation_figure"], "要素群Permutation importance"),
            ],
        ),
    ]

    cells = [
        markdown_cell(
            f"""
# {variant['title']}

**{variant['subtitle']}**

このノートブックは保存済みのOOF・SHAP・Permutation importance結果から再構成した、
実行結果付きの閲覧用ノートブックです。モデル学習やSHAP計算は再実行していません。

- 結果ディレクトリ：`{result_dir}`
- 解析コード本体とは独立した閲覧専用ノートブックです。
"""
        ),
        code_cell(
            f"""
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from IPython.display import Image as IPImage, Markdown, display

RESULT_DIR = Path(r"{result_dir}")
assert RESULT_DIR.is_dir(), RESULT_DIR

def show_table(relative_path, caption, max_rows=50):
    path = RESULT_DIR / relative_path
    display(Markdown(f"### {{caption}}\\n\\n`{{relative_path}}`"))
    if not path.is_file():
        display(Markdown("**ファイルが見つかりません。**"))
        return
    size_mb = path.stat().st_size / 1024**2
    if size_mb > 25:
        frame = pd.read_csv(path, nrows=max_rows)
        note = f"先頭{{len(frame):,}}行のみ表示（ファイルサイズ {{size_mb:.1f}} MB）"
    else:
        frame = pd.read_csv(path)
        note = f"{{len(frame):,}}行 × {{len(frame.columns):,}}列"
        if len(frame) > max_rows:
            frame = frame.head(max_rows)
            note += f"；先頭{{max_rows}}行を表示"
    display(Markdown(note))
    display(frame)

def show_image(relative_path, caption, max_width=1800, max_height=1300):
    path = RESULT_DIR / relative_path
    display(Markdown(f"### {{caption}}\\n\\n`{{relative_path}}`"))
    if not path.is_file():
        display(Markdown("**画像が見つかりません。**"))
        return
    with PILImage.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=6)
    display(IPImage(data=buffer.getvalue(), format="png"))

print("Result directory:", RESULT_DIR)
print("Saved files:", sum(path.is_file() for path in RESULT_DIR.rglob("*")))
"""
        ),
        markdown_cell("## 1. 実行マニフェストと使用変数"),
        code_cell(
            f"""
manifest_path = RESULT_DIR / {variant['manifest']!r}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
display(pd.DataFrame({{
    "item": [
        "target year", "analysis rows", "features", "Spatial OOF folds",
        "final model", "GDP feature", "GDP factor group",
    ],
    "value": [
        manifest["target_year"], f'{{manifest["n_analysis_rows"]:,}}',
        manifest["n_features"], manifest["spatial_oof_folds"],
        manifest["final_model"], manifest["gdp_feature"],
        manifest["gdp_factor_group"],
    ],
}}))
display(Markdown("### 全説明変数"))
display(pd.DataFrame({{
    "number": range(1, len(manifest["features"]) + 1),
    "feature": manifest["features"],
}}))
display(Markdown("### 欠損補完とGDPデータ情報"))
manifest_details = {{key: value for key, value in manifest.items()
                    if key not in {{"features", "global_metrics"}}}}
display(pd.DataFrame(manifest_details.items(), columns=["key", "value"]))
"""
        ),
        markdown_cell("## 2. 精度・SHAP・Permutation importanceの保存表"),
        code_cell(
            f"""
table_specs = {repr_rows(table_rows)}
for relative_path, caption in table_specs:
    show_table(relative_path, caption)
"""
        ),
        markdown_cell("## 3. 大規模セル別結果の内容確認"),
        code_cell(
            f"""
large_data_specs = {repr_rows(LARGE_DATA_SAMPLES)}
for relative_path, caption in large_data_specs:
    show_table(relative_path, caption, max_rows=8)
"""
        ),
    ]

    for section_title, figures in figure_sections:
        cells.append(markdown_cell(f"## {section_title}"))
        cells.append(
            code_cell(
                f"""
figure_specs = {repr_rows(figures)}
for relative_path, caption in figure_specs:
    show_image(relative_path, caption)
"""
            )
        )

    cells.extend(
        [
            markdown_cell("## 保存結果ファイルの完全な索引"),
            code_cell(
                """
inventory = []
for path in sorted(RESULT_DIR.rglob("*")):
    if not path.is_file():
        continue
    inventory.append({
        "relative_path": str(path.relative_to(RESULT_DIR)),
        "type": "".join(path.suffixes) or "file",
        "size_MB": path.stat().st_size / 1024**2,
    })
inventory_df = pd.DataFrame(inventory)
display(Markdown(
    f"**全{len(inventory_df):,}ファイル、合計"
    f"{inventory_df['size_MB'].sum():,.1f} MB**"
))
display(inventory_df)
"""
            ),
            markdown_cell(
                """
## 注意

この閲覧用ノートブックの出力は、保存済み結果を読み込んで再表示したものです。
新しいモデルの学習、Spatial OOF、SHAP、Permutation importanceの再計算は行っていません。
元の高解像度PNG・完全なCSV/Parquetは、各セルに記載した結果ディレクトリに残っています。
"""
            ),
        ]
    )

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (research)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "analysis_type": "reconstructed_saved_results",
            "source_result_directory": str(result_dir),
        },
    )
    return notebook


def main():
    for variant in VARIANTS:
        notebook = build_notebook(variant)
        destination = variant["destination"]
        nbformat.write(notebook, destination)
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        executed = client.execute()
        nbformat.write(executed, destination)
        print("Created and executed:", destination)
        print("Size MB:", round(destination.stat().st_size / 1024**2, 1))


if __name__ == "__main__":
    main()
