from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path("/work/tsuda/GAEZ")
TEMPLATE_NOTEBOOK = (
    ROOT / "groundwater_gdp_dominant_factor_map_high_contrast_executed.ipynb"
)
DESTINATION = (
    ROOT
    / "rural_population_gdp_dominant_factor_map_high_contrast_executed.ipynb"
)

OLD_RESULT_DIR = (
    ROOT
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_groundwater_gdp_target2020"
)
NEW_RESULT_DIR = (
    ROOT
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_groundwater_rural_population_gdp_target2020"
)


notebook = nbformat.read(TEMPLATE_NOTEBOOK, as_version=4)
code_cell = next(cell for cell in notebook.cells if cell.cell_type == "code")
source = code_cell.source

source = source.replace(str(OLD_RESULT_DIR), str(NEW_RESULT_DIR))
source = source.replace(
    '    "Land and soil capital": "#009E73",\n'
    '    "Economic prosperity": "#6A0572",',
    '    "Land and soil capital": "#009E73",\n'
    '    "Rural population": "#C62828",\n'
    '    "Economic prosperity": "#6A0572",',
)
source = source.replace(
    '    "Land and soil capital",\n'
    '    "Economic prosperity",',
    '    "Land and soil capital",\n'
    '    "Rural population",\n'
    '    "Economic prosperity",',
)

required = [
    str(NEW_RESULT_DIR),
    '"Rural population": "#C62828"',
    '"Economic prosperity": "#6A0572"',
    '"Market access and infrastructure": "#F4A261"',
]
for token in required:
    if token not in source:
        raise RuntimeError(f"Generated code is missing: {token}")
if str(OLD_RESULT_DIR) in source:
    raise RuntimeError("Old groundwater-only result directory remains")

code_cell.source = source
code_cell.execution_count = None
code_cell.outputs = []
notebook.cells[0].source = """
# High-contrast dominant-factor map — rural population version

地下水2変数＋非都市人口＋一人当たりGDPモデルについて、保存済み全240,000セルの
統合グループShapleyから、最大寄与要素とその符号を高コントラスト配色で再描画します。

全要素を同じ大きさの四角で表示します。モデル学習・Spatial OOF・SHAP計算は再実行しません。
""".strip()
notebook.metadata["source_result_directory"] = str(NEW_RESULT_DIR)
notebook.metadata["analysis_type"] = (
    "high-contrast dominant-factor visualization with rural population"
)

nbformat.write(notebook, DESTINATION)
executed = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute()
nbformat.write(executed, DESTINATION)
print("Created and executed:", DESTINATION)
print("Size MB:", round(DESTINATION.stat().st_size / 1024**2, 2))
