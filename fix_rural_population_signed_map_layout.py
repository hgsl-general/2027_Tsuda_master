from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path(
    "/work/tsuda/GAEZ/"
    "cropland_soil_group_wetland_excluded_groundwater_"
    "rural_population_oof_shap_target2020.ipynb"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
matches = [
    cell
    for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
    and "def plot_signed_factor_maps(" in "".join(cell.get("source", []))
    and "signed_factor_maps_stage1_presence.png" in "".join(cell.get("source", []))
]
if len(matches) != 1:
    raise RuntimeError(f"Expected one signed-map cell, found {len(matches)}")

cell = matches[0]
source = "".join(cell["source"])
source = replace_once(
    source,
    "1つの段階について、3生産要素のsigned group SHAPを",
    "1つの段階について、4生産要素のsigned group SHAPを",
    "docstring",
)
source = replace_once(
    source,
    "# 3要素をまとめた上で共通カラースケールを計算",
    "# 4要素をまとめた上で共通カラースケールを計算",
    "scale comment",
)
source = replace_once(
    source,
    '''    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 7),
        sharex=True,
        sharey=True,
    )

    for ax, factor_name in zip(''',
    '''    fig, axes = plt.subplots(
        2,
        2,
        figsize=(20, 12),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, factor_name in zip(''',
    "subplot layout",
)
source = replace_once(
    source,
    "# 3パネル共通カラーバー",
    "# 4パネル共通カラーバー",
    "colorbar comment",
)
source = replace_once(source, "axes[2].legend(", "axes[-1].legend(", "legend axis")
source = replace_once(
    source,
    '''    fig.subplots_adjust(
        top=0.78,
        bottom=0.20,
        left=0.05,
        right=0.98,
        wspace=0.08,
    )''',
    '''    fig.subplots_adjust(
        top=0.84,
        bottom=0.14,
        left=0.05,
        right=0.98,
        wspace=0.08,
        hspace=0.22,
    )''',
    "subplot spacing",
)

ast.parse(source, filename="signed_factor_map_cell")
cell["source"] = source.splitlines(keepends=True)
NOTEBOOK.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(f"Updated: {NOTEBOOK}")
