import json
from pathlib import Path


ROOT = Path("/work/tsuda/GAEZ")
BUILDER = ROOT / "build_groundwater_gdp_oof_shap_notebooks.py"
NOTEBOOKS = [
    ROOT / "cropland_soil_group_wetland_excluded_groundwater_gdp_oof_shap_target2020.ipynb",
    ROOT / "cropland_soil_group_wetland_excluded_groundwater_rural_population_gdp_oof_shap_target2020.ipynb",
]
BAD_LINE = 'group_colors.setdefault("Economic prosperity", "#31A354")\n'


builder_source = BUILDER.read_text(encoding="utf-8")
old = """        elif name in color_dicts:
            statements.append(
                f'{name}.setdefault(\"{GDP_GROUP}\", \"{GDP_COLOR}\")'
            )
"""
new = """        elif name in color_dicts and (
            name != \"group_colors\"
            or isinstance(getattr(node, \"value\", None), ast.Dict)
        ):
            statements.append(
                f'{name}.setdefault(\"{GDP_GROUP}\", \"{GDP_COLOR}\")'
            )
"""
if old not in builder_source:
    if new not in builder_source:
        raise RuntimeError("Expected builder block was not found")
else:
    BUILDER.write_text(builder_source.replace(old, new, 1), encoding="utf-8")


for notebook_path in NOTEBOOKS:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    removed = 0
    retained_dict_insertions = 0
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        if BAD_LINE not in source:
            continue
        joined = "".join(source)
        list_assignment = "group_colors = [\n" in joined
        dict_assignment = "group_colors = {\n" in joined
        if list_assignment:
            cell["source"] = [line for line in source if line != BAD_LINE]
            removed += source.count(BAD_LINE)
        if dict_assignment:
            retained_dict_insertions += source.count(BAD_LINE)
    if removed != 2:
        raise RuntimeError(
            f"Expected two invalid list insertions in {notebook_path.name}; found {removed}"
        )
    if retained_dict_insertions != 1:
        raise RuntimeError(
            f"Expected one valid dict insertion in {notebook_path.name}; found "
            f"{retained_dict_insertions}"
        )
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("Fixed:", notebook_path)
    print("Removed invalid list insertions:", removed)
    print("Retained valid dictionary insertion:", retained_dict_insertions)

print("Fixed builder:", BUILDER)
