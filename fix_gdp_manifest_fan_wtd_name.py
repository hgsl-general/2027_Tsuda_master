import json
from pathlib import Path


ROOT = Path("/work/tsuda/GAEZ")
BUILDER = ROOT / "build_groundwater_gdp_oof_shap_notebooks.py"
NOTEBOOKS = [
    ROOT / "cropland_soil_group_wetland_excluded_groundwater_gdp_oof_shap_target2020.ipynb",
    ROOT / "cropland_soil_group_wetland_excluded_groundwater_rural_population_gdp_oof_shap_target2020.ipynb",
]

OLD = 'float(fan_wtd_imputation_median_m)'
NEW = 'float(fan_wtd_imputation_m)'


builder_source = BUILDER.read_text(encoding="utf-8")
builder_count = builder_source.count(OLD)
if builder_count != 1:
    raise RuntimeError(f"Expected one builder occurrence; found {builder_count}")
BUILDER.write_text(builder_source.replace(OLD, NEW), encoding="utf-8")
print("Fixed builder:", BUILDER)


for notebook_path in NOTEBOOKS:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    replacements = 0
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        updated = []
        for line in source:
            count = line.count(OLD)
            replacements += count
            updated.append(line.replace(OLD, NEW))
        cell["source"] = updated
    if replacements != 1:
        raise RuntimeError(
            f"Expected one notebook occurrence in {notebook_path.name}; "
            f"found {replacements}"
        )
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("Fixed notebook:", notebook_path)
