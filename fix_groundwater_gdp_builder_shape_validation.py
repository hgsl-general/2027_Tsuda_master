from pathlib import Path


PATH = Path("/work/tsuda/GAEZ/build_groundwater_gdp_oof_shap_notebooks.py")
text = PATH.read_text(encoding="utf-8")
old = '''    source = replace_once(
        source,
        '    "fan_water_table_depth_m": fan_water_table_depth_m,\\n}',
        '    "fan_water_table_depth_m": fan_water_table_depth_m,\\n'
        '    "gdp_pc_2020_ppp2017usd": gdp_pc_2020_ppp2017usd,\\n}',
        "GDP shape validation",
    )
'''
new = '''    source = replace_once(
        source,
        '    "fan_water_table_depth_m": fan_water_table_depth_m,\\n',
        '    "fan_water_table_depth_m": fan_water_table_depth_m,\\n'
        '    "gdp_pc_2020_ppp2017usd": gdp_pc_2020_ppp2017usd,\\n',
        "GDP shape validation",
    )
'''
if text.count(old) != 1:
    raise RuntimeError(f"Expected one shape-validation block, found {text.count(old)}")
text = text.replace(old, new, 1)
compile(text, str(PATH), "exec")
PATH.write_text(text, encoding="utf-8")
print("Updated:", PATH)
