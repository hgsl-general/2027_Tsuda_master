from pathlib import Path


PATH = Path("/work/tsuda/GAEZ/build_groundwater_gdp_oof_shap_notebooks.py")
text = PATH.read_text(encoding="utf-8")
old = '''    source = replace_once(
        source,
        "    WATERGAP_RECHARGE_PATH,\\n]",
        "    WATERGAP_RECHARGE_PATH,\\n    GDP_PC_2020_PATH,\\n]",
        "GDP required path",
    )
'''
new = '''    source = replace_once(
        source,
        "    WATERGAP_RECHARGE_PATH,\\n",
        "    WATERGAP_RECHARGE_PATH,\\n    GDP_PC_2020_PATH,\\n",
        "GDP required path",
    )
'''
if text.count(old) != 1:
    raise RuntimeError(f"Expected one required-path block, found {text.count(old)}")
text = text.replace(old, new, 1)
compile(text, str(PATH), "exec")
PATH.write_text(text, encoding="utf-8")
print("Updated:", PATH)
