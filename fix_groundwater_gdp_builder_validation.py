from pathlib import Path


PATH = Path("/work/tsuda/GAEZ/build_groundwater_gdp_oof_shap_notebooks.py")
text = PATH.read_text(encoding="utf-8")
old = '''    if variant["old_soil_model"] in combined:
        raise RuntimeError(
            f"Old model name remains in generated code: {variant['old_soil_model']}"
        )
'''
new = '''    old_model_literal = f'"{variant["old_soil_model"]}"'
    if old_model_literal in combined:
        raise RuntimeError(
            f"Old model literal remains in generated code: {old_model_literal}"
        )
'''
if text.count(old) != 1:
    raise RuntimeError(f"Expected one validation block, found {text.count(old)}")
text = text.replace(old, new, 1)
compile(text, str(PATH), "exec")
PATH.write_text(text, encoding="utf-8")
print("Updated:", PATH)
