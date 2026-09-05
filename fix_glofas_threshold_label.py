from pathlib import Path

script_path = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\glofas_reliable_river_pipeline.py")
text = script_path.read_text(encoding="utf-8")
old = '''def threshold_label(threshold: float) -> str:
    return str(threshold).replace(".", "p").rstrip("0").rstrip("p")
'''
new = '''def threshold_label(threshold: float) -> str:
    threshold_float = float(threshold)
    if threshold_float.is_integer():
        return str(int(threshold_float))
    return ("%g" % threshold_float).replace(".", "p")
'''
if old not in text:
    raise RuntimeError("Original threshold_label function not found")
script_path.write_text(text.replace(old, new), encoding="utf-8")
print("Fixed threshold_label in", script_path)
