from pathlib import Path
path = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\build_and_run_cropland_lgbm_notebook.py")
text = path.read_text(encoding="utf-8")
text = text.replace('out = np.asarray(array, dtype=np.float32)\n    for nodata in nodata_values:', 'out = np.array(array, dtype=np.float32, copy=True)\n    for nodata in nodata_values:')
path.write_text(text, encoding="utf-8")
print('fixed clean_float copy')
