from pathlib import Path
script_path = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\download_crop_standardization_data.py")
text = script_path.read_text(encoding="utf-8")
text = text.replace('''    usecols = [\n        "Domain Code", "Domain", "Area Code", "Area", "Element Code", "Element",\n        "Item Code", "Item", "Year", "Months", "Unit", "Value",\n    ]\n''', '''    usecols = [\n        "Area Code", "Area", "Element Code", "Element",\n        "Item Code", "Item", "Year", "Months", "Unit", "Value",\n    ]\n''')
script_path.write_text(text, encoding='utf-8')
print('patched usecols')
