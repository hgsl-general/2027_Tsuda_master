from pathlib import Path
script_path = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\download_crop_standardization_data.py")
text = script_path.read_text(encoding="utf-8")
old = '''def find_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not csv_names:
        raise RuntimeError(f"No CSV found in {zip_path}")
    # Prefer normalized data over flags/metadata if multiple csv files exist.
    csv_names = sorted(csv_names, key=lambda name: ("flag" in name.lower(), len(name)))
    return csv_names[0]
'''
new = '''def find_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not csv_names:
        raise RuntimeError(f"No CSV found in {zip_path}")
    main_names = [name for name in csv_names if "_All_Data" in name]
    if main_names:
        return sorted(main_names)[0]
    return sorted(csv_names, key=lambda name: ("flag" in name.lower(), "element" in name.lower(), "item" in name.lower(), len(name)))[0]
'''
if old not in text:
    raise RuntimeError('target block not found')
text = text.replace(old, new)
old2 = '''def build_calorie_summary() -> pd.DataFrame:
    fbs = read_faostat_zip(FBS_ZIP)
    required = {"Area", "Element", "Item Code", "Item", "Year", "Unit", "Value"}
'''
new2 = '''def build_calorie_summary() -> pd.DataFrame:
    fbs_usecols = ["Area", "Element", "Item Code", "Item", "Year", "Unit", "Value"]
    fbs = read_faostat_zip(FBS_ZIP, usecols=fbs_usecols)
    required = {"Area", "Element", "Item Code", "Item", "Year", "Unit", "Value"}
'''
if old2 not in text:
    raise RuntimeError('calorie block not found')
text = text.replace(old2, new2)
script_path.write_text(text, encoding='utf-8')
print('fixed', script_path)
