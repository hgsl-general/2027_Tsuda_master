from pathlib import Path
readme = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\StandardizationData\README_standardization_data.md")
text = readme.read_text(encoding="utf-8")
insert = """
- `gaez_representative_crop_standardization_checked.csv`  
  Manually checked mapping for the main representative GAEZ crop codes. Prefer this file over the fuzzy draft for first regressions.
"""
if "gaez_representative_crop_standardization_checked.csv" not in text:
    text = text.replace("- `gaez_crop_standardization_lookup_draft.csv`  \n  Automatic fuzzy matching from GAEZ RES02 crop codes to FAOSTAT price and calorie items. This is a draft; manually check low-score matches before using.\n", "- `gaez_crop_standardization_lookup_draft.csv`  \n  Automatic fuzzy matching from GAEZ RES02 crop codes to FAOSTAT price and calorie items. This is a draft; manually check low-score matches before using.\n" + insert)
readme.write_text(text, encoding='utf-8')
print('updated', readme)
