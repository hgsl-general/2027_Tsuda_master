from pathlib import Path
import pandas as pd
import numpy as np

base = Path(r"C:\masterresearch\Comparative_advantage\GAEZ")
out = base / "StandardizationData"
price = pd.read_csv(out / "faostat_producer_prices_recent_item_median_usd_per_tonne.csv")
cal = pd.read_csv(out / "faostat_food_balance_kcal_per_kg_recent.csv")
gaez = pd.read_excel(base / "_readme_res02.xlsx", sheet_name="CODES_CROP")
gaez = gaez[gaez["UPLOADED IN THE GOOGLE CLOUD STORAGE"].astype(str).str.upper().eq("YES")][["CODE", "Crop name", "Crop group"]]

mapping = {
    # Wheat variants
    "WHEA": ("Wheat", "Wheat and products"),
    "HWHE": ("Wheat", "Wheat and products"),
    "SWHE": ("Wheat", "Wheat and products"),
    "TWHE": ("Wheat", "Wheat and products"),
    # Maize variants
    "MAIZ": ("Maize (corn)", "Maize and products"),
    "HMZE": ("Maize (corn)", "Maize and products"),
    "LMZE": ("Maize (corn)", "Maize and products"),
    "TMZE": ("Maize (corn)", "Maize and products"),
    "MZSI": ("Maize (corn)", "Maize and products"),
    # Rice variants
    "RICW": ("Rice", "Rice and products"),
    "RICD": ("Rice", "Rice and products"),
    # Cereals
    "BARL": ("Barley", "Barley and products"),
    "HBRL": ("Barley", "Barley and products"),
    "SBRL": ("Barley", "Barley and products"),
    "TBRL": ("Barley", "Barley and products"),
    "SORG": ("Sorghum", "Sorghum and products"),
    "BSRG": ("Sorghum", "Sorghum and products"),
    "HSRG": ("Sorghum", "Sorghum and products"),
    "LSRG": ("Sorghum", "Sorghum and products"),
    "TSRG": ("Sorghum", "Sorghum and products"),
    "MLLT": ("Millet", "Millet and products"),
    "FIMLT": ("Millet", "Millet and products"),
    "FMLT": ("Millet", "Millet and products"),
    "PMLT": ("Millet", "Millet and products"),
    # Oil crops and pulses
    "SOYB": ("Soya beans", "Soyabeans"),
    "RAPE": ("Rape or colza seed", "Rape and Mustardseed"),
    "SRAP": ("Rape or colza seed", "Rape and Mustardseed"),
    "WRAP": ("Rape or colza seed", "Rape and Mustardseed"),
    "SUNF": ("Sunflower seed", "Sunflower seed"),
    "GRND": ("Groundnuts, excluding shelled", "Groundnuts"),
    "BEAN": ("Beans, dry", "Beans"),
    # Roots and sugar crops
    "WPOT": ("Potatoes", "Potatoes and products"),
    "SPOT": ("Sweet potatoes", "Sweet potatoes"),
    "CASV": ("Cassava, fresh", "Cassava and products"),
    "SUGB": ("Sugar beet", "Sugar beet"),
    "SUGC": ("Sugar cane", "Sugar cane"),
}

def one_row(df, item):
    hit = df[df["Item"].astype(str).eq(item)]
    if hit.empty:
        raise KeyError(item)
    return hit.iloc[0]

records = []
for code, (price_item, calorie_item) in mapping.items():
    crop = gaez[gaez["CODE"].eq(code)]
    if crop.empty:
        continue
    crop = crop.iloc[0]
    p = one_row(price, price_item)
    c = one_row(cal, calorie_item)
    records.append({
        "gaez_code": code,
        "gaez_crop_name": crop["Crop name"],
        "gaez_crop_group": crop["Crop group"],
        "checked_mapping": True,
        "faostat_price_item_code": p["Item Code"],
        "faostat_price_item": p["Item"],
        "price_usd_per_tonne_median": p["price_usd_per_tonne_median"],
        "price_observations": p["price_observations"],
        "price_country_count": p["price_country_count"],
        "faostat_calorie_item_code": c["Item Code"],
        "faostat_calorie_item": c["Item"],
        "kcal_per_kg_median": c["kcal_per_kg_median"],
        "calorie_observations": c["calorie_observations"],
        "note": "Representative crop checked mapping; verify processing crops before calorie use" if code in {"SUGB", "SUGC", "RAPE", "SUNF", "GRND"} else "Representative crop checked mapping",
    })

checked = pd.DataFrame(records).sort_values(["gaez_crop_group", "gaez_code"]).reset_index(drop=True)
path = out / "gaez_representative_crop_standardization_checked.csv"
checked.to_csv(path, index=False, encoding="utf-8-sig")
print(path)
print('rows=', len(checked))
print(checked[["gaez_code", "gaez_crop_name", "faostat_price_item", "price_usd_per_tonne_median", "faostat_calorie_item", "kcal_per_kg_median"]].to_string(index=False))
