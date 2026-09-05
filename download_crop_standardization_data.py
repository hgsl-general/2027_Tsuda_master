from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\masterresearch\Comparative_advantage\GAEZ")
OUT_DIR = BASE_DIR / "StandardizationData"
RAW_DIR = OUT_DIR / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

FAOSTAT_BULK = "https://fenixservices.fao.org/faostat/static/bulkdownloads"
PRICE_URL = f"{FAOSTAT_BULK}/Prices_E_All_Data_(Normalized).zip"
FBS_URL = f"{FAOSTAT_BULK}/FoodBalanceSheets_E_All_Data_(Normalized).zip"
README_RES02 = BASE_DIR / "_readme_res02.xlsx"

PRICE_ZIP = RAW_DIR / "Prices_E_All_Data_(Normalized).zip"
FBS_ZIP = RAW_DIR / "FoodBalanceSheets_E_All_Data_(Normalized).zip"

PRICE_SUMMARY = OUT_DIR / "faostat_producer_prices_recent_item_median_usd_per_tonne.csv"
CALORIE_SUMMARY = OUT_DIR / "faostat_food_balance_kcal_per_kg_recent.csv"
MATCH_DRAFT = OUT_DIR / "gaez_crop_standardization_lookup_draft.csv"
README_MD = OUT_DIR / "README_standardization_data.md"
METADATA_JSON = OUT_DIR / "standardization_data_metadata.json"


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"[download] skip existing: {path.name} ({path.stat().st_size / 1024**2:.1f} MB)")
        return
    print(f"[download] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    tmp.replace(path)
    print(f"[download] saved: {path} ({path.stat().st_size / 1024**2:.1f} MB)")


def find_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not csv_names:
        raise RuntimeError(f"No CSV found in {zip_path}")
    main_names = [name for name in csv_names if "_All_Data" in name]
    if main_names:
        return sorted(main_names)[0]
    return sorted(csv_names, key=lambda name: ("flag" in name.lower(), "element" in name.lower(), "item" in name.lower(), len(name)))[0]


def read_faostat_zip(zip_path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    csv_name = find_csv_in_zip(zip_path)
    print(f"[read] {zip_path.name} :: {csv_name}")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(csv_name) as handle:
            return pd.read_csv(handle, encoding="utf-8", usecols=usecols, low_memory=False)


def normalize_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    replacements = {
        "wetland rice": "rice paddy",
        "dryland rice": "rice paddy",
        "rice wetland": "rice paddy",
        "rice dryland": "rice paddy",
        "maize": "maize corn",
        "soybean": "soybeans",
        "soy bean": "soybeans",
        "white potato": "potatoes",
        "sweet potato": "sweet potatoes",
        "sugar beet": "sugar beet",
        "sugar cane": "sugar cane",
        "rapeseed": "rape mustard seed",
        "rape": "rape mustard seed",
        "groundnut": "groundnuts",
        "pearl millet": "millet",
        "proso millet": "millet",
        "finger millet": "millet",
        "sorghum": "sorghum",
        "barley": "barley",
        "wheat": "wheat",
        "cassava": "cassava",
        "beans": "beans dry",
        "bean": "beans dry",
        "sunflower": "sunflower seed",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_match_one(query: str, candidates: pd.DataFrame, name_col: str) -> tuple[object, str, float]:
    q = normalize_text(query)
    best_idx = None
    best_name = ""
    best_score = -1.0
    for idx, name in candidates[name_col].dropna().astype(str).items():
        c = normalize_text(name)
        score = SequenceMatcher(None, q, c).ratio()
        q_tokens = set(q.split())
        c_tokens = set(c.split())
        if q_tokens and c_tokens:
            score = 0.65 * score + 0.35 * (len(q_tokens & c_tokens) / len(q_tokens | c_tokens))
        if score > best_score:
            best_idx = idx
            best_name = name
            best_score = score
    return best_idx, best_name, best_score


def build_price_summary() -> pd.DataFrame:
    usecols = [
        "Area Code", "Area", "Element Code", "Element",
        "Item Code", "Item", "Year", "Months", "Unit", "Value",
    ]
    prices = read_faostat_zip(PRICE_ZIP, usecols=usecols)
    prices = prices[prices["Element"].astype(str).eq("Producer Price (USD/tonne)")].copy()
    if "Months" in prices.columns:
        prices = prices[prices["Months"].astype(str).eq("Annual value")].copy()
    prices["Year"] = pd.to_numeric(prices["Year"], errors="coerce")
    prices["Value"] = pd.to_numeric(prices["Value"], errors="coerce")
    prices = prices.dropna(subset=["Year", "Value"])
    prices = prices[prices["Value"] > 0]
    max_year = int(prices["Year"].max())
    min_year = max_year - 4
    recent = prices[prices["Year"].between(min_year, max_year)].copy()
    grouped = (
        recent.groupby(["Item Code", "Item"], as_index=False)
        .agg(
            price_usd_per_tonne_median=("Value", "median"),
            price_usd_per_tonne_mean=("Value", "mean"),
            price_observations=("Value", "size"),
            price_country_count=("Area", "nunique"),
            price_min_year=("Year", "min"),
            price_max_year=("Year", "max"),
        )
        .sort_values("Item")
        .reset_index(drop=True)
    )
    grouped.insert(0, "source", "FAOSTAT Prices_E_All_Data_(Normalized)")
    grouped.insert(1, "aggregation", f"country-year median, {min_year}-{max_year}")
    grouped.to_csv(PRICE_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[write] {PRICE_SUMMARY} rows={len(grouped)}")
    return grouped


def build_calorie_summary() -> pd.DataFrame:
    fbs_usecols = ["Area", "Element", "Item Code", "Item", "Year", "Unit", "Value"]
    fbs = read_faostat_zip(FBS_ZIP, usecols=fbs_usecols)
    required = {"Area", "Element", "Item Code", "Item", "Year", "Unit", "Value"}
    missing = required - set(fbs.columns)
    if missing:
        raise RuntimeError(f"FBS missing columns: {missing}; columns={list(fbs.columns)}")
    fbs["Year"] = pd.to_numeric(fbs["Year"], errors="coerce")
    fbs["Value"] = pd.to_numeric(fbs["Value"], errors="coerce")
    area_values = set(fbs["Area"].dropna().astype(str).unique())
    area_choice = "World" if "World" in area_values else None
    if area_choice is None:
        # fallback: aggregate medians across countries after item-year calculation
        fbs_area = fbs.copy()
        area_note = "all areas median"
    else:
        fbs_area = fbs[fbs["Area"].astype(str).eq(area_choice)].copy()
        area_note = "World"

    element_values = fbs_area["Element"].dropna().astype(str).unique().tolist()
    qty_candidates = [e for e in element_values if "Food supply quantity" in e]
    kcal_candidates = [e for e in element_values if "Food supply (kcal" in e]
    if not qty_candidates or not kcal_candidates:
        raise RuntimeError(f"Could not find FBS quantity/kcal elements. Elements sample={element_values[:50]}")
    qty_element = qty_candidates[0]
    kcal_element = kcal_candidates[0]
    subset = fbs_area[fbs_area["Element"].isin([qty_element, kcal_element])].copy()
    max_year = int(subset["Year"].max())
    min_year = max_year - 4
    subset = subset[subset["Year"].between(min_year, max_year)]

    if area_choice == "World":
        index_cols = ["Item Code", "Item", "Year"]
    else:
        index_cols = ["Area", "Item Code", "Item", "Year"]

    pivot = subset.pivot_table(index=index_cols, columns="Element", values="Value", aggfunc="mean").reset_index()
    pivot = pivot.dropna(subset=[qty_element, kcal_element])
    pivot = pivot[pivot[qty_element] > 0]
    pivot["kcal_per_kg"] = pivot[kcal_element] * 365.0 / pivot[qty_element]
    pivot = pivot[np.isfinite(pivot["kcal_per_kg"])]
    pivot = pivot[(pivot["kcal_per_kg"] > 0) & (pivot["kcal_per_kg"] < 10000)]

    grouped = (
        pivot.groupby(["Item Code", "Item"], as_index=False)
        .agg(
            kcal_per_kg_median=("kcal_per_kg", "median"),
            kcal_per_kg_mean=("kcal_per_kg", "mean"),
            calorie_observations=("kcal_per_kg", "size"),
            calorie_min_year=("Year", "min"),
            calorie_max_year=("Year", "max"),
        )
        .sort_values("Item")
        .reset_index(drop=True)
    )
    grouped.insert(0, "source", "FAOSTAT FoodBalanceSheets_E_All_Data_(Normalized)")
    grouped.insert(1, "aggregation", f"{area_note}; median of kcal/cap/day * 365 / kg/cap/year, {min_year}-{max_year}")
    grouped.insert(2, "quantity_element", qty_element)
    grouped.insert(3, "kcal_element", kcal_element)
    grouped.to_csv(CALORIE_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[write] {CALORIE_SUMMARY} rows={len(grouped)}")
    return grouped


def build_match_draft(price_summary: pd.DataFrame, calorie_summary: pd.DataFrame) -> pd.DataFrame:
    crops = pd.read_excel(README_RES02, sheet_name="CODES_CROP")
    crops = crops[crops["UPLOADED IN THE GOOGLE CLOUD STORAGE"].astype(str).str.upper().eq("YES")]
    crops = crops[["CODE", "Crop name", "Crop group"]].drop_duplicates().sort_values("CODE").reset_index(drop=True)

    price_candidates = price_summary[["Item Code", "Item", "price_usd_per_tonne_median", "price_observations", "price_country_count"]].copy()
    calorie_candidates = calorie_summary[["Item Code", "Item", "kcal_per_kg_median", "calorie_observations"]].copy()

    records = []
    for _, crop in crops.iterrows():
        price_idx, price_item, price_score = fuzzy_match_one(crop["Crop name"], price_candidates, "Item")
        calorie_idx, calorie_item, calorie_score = fuzzy_match_one(crop["Crop name"], calorie_candidates, "Item")
        price_row = price_candidates.loc[price_idx] if price_idx is not None else pd.Series(dtype=object)
        calorie_row = calorie_candidates.loc[calorie_idx] if calorie_idx is not None else pd.Series(dtype=object)
        records.append({
            "gaez_code": crop["CODE"],
            "gaez_crop_name": crop["Crop name"],
            "gaez_crop_group": crop["Crop group"],
            "suggested_price_item_code": price_row.get("Item Code", np.nan),
            "suggested_price_item": price_item,
            "price_match_score": round(float(price_score), 3),
            "price_usd_per_tonne_median": price_row.get("price_usd_per_tonne_median", np.nan),
            "price_observations": price_row.get("price_observations", np.nan),
            "price_country_count": price_row.get("price_country_count", np.nan),
            "suggested_calorie_item_code": calorie_row.get("Item Code", np.nan),
            "suggested_calorie_item": calorie_item,
            "calorie_match_score": round(float(calorie_score), 3),
            "kcal_per_kg_median": calorie_row.get("kcal_per_kg_median", np.nan),
            "calorie_observations": calorie_row.get("calorie_observations", np.nan),
            "needs_manual_check": bool(price_score < 0.72 or calorie_score < 0.72),
        })
    draft = pd.DataFrame(records)
    draft.to_csv(MATCH_DRAFT, index=False, encoding="utf-8-sig")
    print(f"[write] {MATCH_DRAFT} rows={len(draft)}; manual_check={int(draft['needs_manual_check'].sum())}")
    return draft


def write_docs(price_summary: pd.DataFrame, calorie_summary: pd.DataFrame, draft: pd.DataFrame) -> None:
    metadata = {
        "created_files": {
            "price_summary": str(PRICE_SUMMARY),
            "calorie_summary": str(CALORIE_SUMMARY),
            "gaez_match_draft": str(MATCH_DRAFT),
            "readme": str(README_MD),
        },
        "source_urls": {
            "faostat_prices_bulk": PRICE_URL,
            "faostat_food_balance_sheets_bulk": FBS_URL,
            "faostat_home": "https://www.fao.org/faostat/",
        },
        "notes": [
            "Price table is a recent country-year median of FAOSTAT producer prices in USD/tonne.",
            "Calorie table converts Food Balance Sheet kcal/capita/day and kg/capita/year to kcal/kg.",
            "GAEZ matching file is an automatic fuzzy-match draft and should be manually checked before analysis.",
        ],
        "row_counts": {
            "price_items": int(len(price_summary)),
            "calorie_items": int(len(calorie_summary)),
            "gaez_crop_matches": int(len(draft)),
            "matches_needing_manual_check": int(draft["needs_manual_check"].sum()),
        },
    }
    METADATA_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Crop standardization data

This folder contains FAOSTAT-derived tables for converting GAEZ climate potential yield into value- or calorie-based suitability proxies.

## Files

- `faostat_producer_prices_recent_item_median_usd_per_tonne.csv`  
  FAOSTAT Producer Prices. Recent country-year median producer price by item, in USD/tonne.

- `faostat_food_balance_kcal_per_kg_recent.csv`  
  FAOSTAT Food Balance Sheets. Item-level kcal/kg inferred from `Food supply (kcal/capita/day) * 365 / Food supply quantity (kg/capita/year)`.

- `gaez_crop_standardization_lookup_draft.csv`  
  Automatic fuzzy matching from GAEZ RES02 crop codes to FAOSTAT price and calorie items. This is a draft; manually check low-score matches before using.

## Recommended use

For each crop `c` and grid cell:

```text
value_potential_c = yield_t_per_ha_c * price_usd_per_tonne_c
calorie_potential_c = yield_t_per_ha_c * 1000 * kcal_per_kg_c
```

Then aggregate across crops using `max`, `top3 mean`, or `top5 mean`.

## Sources

- FAOSTAT bulk downloads: https://www.fao.org/faostat/
- Prices bulk ZIP: {PRICE_URL}
- Food Balance Sheets bulk ZIP: {FBS_URL}

## Caution

- Producer prices are country-year observations. The summary uses median values, not global market prices.
- Food Balance Sheet kcal/kg is an item-level food-use conversion. It may not match raw harvested crop mass for processing crops such as sugar cane, oil palm, cotton, tea, coffee, or fodder crops.
- The GAEZ matching file is intentionally marked as a draft. Use only checked crop mappings for regressions.
"""
    README_MD.write_text(readme, encoding="utf-8")
    print(f"[write] {README_MD}")
    print(f"[write] {METADATA_JSON}")


def main() -> None:
    download(PRICE_URL, PRICE_ZIP)
    download(FBS_URL, FBS_ZIP)
    price_summary = build_price_summary()
    calorie_summary = build_calorie_summary()
    draft = build_match_draft(price_summary, calorie_summary)
    write_docs(price_summary, calorie_summary, draft)

    print("\nDone.")
    print(f"Output directory: {OUT_DIR}")
    print("Key samples:")
    for name in ["Wheat", "Maize", "Rice", "Soybeans", "Potatoes", "Sugar cane"]:
        p = price_summary[price_summary["Item"].astype(str).str.contains(name, case=False, regex=False, na=False)].head(3)
        if not p.empty:
            print("\nPRICE", name)
            print(p[["Item Code", "Item", "price_usd_per_tonne_median", "price_observations"]].to_string(index=False))
        c = calorie_summary[calorie_summary["Item"].astype(str).str.contains(name, case=False, regex=False, na=False)].head(3)
        if not c.empty:
            print("CALORIE", name)
            print(c[["Item Code", "Item", "kcal_per_kg_median", "calorie_observations"]].to_string(index=False))


if __name__ == "__main__":
    main()
