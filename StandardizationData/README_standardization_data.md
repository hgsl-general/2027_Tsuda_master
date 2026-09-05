# Crop standardization data

This folder contains FAOSTAT-derived tables for converting GAEZ climate potential yield into value- or calorie-based suitability proxies.

## Files

- `faostat_producer_prices_recent_item_median_usd_per_tonne.csv`  
  FAOSTAT Producer Prices. Recent country-year median producer price by item, in USD/tonne.

- `faostat_food_balance_kcal_per_kg_recent.csv`  
  FAOSTAT Food Balance Sheets. Item-level kcal/kg inferred from `Food supply (kcal/capita/day) * 365 / Food supply quantity (kg/capita/year)`.

- `gaez_crop_standardization_lookup_draft.csv`  
  Automatic fuzzy matching from GAEZ RES02 crop codes to FAOSTAT price and calorie items. This is a draft; manually check low-score matches before using.

- `gaez_representative_crop_standardization_checked.csv`  
  Manually checked mapping for the main representative GAEZ crop codes. Prefer this file over the fuzzy draft for first regressions.

## Recommended use

For each crop `c` and grid cell:

```text
value_potential_c = yield_t_per_ha_c * price_usd_per_tonne_c
calorie_potential_c = yield_t_per_ha_c * 1000 * kcal_per_kg_c
```

Then aggregate across crops using `max`, `top3 mean`, or `top5 mean`.

## Sources

- FAOSTAT bulk downloads: https://www.fao.org/faostat/
- Prices bulk ZIP: https://fenixservices.fao.org/faostat/static/bulkdownloads/Prices_E_All_Data_(Normalized).zip
- Food Balance Sheets bulk ZIP: https://fenixservices.fao.org/faostat/static/bulkdownloads/FoodBalanceSheets_E_All_Data_(Normalized).zip

## Caution

- Producer prices are country-year observations. The summary uses median values, not global market prices.
- Food Balance Sheet kcal/kg is an item-level food-use conversion. It may not match raw harvested crop mass for processing crops such as sugar cane, oil palm, cotton, tea, coffee, or fodder crops.
- The GAEZ matching file is intentionally marked as a draft. Use only checked crop mappings for regressions.
