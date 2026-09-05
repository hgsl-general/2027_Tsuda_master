from pathlib import Path
path = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\build_and_run_cropland_lgbm_notebook.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
'''def build_weighted_climate_topk(condition: str, weight_col: str, out_name: str, chunk_rows: int = 120) -> np.ndarray:\n    out_path = FEATURE_CACHE / out_name\n''',
'''def build_weighted_climate_topk(condition: str, weight_col: str, out_name: str, chunk_rows: int = 120) -> np.ndarray:\n    out_path = FEATURE_CACHE / out_name\n    # GAEZ RES02 yield values are treated as kg/ha.\n    # price_usd_per_tonne_median converts kg/ha -> USD/ha by dividing by 1000.\n    # kcal_per_kg_median converts kg/ha -> kcal/ha directly.\n    unit_scale = 1.0 / 1000.0 if weight_col == "price_usd_per_tonne_median" else 1.0\n''')

text = text.replace(
'''            values = np.asarray(arr[row_start:row_stop, :], dtype=np.float32) * weight\n''',
'''            values = np.asarray(arr[row_start:row_stop, :], dtype=np.float32) * weight * unit_scale\n''')

text = text.replace(
'''rainfed_value_top5 = build_weighted_climate_topk("rainfed", "price_usd_per_tonne_median", "rainfed_value_top5_checked_36crops.npy")\nirrigated_value_top5 = build_weighted_climate_topk("irrigated", "price_usd_per_tonne_median", "irrigated_value_top5_checked_36crops.npy")\nrainfed_calorie_top5 = build_weighted_climate_topk("rainfed", "kcal_per_kg_median", "rainfed_calorie_top5_checked_36crops.npy")\nirrigated_calorie_top5 = build_weighted_climate_topk("irrigated", "kcal_per_kg_median", "irrigated_calorie_top5_checked_36crops.npy")\n''',
'''rainfed_value_top5 = build_weighted_climate_topk("rainfed", "price_usd_per_tonne_median", "rainfed_value_top5_usd_per_ha_checked_36crops.npy")\nirrigated_value_top5 = build_weighted_climate_topk("irrigated", "price_usd_per_tonne_median", "irrigated_value_top5_usd_per_ha_checked_36crops.npy")\nrainfed_calorie_top5 = build_weighted_climate_topk("rainfed", "kcal_per_kg_median", "rainfed_calorie_top5_kcal_per_ha_checked_36crops.npy")\nirrigated_calorie_top5 = build_weighted_climate_topk("irrigated", "kcal_per_kg_median", "irrigated_calorie_top5_kcal_per_ha_checked_36crops.npy")\n''')

text = text.replace(
'''print("loaded rasters")\nprint("cropland", cropland_2024.shape, cropland_2024.dtype, "finite", int(np.isfinite(cropland_2024).sum()))\nprint("land finite elevation", int(np.isfinite(elevation_m).sum()))\n''',
'''print("loaded rasters")\nprint("cropland", cropland_2024.shape, cropland_2024.dtype, "finite", int(np.isfinite(cropland_2024).sum()))\nprint("land finite elevation", int(np.isfinite(elevation_m).sum()))\nprint("climate potential value unit: USD/ha, computed as GAEZ yield kg/ha / 1000 * FAOSTAT USD/tonne")\nprint("climate potential calorie unit: kcal/ha, computed as GAEZ yield kg/ha * FAOSTAT kcal/kg")\n''')

text = text.replace(
'''## 5. 主要変数の関係を見る\n\n下の図は、テストデータを各変数の分位点でbinningし、観測値とモデル予測の平均を重ねたものです。Partial dependence ではなく、まず実データ上の関係確認です。\n''',
'''## 5. 主要変数の関係を見る\n\n下の図は、テストデータを各変数の分位点でbinningし、観測値とモデル予測の平均を重ねたものです。Partial dependence ではなく、まず実データ上の関係確認です。\n\n横軸は各binの中央値です。`log_*` 変数は `log(1 + 元の値)` なので、軸ラベルに「元の値の単位」も明記しています。\n''')

old_block = '''plot_vars = [\n    "log_rainfed_value_top5",\n    "log_irrigation_value_gain_top5",\n    "log_distance_river_gt10_2020",\n    "log_glofas_p10_2020",\n    "log_city_time_20k_min",\n    "slope",\n]\n'''
new_block = '''plot_specs = [\n    ("log_rainfed_value_top5", "Rainfed climate value potential\\ntop-5 mean", "log(1 + USD/ha)"),\n    ("log_irrigation_value_gain_top5", "Irrigation value gain\\ntop-5 mean", "log(1 + USD/ha)"),\n    ("log_distance_river_gt10_2020", "Distance to reliable river\\np10 discharge > 10 m3/s", "log(1 + km)"),\n    ("log_glofas_p10_2020", "GloFAS annual p10 discharge\\n2020", "log(1 + m3/s)"),\n    ("log_city_time_20k_min", "Travel time to city >=20k", "log(1 + minutes)"),\n    ("slope", "Slope", "GAEZ slope class, 0-10"),\n]\n\nunit_table = pd.DataFrame([\n    {"feature": "log_rainfed_value_top5", "x_axis_unit": "log(1 + USD/ha)", "raw_unit": "USD/ha", "definition": "top-5 mean of rainfed GAEZ climate potential kg/ha / 1000 * FAOSTAT USD/tonne"},\n    {"feature": "log_irrigation_value_gain_top5", "x_axis_unit": "log(1 + USD/ha)", "raw_unit": "USD/ha", "definition": "max(irrigated value potential - rainfed value potential, 0), top-5 mean"},\n    {"feature": "log_distance_river_gt10_2020", "x_axis_unit": "log(1 + km)", "raw_unit": "km", "definition": "distance to nearest 5-min cell where GloFAS annual p10 discharge > 10 m3/s"},\n    {"feature": "log_glofas_p10_2020", "x_axis_unit": "log(1 + m3/s)", "raw_unit": "m3/s", "definition": "GloFAS 2020 annual p10 river discharge, max within 5-min cell"},\n    {"feature": "log_city_time_20k_min", "x_axis_unit": "log(1 + minutes)", "raw_unit": "minutes", "definition": "travel time to nearest city with population >=20k"},\n    {"feature": "slope", "x_axis_unit": "GAEZ slope class, 0-10", "raw_unit": "class", "definition": "GAEZ slope class aggregated to 5-min grid"},\n])\nprint(unit_table.to_string(index=False))\n'''
if old_block not in text:
    raise RuntimeError('plot_vars block not found')
text = text.replace(old_block, new_block)

text = text.replace('for ax, x_col in zip(axes, plot_vars):\n', 'for ax, (x_col, title, xlabel) in zip(axes, plot_specs):\n')
text = text.replace('    ax.set_title(x_col)\n    ax.set_ylabel("cropland presence rate")\n', '    ax.set_title(title)\n    ax.set_xlabel(xlabel)\n    ax.set_ylabel("cropland presence rate")\n')
text = text.replace('    ax.set_title(x_col)\n    ax.set_ylabel("cropland fraction | positive")\n', '    ax.set_title(title)\n    ax.set_xlabel(xlabel)\n    ax.set_ylabel("cropland fraction | positive")\n')

path.write_text(text, encoding="utf-8")
print('patched notebook builder with explicit units')
