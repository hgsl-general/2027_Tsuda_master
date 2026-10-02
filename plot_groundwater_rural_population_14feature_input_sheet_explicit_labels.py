from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_SCRIPT = Path(
    "/work/tsuda/GAEZ/plot_groundwater_rural_population_14feature_input_sheet.py"
)

spec = importlib.util.spec_from_file_location("input_sheet_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load plotting module: {BASE_SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


EXPLICIT_LABELS = {
    "elevation_m": {
        "label": "標高 / Elevation\n地表の標高",
        "unit": "標高 (m)",
    },
    "slope": {
        "label": "傾斜 / Slope\n地表面の傾斜を表すモデル入力値",
        "unit": "傾斜のモデル入力値",
    },
    "exclusion_class": {
        "label": "環境除外区分 / Environmental exclusion\n保護・環境条件による除外クラス",
        "unit": "除外クラスコード",
    },
    "soil_group_class": {
        "label": "土壌群 / Soil group\n5分グリッドの土壌グループ分類",
        "unit": "土壌群クラスコード",
    },
    "rainfed_calorie_top5_raw": {
        "label": "天水農業カロリー潜在量 / Local rainfed potential\n対象グリッドの天水条件Top-5作物カロリー潜在量",
        "unit": "未変換のカロリー潜在量入力値",
    },
    "irrigation_calorie_gain_top5_raw": {
        "label": "灌漑によるカロリー増分 / Irrigation gain\n灌漑時と天水時のTop-5作物カロリー潜在量の差",
        "unit": "未変換の灌漑増分入力値",
    },
    "wx_50km_rainfed_calorie_top5_raw": {
        "label": "周囲50 kmの天水カロリー潜在量 / 50-km rainfed potential\n近隣50 kmの天水条件Top-5作物カロリー指標",
        "unit": "未変換の50 km周辺入力値",
    },
    "log_distance_river_gt10_2020": {
        "label": "大河川までの距離 / Distance to large river\nGloFAS p10流量 > 10 m³/sの河川までの距離",
        "unit": "log1p(河川までの距離)",
    },
    "log_glofas_p10_2020": {
        "label": "局所河川流量 / Local GloFAS p10 flow\n対象グリッドのGloFAS p10河川流量",
        "unit": "log1p(流量 m³/s)",
    },
    "log_fan_water_table_depth_m": {
        "label": "地下水面深度 / Groundwater-table depth\nFanデータによる地表から地下水面までの深さ",
        "unit": "log1p(地下水面深度 m)",
    },
    "log_watergap_total_recharge_mm_yr": {
        "label": "地下水涵養量 / Groundwater recharge\nWaterGAPの年間総地下水涵養量",
        "unit": "log1p(涵養量 mm/年)",
    },
    "log_city_time_20k_min": {
        "label": "都市への移動時間 / City accessibility\n人口2万人以上の都市までの所要時間",
        "unit": "log1p(移動時間 分)",
    },
    "log_port_time_any_min": {
        "label": "港への移動時間 / Port accessibility\n最寄りの港までの所要時間",
        "unit": "log1p(移動時間 分)",
    },
    "log_rural_population_density_2020": {
        "label": "非都市人口密度 / Rural population density\n2020年非都市人口 ÷ グリッド面積",
        "unit": "log1p(人/km²)",
    },
}


features_in_sheet = {item["feature"] for item in module.FEATURE_SPECS}
if features_in_sheet != set(EXPLICIT_LABELS):
    raise RuntimeError(
        "Explicit labels do not match the 14 plotted features: "
        f"missing={features_in_sheet - set(EXPLICIT_LABELS)}, "
        f"extra={set(EXPLICIT_LABELS) - features_in_sheet}"
    )

for item in module.FEATURE_SPECS:
    item.update(EXPLICIT_LABELS[item["feature"]])


if __name__ == "__main__":
    module.main()
