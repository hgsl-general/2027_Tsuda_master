from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import textwrap
import traceback
from pathlib import Path

NOTEBOOK_PATH = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\cropland_fraction_two_stage_lgbm.ipynb")

cells = []

def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)})

def code(source: str):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(source).strip().splitlines(keepends=True)})

md(r'''
# Cropland fraction two-stage LightGBM, first pass

目的は、2024年の `cropland fraction` を、地形・市場アクセス・気候ポテンシャル・水資源アクセスでどれくらい説明できるかを見ることです。

このノートブックではまず exploratory に二段階モデルを使います。

1. `cropland_fraction > 0.01` かどうかを分類
2. 農地がある場所だけで `cropland_fraction` の大きさを回帰

注意: これは因果推定ではなく、農地化パターンを読むための最初の説明・予測モデルです。
''')

code(r'''
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import rasterio
from rasterio.enums import Resampling

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(r"C:\masterresearch\Comparative_advantage")
GAEZ_DIR = ROOT / "GAEZ"
HYDE_DIR = ROOT / "HYDE3.4"
CROPLAND_DIR = HYDE_DIR / "cropland_npys"
DIST_DIR = ROOT / "distance_to_cities"
GLOFAS_DIR = ROOT / "GloFAS" / "processed_5min"
STD_DIR = GAEZ_DIR / "StandardizationData"
FEATURE_CACHE = GAEZ_DIR / "CroplandRegression" / "features_cache"
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
YEAR = 2024
PRESENCE_THRESHOLD = 0.01
N_POS_SAMPLE = 120_000
N_ZERO_SAMPLE = 120_000
TOP_K_CROPS = 5
REPRESENTATIVE_CROP_TABLE = STD_DIR / "gaez_representative_crop_standardization_checked.csv"

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "DejaVu Sans"]

print("Notebook:", Path.cwd())
print("Feature cache:", FEATURE_CACHE)
''')

md(r'''
## 1. 入力ラスタを同じ 2160×4320 グリッドに揃える

- 被説明変数: HYDE cropland fraction 2024
- 人口密度: HYDE population density 2024
- 地形: elevation / slope
- 市場アクセス: 都市20k以上・任意港までの最短時間
- 水資源: GloFAS 2020 annual p10 discharge / reliable river 距離
- 気候資源: rainfed / irrigated climate potential を価格・カロリー換算して上位5作物平均
''')

code(r'''
HEIGHT = 2160
WIDTH = 4320
SHAPE = (HEIGHT, WIDTH)


def save_npy(path: Path, array: np.ndarray) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32, copy=False))
    return array


def load_or_build(path: Path, builder):
    if path.exists():
        print(f"load cache: {path.name}")
        return np.load(path, mmap_mode="r")
    print(f"build cache: {path.name}")
    array = builder()
    save_npy(path, array)
    return np.load(path, mmap_mode="r")


def clean_float(array: np.ndarray, nodata_values=(-9999, 65535)) -> np.ndarray:
    out = np.array(array, dtype=np.float32, copy=True)
    for nodata in nodata_values:
        out[out == nodata] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def safe_log1p(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    out = np.where(np.isfinite(out) & (out > 0), out, 0.0)
    return np.log1p(out).astype(np.float32)


def load_cropland_2024() -> np.ndarray:
    cropland = np.load(CROPLAND_DIR / "cropland_fraction_1950_2024.npy", mmap_mode="r")
    years = np.load(CROPLAND_DIR / "years.npy")
    year_idx = int(np.where(years == YEAR)[0][0])
    return clean_float(cropland[year_idx])


def load_population_2024() -> np.ndarray:
    cache_path = FEATURE_CACHE / "population_density_2024.npy"
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    with netCDF4.Dataset(HYDE_DIR / "population_density.nc") as ds:
        time_var = ds.variables["time"]
        dates = netCDF4.num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        years = np.array([date.year for date in dates])
        year_idx = int(np.where(years == YEAR)[0][0])
        pop = clean_float(ds.variables["population_density"][year_idx, :, :])
        pop[pop < 0] = np.nan
    return save_npy(cache_path, pop)


def load_tif_5min(tif_name: str, cache_name: str, resampling=Resampling.average, categorical: bool = False) -> np.ndarray:
    cache_path = FEATURE_CACHE / cache_name
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    with rasterio.open(GAEZ_DIR / tif_name) as src:
        data = src.read(1, out_shape=SHAPE, masked=True, resampling=resampling)
        if np.ma.isMaskedArray(data):
            arr = np.asarray(data.astype("float32").filled(np.nan), dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    if categorical:
        arr = np.where(np.isfinite(arr), np.rint(arr), np.nan).astype(np.float32)
    return save_npy(cache_path, arr)


def update_topk(top: np.ndarray, values: np.ndarray, row_start: int, row_stop: int) -> None:
    chunk_top = top[:, row_start:row_stop, :]
    min_idx = np.argmin(chunk_top, axis=0)
    min_vals = np.take_along_axis(chunk_top, min_idx[None, :, :], axis=0)[0]
    mask = values > min_vals
    if mask.any():
        rr, cc = np.where(mask)
        chunk_top[min_idx[rr, cc], rr, cc] = values[rr, cc]


def build_weighted_climate_topk(condition: str, weight_col: str, out_name: str, chunk_rows: int = 120) -> np.ndarray:
    out_path = FEATURE_CACHE / out_name
    # GAEZ RES02 yield values are treated as kg/ha.
    # price_usd_per_tonne_median converts kg/ha -> USD/ha by dividing by 1000.
    # kcal_per_kg_median converts kg/ha -> kcal/ha directly.
    unit_scale = 1.0 / 1000.0 if weight_col == "price_usd_per_tonne_median" else 1.0
    if out_path.exists():
        return np.load(out_path, mmap_mode="r")

    crops = pd.read_csv(REPRESENTATIVE_CROP_TABLE)
    if condition == "rainfed":
        folder = GAEZ_DIR / "ClimatePotentialRainfed"
        template = "climate_potential_yield_rainfed_{code}.npy"
    elif condition == "irrigated":
        folder = GAEZ_DIR / "ClimatePotential"
        template = "climate_potential_yield_{code}.npy"
    else:
        raise ValueError(condition)

    top = np.zeros((TOP_K_CROPS, HEIGHT, WIDTH), dtype=np.float32)
    for _, row in crops.iterrows():
        code = str(row["gaez_code"])
        weight = float(row[weight_col])
        path = folder / template.format(code=code)
        if not path.exists():
            print(f"missing climate file, skip: {path.name}")
            continue
        arr = np.load(path, mmap_mode="r")
        for row_start in range(0, HEIGHT, chunk_rows):
            row_stop = min(row_start + chunk_rows, HEIGHT)
            values = np.asarray(arr[row_start:row_stop, :], dtype=np.float32) * weight * unit_scale
            values = np.where(np.isfinite(values) & (values > 0), values, 0.0).astype(np.float32)
            update_topk(top, values, row_start, row_stop)
        print(f"  {condition} {weight_col}: {code}")

    top_mean = top.mean(axis=0).astype(np.float32)
    save_npy(out_path, top_mean)
    return np.load(out_path, mmap_mode="r")

cropland_2024 = load_cropland_2024()
pop_density_2024 = load_population_2024()
elevation_m = load_tif_5min("elevation.tif", "elevation_5min.npy", Resampling.average)
slope = load_tif_5min("slope.tif", "slope_5min.npy", Resampling.average)
exclusion = load_tif_5min("exclusion.tif", "exclusion_5min_mode.npy", Resampling.mode, categorical=True)

city_time_20k_min = np.load(DIST_DIR / "cities_10_1_12deg_min.npy", mmap_mode="r")
port_time_any_min = np.load(DIST_DIR / "ports_05_1_12deg_min.npy", mmap_mode="r")
glofas_p10 = np.load(GLOFAS_DIR / "p10_discharge_max_5min_2020.npy", mmap_mode="r")
distance_river_gt10 = np.load(GLOFAS_DIR / "distance_to_reliable_river_p10_gt_10_m3s_km_5min_2020.npy", mmap_mode="r")

rainfed_value_top5 = build_weighted_climate_topk("rainfed", "price_usd_per_tonne_median", "rainfed_value_top5_usd_per_ha_checked_36crops.npy")
irrigated_value_top5 = build_weighted_climate_topk("irrigated", "price_usd_per_tonne_median", "irrigated_value_top5_usd_per_ha_checked_36crops.npy")
rainfed_calorie_top5 = build_weighted_climate_topk("rainfed", "kcal_per_kg_median", "rainfed_calorie_top5_kcal_per_ha_checked_36crops.npy")
irrigated_calorie_top5 = build_weighted_climate_topk("irrigated", "kcal_per_kg_median", "irrigated_calorie_top5_kcal_per_ha_checked_36crops.npy")

print("loaded rasters")
print("cropland", cropland_2024.shape, cropland_2024.dtype, "finite", int(np.isfinite(cropland_2024).sum()))
print("land finite elevation", int(np.isfinite(elevation_m).sum()))
print("climate potential value unit: USD/ha, computed as GAEZ yield kg/ha / 1000 * FAOSTAT USD/tonne")
print("climate potential calorie unit: kcal/ha, computed as GAEZ yield kg/ha * FAOSTAT kcal/kg")
''')

md(r'''
## 2. 学習用サンプルを作る

全グリッドをそのまま使うと重いので、まずは exploratory に以下でサンプリングします。

- `cropland_fraction > 0.01` のセルから最大 120,000
- それ以外の陸上セルから最大 120,000
- 空間分割は 10度ブロック単位で train/test を分ける

この分割はランダムセル分割より厳しめで、近傍セルの漏洩を抑えます。
''')

code(r'''
lat = np.load(CROPLAND_DIR / "lat.npy")
lon = np.load(CROPLAND_DIR / "lon.npy")
lat_grid = np.broadcast_to(lat[:, None], SHAPE)
lon_grid = np.broadcast_to(lon[None, :], SHAPE)

land_mask = np.isfinite(elevation_m) & np.isfinite(cropland_2024) & np.isfinite(pop_density_2024) & (pop_density_2024 >= 0)
presence = land_mask & (cropland_2024 > PRESENCE_THRESHOLD)
absence = land_mask & ~presence

rng = np.random.default_rng(RANDOM_SEED)
pos_flat = np.flatnonzero(presence.ravel())
zero_flat = np.flatnonzero(absence.ravel())
pos_sample = rng.choice(pos_flat, size=min(N_POS_SAMPLE, len(pos_flat)), replace=False)
zero_sample = rng.choice(zero_flat, size=min(N_ZERO_SAMPLE, len(zero_flat)), replace=False)
sample_flat = np.concatenate([pos_sample, zero_sample])
rng.shuffle(sample_flat)
rows, cols = np.unravel_index(sample_flat, SHAPE)


def take(array: np.ndarray) -> np.ndarray:
    return np.asarray(array[rows, cols])

rainfed_value = take(rainfed_value_top5)
irrigated_value = take(irrigated_value_top5)
rainfed_calorie = take(rainfed_calorie_top5)
irrigated_calorie = take(irrigated_calorie_top5)

sample = pd.DataFrame({
    "row": rows.astype(np.int32),
    "col": cols.astype(np.int32),
    "lat": lat[rows].astype(np.float32),
    "lon": lon[cols].astype(np.float32),
    "cropland_fraction": take(cropland_2024).astype(np.float32),
    "presence": (take(cropland_2024) > PRESENCE_THRESHOLD).astype(np.uint8),
    "elevation_m": take(elevation_m).astype(np.float32),
    "slope": take(slope).astype(np.float32),
    "exclusion_class": np.nan_to_num(take(exclusion), nan=-1).astype(np.int16),
    "log_pop_density_2024": safe_log1p(take(pop_density_2024)),
    "log_city_time_20k_min": safe_log1p(take(city_time_20k_min)),
    "log_port_time_any_min": safe_log1p(take(port_time_any_min)),
    "log_glofas_p10_2020": safe_log1p(take(glofas_p10)),
    "log_distance_river_gt10_2020": safe_log1p(take(distance_river_gt10)),
    "log_rainfed_value_top5": safe_log1p(rainfed_value),
    "log_rainfed_calorie_top5": safe_log1p(rainfed_calorie),
    "log_irrigation_value_gain_top5": safe_log1p(np.maximum(irrigated_value - rainfed_value, 0)),
    "log_irrigation_calorie_gain_top5": safe_log1p(np.maximum(irrigated_calorie - rainfed_calorie, 0)),
})

sample = sample.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
sample["spatial_block"] = (
    np.floor((sample["lat"] + 90) / 10).astype(int) * 36
    + np.floor((sample["lon"] + 180) / 10).astype(int)
)

feature_cols = [
    "elevation_m",
    "slope",
    "exclusion_class",
    "log_pop_density_2024",
    "log_city_time_20k_min",
    "log_port_time_any_min",
    "log_glofas_p10_2020",
    "log_distance_river_gt10_2020",
    "log_rainfed_value_top5",
    "log_rainfed_calorie_top5",
    "log_irrigation_value_gain_top5",
    "log_irrigation_calorie_gain_top5",
]

print("sample rows:", len(sample))
print("presence share in sample:", sample["presence"].mean())
print("cropland summary:")
print(sample["cropland_fraction"].describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).to_string())
print("\nfeatures:")
print(sample[feature_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())
''')

md(r'''
## 3. 二段階 LightGBM

ここでは `10度空間ブロック` で train/test を分けています。

- Stage 1: 農地があるかを分類
- Stage 2: 農地があるセルだけで cropland fraction を回帰
''')

code(r'''
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(sample, groups=sample["spatial_block"]))
train = sample.iloc[train_idx].copy()
test = sample.iloc[test_idx].copy()

X_train = train[feature_cols]
y_train = train["presence"]
X_test = test[feature_cols]
y_test = test["presence"]

clf = LGBMClassifier(
    objective="binary",
    n_estimators=450,
    learning_rate=0.035,
    num_leaves=31,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=RANDOM_SEED,
    n_jobs=4,
    verbose=-1,
)
clf.fit(X_train, y_train, categorical_feature=["exclusion_class"])
proba_test = clf.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(np.uint8)

presence_metrics = {
    "test_auc": roc_auc_score(y_test, proba_test),
    "test_average_precision": average_precision_score(y_test, proba_test),
    "test_accuracy_0p5": accuracy_score(y_test, pred_test),
    "train_rows": len(train),
    "test_rows": len(test),
    "test_presence_share": float(y_test.mean()),
}

positive_train = train[train["presence"].eq(1)].copy()
positive_test = test[test["presence"].eq(1)].copy()

reg = LGBMRegressor(
    objective="regression",
    n_estimators=550,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=60,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=RANDOM_SEED,
    n_jobs=4,
    verbose=-1,
)
reg.fit(
    positive_train[feature_cols],
    positive_train["cropland_fraction"],
    categorical_feature=["exclusion_class"],
)
reg_pred = np.clip(reg.predict(positive_test[feature_cols]), 0, 1)
reg_y = positive_test["cropland_fraction"].to_numpy()
reg_metrics = {
    "positive_test_mae": mean_absolute_error(reg_y, reg_pred),
    "positive_test_rmse": math.sqrt(mean_squared_error(reg_y, reg_pred)),
    "positive_test_r2": r2_score(reg_y, reg_pred),
    "positive_train_rows": len(positive_train),
    "positive_test_rows": len(positive_test),
}

metrics = {"presence_model": presence_metrics, "fraction_model_positive_only": reg_metrics}
print(json.dumps(metrics, indent=2))
''')

md(r'''
## 4. 重要変数

`gain` importance は、モデル内でその変数がどれだけ損失改善に使われたかです。相関が強い変数間では重要度が分散・偏り得るので、解釈は「入口」として使います。
''')

code(r'''
def importance_frame(model, model_name: str) -> pd.DataFrame:
    booster = model.booster_
    return pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
        "model": model_name,
    }).sort_values("gain", ascending=False)

imp_clf = importance_frame(clf, "presence classifier")
imp_reg = importance_frame(reg, "fraction regressor")
print("Presence classifier importance")
print(imp_clf[["feature", "gain", "split"]].round(2).to_string(index=False))
print("\nPositive fraction regressor importance")
print(imp_reg[["feature", "gain", "split"]].round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
for ax, imp, title in [
    (axes[0], imp_clf, "Stage 1: cropland presence"),
    (axes[1], imp_reg, "Stage 2: fraction among positive cells"),
]:
    plot_df = imp.sort_values("gain", ascending=True).tail(12)
    ax.barh(plot_df["feature"], plot_df["gain"], color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel("LightGBM gain importance")
    ax.grid(axis="x", alpha=0.25)
plt.show()
''')

md(r'''
## 5. 主要変数の関係を見る

下の図は、テストデータを各変数の分位点でbinningし、観測値とモデル予測の平均を重ねたものです。Partial dependence ではなく、まず実データ上の関係確認です。

横軸は各binの中央値です。`log_*` 変数は `log(1 + 元の値)` なので、軸ラベルに「元の値の単位」も明記しています。
''')

code(r'''
def binned_curve(df: pd.DataFrame, x_col: str, y_col: str, pred_col: str, q: int = 20) -> pd.DataFrame:
    work = df[[x_col, y_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work[x_col].nunique() < 4:
        return pd.DataFrame()
    work["bin"] = pd.qcut(work[x_col], q=min(q, work[x_col].nunique()), duplicates="drop")
    out = work.groupby("bin", observed=True).agg(
        x_median=(x_col, "median"),
        observed=(y_col, "mean"),
        predicted=(pred_col, "mean"),
        n=(y_col, "size"),
    ).reset_index(drop=True)
    return out

plot_specs = [
    ("log_rainfed_value_top5", "Rainfed climate value potential\ntop-5 mean", "log(1 + USD/ha)"),
    ("log_irrigation_value_gain_top5", "Irrigation value gain\ntop-5 mean", "log(1 + USD/ha)"),
    ("log_distance_river_gt10_2020", "Distance to reliable river\np10 discharge > 10 m3/s", "log(1 + km)"),
    ("log_glofas_p10_2020", "GloFAS annual p10 discharge\n2020", "log(1 + m3/s)"),
    ("log_city_time_20k_min", "Travel time to city >=20k", "log(1 + minutes)"),
    ("slope", "Slope", "GAEZ slope class, 0-10"),
]

unit_table = pd.DataFrame([
    {"feature": "log_rainfed_value_top5", "x_axis_unit": "log(1 + USD/ha)", "raw_unit": "USD/ha", "definition": "top-5 mean of rainfed GAEZ climate potential kg/ha / 1000 * FAOSTAT USD/tonne"},
    {"feature": "log_irrigation_value_gain_top5", "x_axis_unit": "log(1 + USD/ha)", "raw_unit": "USD/ha", "definition": "max(irrigated value potential - rainfed value potential, 0), top-5 mean"},
    {"feature": "log_distance_river_gt10_2020", "x_axis_unit": "log(1 + km)", "raw_unit": "km", "definition": "distance to nearest 5-min cell where GloFAS annual p10 discharge > 10 m3/s"},
    {"feature": "log_glofas_p10_2020", "x_axis_unit": "log(1 + m3/s)", "raw_unit": "m3/s", "definition": "GloFAS 2020 annual p10 river discharge, max within 5-min cell"},
    {"feature": "log_city_time_20k_min", "x_axis_unit": "log(1 + minutes)", "raw_unit": "minutes", "definition": "travel time to nearest city with population >=20k"},
    {"feature": "slope", "x_axis_unit": "GAEZ slope class, 0-10", "raw_unit": "class", "definition": "GAEZ slope class aggregated to 5-min grid"},
])
print(unit_table.to_string(index=False))

test_plot = test.copy()
test_plot["pred_presence_prob"] = proba_test
pos_plot = positive_test.copy()
pos_plot["pred_fraction"] = reg_pred

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
axes = axes.ravel()
for ax, (x_col, title, xlabel) in zip(axes, plot_specs):
    curve = binned_curve(test_plot, x_col, "presence", "pred_presence_prob", q=20)
    if curve.empty:
        ax.set_axis_off()
        continue
    ax.plot(curve["x_median"], curve["observed"], marker="o", linewidth=1.6, label="observed presence")
    ax.plot(curve["x_median"], curve["predicted"], marker="s", linewidth=1.6, label="predicted probability")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cropland presence rate")
    ax.grid(alpha=0.25)
axes[0].legend(loc="best")
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
axes = axes.ravel()
for ax, (x_col, title, xlabel) in zip(axes, plot_specs):
    curve = binned_curve(pos_plot, x_col, "cropland_fraction", "pred_fraction", q=20)
    if curve.empty:
        ax.set_axis_off()
        continue
    ax.plot(curve["x_median"], curve["observed"], marker="o", linewidth=1.6, label="observed fraction")
    ax.plot(curve["x_median"], curve["predicted"], marker="s", linewidth=1.6, label="predicted fraction")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cropland fraction | positive")
    ax.grid(alpha=0.25)
axes[0].legend(loc="best")
plt.show()
''')

md(r'''
## 6. 空間的な当たり外れをサンプルで確認

点はテストサンプルです。全球フル予測ではなく、まず空間的にどこで外れているかを粗く見ます。
''')

code(r'''
map_sample = test.copy()
map_sample["pred_presence_prob"] = proba_test
map_sample["presence_error"] = map_sample["presence"] - map_sample["pred_presence_prob"]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
sc0 = axes[0].scatter(map_sample["lon"], map_sample["lat"], c=map_sample["presence"], s=1, cmap="Greens", alpha=0.55)
axes[0].set_title("Observed cropland presence")
fig.colorbar(sc0, ax=axes[0], shrink=0.8)

sc1 = axes[1].scatter(map_sample["lon"], map_sample["lat"], c=map_sample["pred_presence_prob"], s=1, cmap="viridis", alpha=0.55, vmin=0, vmax=1)
axes[1].set_title("Predicted presence probability")
fig.colorbar(sc1, ax=axes[1], shrink=0.8)

sc2 = axes[2].scatter(map_sample["lon"], map_sample["lat"], c=map_sample["presence_error"], s=1, cmap="coolwarm", alpha=0.55, vmin=-1, vmax=1)
axes[2].set_title("Observed - predicted")
fig.colorbar(sc2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.15)
plt.show()
''')

md(r'''
## 7. この初回分析の読み方

- このモデルは「どの要因が cropland presence / fraction を説明しやすいか」を見る入口。
- 重要度・binning 曲線で、気候資源、水アクセス、市場アクセス、地形の効き方を確認する。
- 次にやるなら、国境・国固定効果・長期平均GloFAS・空間ブロックの設計を強化する。
''')

# Execute notebook cells and capture outputs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

namespace = {"__name__": "__main__"}
execution_count = 0
executed_cells = []
failed = False

for cell in cells:
    if cell["cell_type"] != "code":
        executed_cells.append(cell)
        continue
    execution_count += 1
    source = "".join(cell["source"])
    stdout_buffer = io.StringIO()
    stderr_text = ""
    outputs = []
    plt.close("all")
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(source, namespace)
    except Exception:
        failed = True
        stderr_text = traceback.format_exc()
    stdout = stdout_buffer.getvalue()
    if stdout:
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.splitlines(keepends=True)})
    if stderr_text:
        outputs.append({"output_type": "error", "ename": "Exception", "evalue": stderr_text.splitlines()[-1] if stderr_text else "", "traceback": stderr_text.splitlines()})
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", bbox_inches="tight", dpi=140)
        encoded = base64.b64encode(png_buffer.getvalue()).decode("ascii")
        outputs.append({"output_type": "display_data", "data": {"image/png": encoded}, "metadata": {}})
    plt.close("all")
    cell = dict(cell)
    cell["execution_count"] = execution_count
    cell["outputs"] = outputs
    executed_cells.append(cell)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": executed_cells + cells[len(executed_cells):],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    if failed:
        break

notebook = {
    "cells": executed_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote notebook: {NOTEBOOK_PATH}")
if failed:
    print("Notebook execution failed. See last cell output.")
    sys.exit(1)
print("Notebook execution completed.")
