from __future__ import annotations

import gc
import json
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.colors import Normalize, TwoSlopeNorm
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import GroupKFold


warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/work/tsuda")
GAEZ_DIR = ROOT / "GAEZ"
HYDE_DIR = ROOT / "HYDE3.4"
CROPLAND_DIR = HYDE_DIR / "cropland_npys"
DIST_DIR = ROOT / "distance_to_cities"
GLOFAS_DIR = ROOT / "GloFAS" / "processed_5min"
FEATURE_CACHE = GAEZ_DIR / "CroplandRegression" / "features_cache"
WX_CACHE_DIR = GAEZ_DIR / "CroplandRegression" / "spatial_wx_comparison"
LAND_MASK_DIR = GAEZ_DIR / "LandMasks" / "derived_5min"
GROUNDWATER_DIR = GAEZ_DIR / "WaterData" / "groundwater"
WTD_RAW_DIR = GROUNDWATER_DIR / "fan_wtd" / "raw_continents"
WATERGAP_DIR = GROUNDWATER_DIR / "watergap_2_2d" / "processed"
BASIC_DIR = (
    GAEZ_DIR
    / "CroplandRegression"
    / "soil_group_calorie_only_50km_target2020"
)
BASIC_OOF_PATH = BASIC_DIR / "soil_comparison_oof_predictions.csv.gz"
OUTPUT_DIR = (
    GAEZ_DIR
    / "CroplandRegression"
    / "water_irrigation_regional_sensitivity_target2020"
)
FEATURE_DIR = OUTPUT_DIR / "derived_features"
CHECKPOINT_DIR = OUTPUT_DIR / "oof_checkpoints"
MAP_DIR = OUTPUT_DIR / "regional_error_change_maps"

for directory in [OUTPUT_DIR, FEATURE_DIR, CHECKPOINT_DIR, MAP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

YEAR = 2020
RANDOM_SEED = 42
N_SPLITS = 5
N_JOBS = 4
PRESENCE_THRESHOLD = 0.01
BOOTSTRAP_REPS = 2000
TARGET_SHAPE = (2160, 4320)
TARGET_RESOLUTION = 1.0 / 12.0
TARGET_TRANSFORM = from_origin(
    -180.0,
    90.0,
    TARGET_RESOLUTION,
    TARGET_RESOLUTION,
)
TARGET_CRS = "EPSG:4326"

# These are exactly the bounding boxes used by the existing high-resolution
# regional error figures in regional_oof_error_diagnostics_2020.
REGIONS = OrderedDict(
    [
        (
            "usa_around_100w",
            {
                "label": "USA around 100°W",
                "bounds": (-115.0, -85.0, 25.0, 50.0),
            },
        ),
        (
            "nile_region",
            {
                "label": "Nile region / Northeast Africa",
                "bounds": (-10.0, 45.0, 0.0, 35.0),
                # Keep the original bounds above for regional metrics, but
                # zoom the map to the Nile corridor and its main headwaters.
                "plot_bounds": (18.0, 42.0, -2.0, 33.0),
                "plot_label": "Nile corridor (zoomed view)",
            },
        ),
        (
            "yellow_river_region",
            {
                "label": "Yellow River region",
                "bounds": (95.0, 125.0, 25.0, 45.0),
            },
        ),
        (
            "japan_region",
            {
                "label": "Japan and surrounding region",
                "bounds": (127.0, 146.0, 30.0, 46.0),
            },
        ),
    ]
)

MODEL_LABELS = {
    "baseline": "Basic model (10 m³/s river threshold)",
    "add_fan_wtd": "+ Fan water-table depth",
    "add_watergap_recharge": "+ WaterGAP groundwater recharge",
    "add_groundwater_both": "+ WTD and groundwater recharge",
    "river_threshold_1": "River threshold relaxed to 1 m³/s",
    "river_threshold_0p1": "River threshold relaxed to 0.1 m³/s",
    "add_glofas_50km": "+ 50-km surrounding GloFAS flow",
    "combined_selected_no_irrigation": (
        "Selected water package (observed irrigation excluded)"
    ),
}

MODEL_CHANGE_LINES = {
    "add_fan_wtd": [
        "Added: groundwater-table depth [log_fan_water_table_depth_m]",
    ],
    "add_watergap_recharge": [
        "Added: groundwater recharge "
        "[log_watergap_total_recharge_mm_yr]",
    ],
    "add_groundwater_both": [
        "Added: groundwater-table depth [log_fan_water_table_depth_m]",
        "Added: groundwater recharge "
        "[log_watergap_total_recharge_mm_yr]",
    ],
    "river_threshold_1": [
        "Replaced river-distance threshold: 10 to 1 m³/s "
        "[log_distance_river_gt1_2020]",
    ],
    "river_threshold_0p1": [
        "Replaced river-distance threshold: 10 to 0.1 m³/s "
        "[log_distance_river_gt0p1_2020]",
    ],
    "add_glofas_50km": [
        "Added: surrounding 50-km GloFAS low-flow statistic "
        "[wx_50km_log_glofas_p10_2020]",
    ],
    "combined_selected_no_irrigation": [
        "Observed irrigation-area variables excluded",
        "Changes: river threshold 10 to 1 m³/s "
        "[log_distance_river_gt1_2020]",
        "Added: groundwater depth [log_fan_water_table_depth_m] + "
        "recharge [log_watergap_total_recharge_mm_yr]",
        "Added: surrounding 50-km GloFAS flow "
        "[wx_50km_log_glofas_p10_2020]",
    ],
}


WATER_FACTORIAL_SPECS = OrderedDict(
    [
        (
            "water_gw_t10",
            {
                "threshold": "10",
                "river_feature": "log_distance_river_gt10_2020",
                "add_wx50": False,
                "label": "WTD + recharge; 10 m³/s threshold; no 50-km flow",
            },
        ),
        (
            "water_gw_t10_wx50",
            {
                "threshold": "10",
                "river_feature": "log_distance_river_gt10_2020",
                "add_wx50": True,
                "label": "WTD + recharge; 10 m³/s threshold; + 50-km flow",
            },
        ),
        (
            "water_gw_t1",
            {
                "threshold": "1",
                "river_feature": "log_distance_river_gt1_2020",
                "add_wx50": False,
                "label": "WTD + recharge; 1 m³/s threshold; no 50-km flow",
            },
        ),
        (
            "water_gw_t1_wx50",
            {
                "threshold": "1",
                "river_feature": "log_distance_river_gt1_2020",
                "add_wx50": True,
                "label": "WTD + recharge; 1 m³/s threshold; + 50-km flow",
            },
        ),
        (
            "water_gw_t0p1",
            {
                "threshold": "0.1",
                "river_feature": "log_distance_river_gt0p1_2020",
                "add_wx50": False,
                "label": "WTD + recharge; 0.1 m³/s threshold; no 50-km flow",
            },
        ),
        (
            "water_gw_t0p1_wx50",
            {
                "threshold": "0.1",
                "river_feature": "log_distance_river_gt0p1_2020",
                "add_wx50": True,
                "label": "WTD + recharge; 0.1 m³/s threshold; + 50-km flow",
            },
        ),
    ]
)
for _model_name, _specification in WATER_FACTORIAL_SPECS.items():
    MODEL_LABELS[_model_name] = _specification["label"]
    _wx_description = (
        "Added: surrounding 50-km GloFAS flow "
        "[wx_50km_log_glofas_p10_2020]"
        if _specification["add_wx50"]
        else "Surrounding 50-km GloFAS flow: not included"
    )
    MODEL_CHANGE_LINES[_model_name] = [
        "Fixed additions: groundwater depth [log_fan_water_table_depth_m] + "
        "recharge [log_watergap_total_recharge_mm_yr]",
        f"River-distance threshold: {_specification['threshold']} m³/s "
        f"[{_specification['river_feature']}]",
        _wx_description,
    ]


def safe_log1p(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    return np.log1p(values).astype(np.float32)


def positive_raw(values):
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        np.isfinite(values) & (values > 0),
        values,
        0.0,
    ).astype(np.float32)


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_distance_to_river_0p1(lat, lon):
    output_path = (
        FEATURE_DIR
        / "distance_to_reliable_river_p10_gt_0p1_m3s_km_5min_2020.npy"
    )
    if output_path.exists():
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape == TARGET_SHAPE:
            print("Reuse:", output_path, flush=True)
            return output_path

    p10_path = require(GLOFAS_DIR / "p10_discharge_max_5min_2020.npy")
    p10 = np.load(p10_path, mmap_mode="r")
    reliable = np.isfinite(p10) & (p10 > 0.1)
    print(
        "Build 0.1 m3/s river distance; reliable cells:",
        f"{int(reliable.sum()):,}",
        flush=True,
    )
    nearest = distance_transform_edt(
        ~reliable,
        return_distances=False,
        return_indices=True,
    )
    nearest_rows = nearest[0].astype(np.int32, copy=False)
    nearest_cols = nearest[1].astype(np.int32, copy=False)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype="float32",
        shape=TARGET_SHAPE,
    )
    earth_radius_km = 6371.0088
    lon_self = np.asarray(lon, dtype=np.float64)[None, :]
    for row_start in range(0, TARGET_SHAPE[0], 128):
        row_stop = min(row_start + 128, TARGET_SHAPE[0])
        row_slice = slice(row_start, row_stop)
        lat1 = np.radians(np.asarray(lat[row_slice], dtype=np.float64)[:, None])
        lat2 = np.radians(lat[nearest_rows[row_slice, :]])
        dlat = lat2 - lat1
        dlon = np.radians(
            lon[nearest_cols[row_slice, :]] - lon_self
        )
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        output[row_slice, :] = (
            2.0
            * earth_radius_km
            * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        ).astype(np.float32)
        if row_start == 0 or row_stop == TARGET_SHAPE[0] or row_start % 640 == 0:
            print(
                f"  0.1 m3/s distance rows {row_start}:{row_stop}",
                flush=True,
            )
    output.flush()
    del output, nearest, nearest_rows, nearest_cols, reliable
    gc.collect()
    print("Saved:", output_path, flush=True)
    return output_path


def build_global_fan_wtd_5min():
    output_path = FEATURE_DIR / "fan_water_table_depth_m_5min_global.npy"
    if output_path.exists():
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape == TARGET_SHAPE:
            print("Reuse:", output_path, flush=True)
            return output_path

    source_paths = [
        require(WTD_RAW_DIR / "AFRICA_WTD_annualmean.nc"),
        require(WTD_RAW_DIR / "EURASIA_WTD_annualmean.nc"),
        require(WTD_RAW_DIR / "NAMERICA_WTD_annualmean.nc"),
        require(WTD_RAW_DIR / "OCEANIA_WTD_annualmean.nc"),
        require(WTD_RAW_DIR / "SAMERICA_WTD_annualmean.nc"),
    ]
    global_depth = np.full(TARGET_SHAPE, np.nan, dtype=np.float32)
    best_coverage = np.zeros(TARGET_SHAPE, dtype=np.float32)

    for path in source_paths:
        print("Aggregate Fan WTD to 5 arc-minutes:", path.name, flush=True)
        raw_average = np.full(TARGET_SHAPE, np.nan, dtype=np.float32)
        mask_fraction = np.zeros(TARGET_SHAPE, dtype=np.float32)
        wtd_uri = f'NETCDF:"{path}":WTD'
        mask_uri = f'NETCDF:"{path}":mask'
        with rasterio.open(wtd_uri) as wtd_source:
            scale = float(wtd_source.scales[0])
            offset = float(wtd_source.offsets[0])
            reproject(
                source=rasterio.band(wtd_source, 1),
                destination=raw_average,
                src_transform=wtd_source.transform,
                src_crs=TARGET_CRS,
                src_nodata=wtd_source.nodata,
                dst_transform=TARGET_TRANSFORM,
                dst_crs=TARGET_CRS,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
        with rasterio.open(mask_uri) as mask_source:
            reproject(
                source=rasterio.band(mask_source, 1),
                destination=mask_fraction,
                src_transform=mask_source.transform,
                src_crs=TARGET_CRS,
                src_nodata=None,
                dst_transform=TARGET_TRANSFORM,
                dst_crs=TARGET_CRS,
                dst_nodata=0.0,
                resampling=Resampling.average,
            )

        # Outside the WTD domain the encoded WTD is zero metres after scale
        # and offset. Dividing by the domain fraction therefore recovers the
        # mean over valid land pixels inside each 5-minute cell.
        decoded_average = scale * raw_average + offset
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = -decoded_average / mask_fraction
        valid = (
            np.isfinite(depth)
            & (depth >= 0.0)
            & (mask_fraction >= 0.25)
            & (mask_fraction > best_coverage)
        )
        global_depth[valid] = np.clip(depth[valid], 0.0, 1000.0)
        best_coverage[valid] = mask_fraction[valid]
        print(
            "  finite global cells so far:",
            f"{int(np.isfinite(global_depth).sum()):,}",
            flush=True,
        )
        del raw_average, mask_fraction, decoded_average, depth, valid
        gc.collect()

    np.save(output_path, global_depth)
    print("Saved:", output_path, flush=True)
    return output_path


def sample_watergap_recharge(sample_lat, sample_lon):
    path = require(
        WATERGAP_DIR / "watergap_total_recharge_mm_yr_2000_2010.nc"
    )
    variable = "groundwater_recharge_total_mm_yr"
    with xr.open_dataset(path, decode_times=False) as dataset:
        data = np.asarray(dataset[variable].values, dtype=np.float32)
        lat_values = np.asarray(dataset["lat"].values, dtype=float)
        lon_values = np.asarray(dataset["lon"].values, dtype=float)

    lat_step = float(np.median(np.diff(lat_values)))
    lon_step = float(np.median(np.diff(lon_values)))
    lat_index = np.rint((sample_lat - lat_values[0]) / lat_step).astype(int)
    lon_index = np.rint((sample_lon - lon_values[0]) / lon_step).astype(int)
    lat_index = np.clip(lat_index, 0, len(lat_values) - 1)
    lon_index = np.clip(lon_index, 0, len(lon_values) - 1)
    return data[lat_index, lon_index]


def load_analysis_sample():
    print("Load saved basic-model OOF sample.", flush=True)
    sample = pd.read_csv(require(BASIC_OOF_PATH), compression="gzip")
    required = {
        "row",
        "col",
        "lat",
        "lon",
        "spatial_block",
        "presence",
        "cropland_fraction",
        "pred_soil_group_wetland_excluded",
    }
    missing = required - set(sample.columns)
    if missing:
        raise KeyError(f"Basic OOF table is missing: {sorted(missing)}")
    if len(sample) != 240_000:
        raise RuntimeError(f"Expected 240,000 OOF rows; got {len(sample):,}")

    rows = sample["row"].to_numpy(dtype=int)
    cols = sample["col"].to_numpy(dtype=int)
    lat = np.load(require(CROPLAND_DIR / "lat.npy"))
    lon = np.load(require(CROPLAND_DIR / "lon.npy"))
    if (len(lat), len(lon)) != TARGET_SHAPE:
        raise RuntimeError("Unexpected target grid shape.")

    distance_0p1_path = build_distance_to_river_0p1(lat, lon)
    wtd_5min_path = build_global_fan_wtd_5min()

    def take(path):
        array = np.load(require(path), mmap_mode="r")
        if array.shape != TARGET_SHAPE:
            raise ValueError(f"Shape mismatch: {path}: {array.shape}")
        return np.asarray(array[rows, cols])

    elevation = take(FEATURE_CACHE / "elevation_5min.npy")
    slope = take(FEATURE_CACHE / "slope_5min.npy")
    exclusion = take(LAND_MASK_DIR / "gaez_v5_exclusion_5min_mode.npy")
    soil = take(LAND_MASK_DIR / "gaez_v5_wrb_soil_group_5min_mode.npy")
    city_time = take(DIST_DIR / "cities_10_1_12deg_min.npy")
    port_time = take(DIST_DIR / "ports_05_1_12deg_min.npy")
    glofas = take(GLOFAS_DIR / "p10_discharge_max_5min_2020.npy")
    distance_10 = take(
        GLOFAS_DIR
        / "distance_to_reliable_river_p10_gt_10_m3s_km_5min_2020.npy"
    )
    distance_1 = take(
        GLOFAS_DIR
        / "distance_to_reliable_river_p10_gt_1_m3s_km_5min_2020.npy"
    )
    distance_0p1 = take(distance_0p1_path)
    rainfed_calorie = take(
        FEATURE_CACHE
        / "rainfed_calorie_top5_kcal_per_ha_checked_36crops.npy"
    ).astype(np.float32)
    irrigated_calorie = take(
        FEATURE_CACHE
        / "irrigated_calorie_top5_kcal_per_ha_checked_36crops.npy"
    ).astype(np.float32)
    wx_calorie = take(
        WX_CACHE_DIR / "wx_50km_rainfed_calorie_top5_raw.npy"
    )
    wx_glofas = take(
        WX_CACHE_DIR / "wx_50km_log_glofas_p10_2020.npy"
    ).astype(np.float32)
    wtd = take(wtd_5min_path).astype(np.float32)
    recharge = sample_watergap_recharge(
        sample["lat"].to_numpy(dtype=float),
        sample["lon"].to_numpy(dtype=float),
    ).astype(np.float32)

    calorie_valid = np.isfinite(rainfed_calorie) & np.isfinite(irrigated_calorie)
    irrigation_gain = np.where(
        calorie_valid,
        np.maximum(irrigated_calorie - rainfed_calorie, 0.0),
        0.0,
    ).astype(np.float32)

    finite_wtd = np.isfinite(wtd) & (wtd >= 0)
    wtd_imputation = float(np.median(wtd[finite_wtd]))
    wtd = np.where(finite_wtd, wtd, wtd_imputation).astype(np.float32)
    finite_recharge = np.isfinite(recharge) & (recharge >= 0)
    recharge_imputation = float(np.median(recharge[finite_recharge]))
    recharge = np.where(
        finite_recharge,
        recharge,
        recharge_imputation,
    ).astype(np.float32)
    sample["elevation_m"] = elevation.astype(np.float32)
    sample["slope"] = slope.astype(np.float32)
    sample["exclusion_class"] = exclusion.astype(np.int16)
    sample["soil_group_class"] = soil.astype(np.int16)
    sample["log_city_time_20k_min"] = safe_log1p(city_time)
    sample["log_port_time_any_min"] = safe_log1p(port_time)
    sample["rainfed_calorie_top5_raw"] = positive_raw(rainfed_calorie)
    sample["irrigation_calorie_gain_top5_raw"] = irrigation_gain
    sample["wx_50km_rainfed_calorie_top5_raw"] = wx_calorie.astype(np.float32)
    sample["log_distance_river_gt10_2020"] = safe_log1p(distance_10)
    sample["log_glofas_p10_2020"] = safe_log1p(glofas)
    sample["log_fan_water_table_depth_m"] = safe_log1p(wtd)
    sample["log_watergap_total_recharge_mm_yr"] = safe_log1p(recharge)
    sample["log_distance_river_gt1_2020"] = safe_log1p(distance_1)
    sample["log_distance_river_gt0p1_2020"] = safe_log1p(distance_0p1)
    sample["wx_50km_log_glofas_p10_2020"] = wx_glofas

    for column in ["exclusion_class", "soil_group_class"]:
        sample[column] = sample[column].astype("category")

    base_features = [
        "elevation_m",
        "slope",
        "exclusion_class",
        "log_city_time_20k_min",
        "log_port_time_any_min",
        "rainfed_calorie_top5_raw",
        "irrigation_calorie_gain_top5_raw",
        "wx_50km_rainfed_calorie_top5_raw",
        "log_distance_river_gt10_2020",
        "log_glofas_p10_2020",
        "soil_group_class",
    ]
    if sample[base_features].isna().any().any():
        missing_counts = sample[base_features].isna().sum()
        raise RuntimeError(
            "Basic features contain missing values:\n"
            + missing_counts[missing_counts > 0].to_string()
        )

    metadata = {
        "n_rows": int(len(sample)),
        "fan_wtd_imputed_share": float(1.0 - finite_wtd.mean()),
        "fan_wtd_imputation_median_m": wtd_imputation,
        "watergap_imputed_share": float(1.0 - finite_recharge.mean()),
        "watergap_imputation_median_mm_yr": recharge_imputation,
    }
    (OUTPUT_DIR / "feature_alignment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Feature alignment:", metadata, flush=True)
    return sample, base_features, lat, lon


def calculate_target_prior():
    cropland_cube = np.load(
        require(CROPLAND_DIR / "cropland_fraction_1950_2024.npy"),
        mmap_mode="r",
    )
    years = np.load(require(CROPLAND_DIR / "years.npy"))
    year_index = int(np.where(years == YEAR)[0][0])
    cropland = np.asarray(cropland_cube[year_index], dtype=np.float32)
    population = np.load(
        require(FEATURE_CACHE / "population_density_2024.npy"),
        mmap_mode="r",
    )
    elevation = np.load(
        require(FEATURE_CACHE / "elevation_5min.npy"),
        mmap_mode="r",
    )
    exclusion = np.load(
        require(LAND_MASK_DIR / "gaez_v5_exclusion_5min_mode.npy"),
        mmap_mode="r",
    )
    soil = np.load(
        require(LAND_MASK_DIR / "gaez_v5_wrb_soil_group_5min_mode.npy"),
        mmap_mode="r",
    )
    land = (
        np.isfinite(elevation)
        & np.isfinite(cropland)
        & np.isfinite(population)
        & (population >= 0)
        & (exclusion >= 1)
        & (exclusion <= 6)
        & (exclusion != 5)
        & (soil >= 1)
        & (soil <= 33)
    )
    presence = land & (cropland > PRESENCE_THRESHOLD)
    target_prior = float(presence.sum() / land.sum())
    print("Population cropland-presence prior:", target_prior, flush=True)
    return target_prior


def adjust_probability_prior_shift(
    probability,
    sample_prior,
    population_prior,
    eps=1e-6,
):
    probability = np.clip(np.asarray(probability, dtype=float), eps, 1 - eps)
    sample_prior = float(np.clip(sample_prior, eps, 1 - eps))
    population_prior = float(np.clip(population_prior, eps, 1 - eps))
    odds = probability / (1.0 - probability)
    odds *= (
        (population_prior / (1.0 - population_prior))
        / (sample_prior / (1.0 - sample_prior))
    )
    return odds / (1.0 + odds)


def fit_oof_model(sample, features, splits, target_prior, model_name):
    checkpoint = CHECKPOINT_DIR / f"pred_{model_name}.npy"
    if checkpoint.exists():
        stored = np.load(checkpoint)
        if stored.shape == (len(sample),) and np.isfinite(stored).all():
            print("Reuse OOF checkpoint:", checkpoint.name, flush=True)
            return stored.astype(np.float32), []

    prediction = np.full(len(sample), np.nan, dtype=np.float32)
    importance_rows = []
    categorical = ["exclusion_class", "soil_group_class"]
    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        print(
            f"  {model_name}: fold {fold_number}/{N_SPLITS}",
            flush=True,
        )
        train = sample.iloc[train_index]
        test = sample.iloc[test_index]
        classifier = LGBMClassifier(
            objective="binary",
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED + fold_number,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        classifier.fit(
            train[features],
            train["presence"],
            categorical_feature=categorical,
        )
        positive_train = train[train["presence"].eq(1)]
        regressor = LGBMRegressor(
            objective="regression",
            n_estimators=550,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED + fold_number,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        regressor.fit(
            positive_train[features],
            positive_train["cropland_fraction"],
            categorical_feature=categorical,
        )
        raw_probability = classifier.predict_proba(test[features])[:, 1]
        probability = adjust_probability_prior_shift(
            raw_probability,
            float(train["presence"].mean()),
            target_prior,
        )
        conditional = np.clip(regressor.predict(test[features]), 0.0, 1.0)
        prediction[test_index] = (probability * conditional).astype(np.float32)

        for stage, model in [
            ("presence_classifier", classifier),
            ("conditional_fraction_regressor", regressor),
        ]:
            gain = np.asarray(model.booster_.feature_importance("gain"), dtype=float)
            gain_share = gain / gain.sum() if gain.sum() > 0 else gain
            for feature, value in zip(features, gain_share, strict=True):
                importance_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "stage": stage,
                        "feature": feature,
                        "gain_importance_share": float(value),
                    }
                )
        del classifier, regressor, raw_probability, probability, conditional
        gc.collect()

    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction: {model_name}")
    np.save(checkpoint, prediction)
    print("Saved OOF checkpoint:", checkpoint.name, flush=True)
    return prediction, importance_rows


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    residual = predicted - observed
    observed_mean = float(np.mean(observed))
    denominator = float(np.sum((observed - observed_mean) ** 2))
    return {
        "n_cells": int(len(observed)),
        "observed_mean": observed_mean,
        "predicted_mean": float(np.mean(predicted)),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": (
            1.0 - float(np.sum(residual**2)) / denominator
            if denominator > 0
            else np.nan
        ),
        "absolute_error_p95": float(np.percentile(np.abs(residual), 95)),
    }


def region_mask(frame, bounds):
    west, east, south, north = bounds
    return (
        frame["lon"].between(west, east).to_numpy()
        & frame["lat"].between(south, north).to_numpy()
    )


def summarize_predictions(sample, predictions):
    observed = sample["cropland_fraction"].to_numpy(dtype=float)
    global_rows = []
    regional_rows = []
    for model_name, prediction in predictions.items():
        global_rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                **regression_metrics(observed, prediction),
            }
        )
        for region_name, specification in REGIONS.items():
            selected = region_mask(sample, specification["bounds"])
            regional_rows.append(
                {
                    "region": region_name,
                    "region_label": specification["label"],
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    **regression_metrics(
                        observed[selected],
                        prediction[selected],
                    ),
                }
            )

    global_table = pd.DataFrame(global_rows)
    regional_table = pd.DataFrame(regional_rows)
    base_global = global_table.loc[
        global_table["model"].eq("baseline")
    ].iloc[0]
    global_table["rmse_improvement_vs_baseline"] = (
        float(base_global["rmse"]) - global_table["rmse"]
    )
    global_table["mae_improvement_vs_baseline"] = (
        float(base_global["mae"]) - global_table["mae"]
    )
    global_table["r2_gain_vs_baseline"] = (
        global_table["r2"] - float(base_global["r2"])
    )
    global_table["absolute_bias_improvement_vs_baseline"] = (
        abs(float(base_global["bias_predicted_minus_observed"]))
        - global_table["bias_predicted_minus_observed"].abs()
    )

    for region_name in REGIONS:
        selected = regional_table["region"].eq(region_name)
        baseline = regional_table.loc[
            selected & regional_table["model"].eq("baseline")
        ].iloc[0]
        regional_table.loc[selected, "rmse_improvement_vs_baseline"] = (
            float(baseline["rmse"])
            - regional_table.loc[selected, "rmse"]
        )
        regional_table.loc[selected, "mae_improvement_vs_baseline"] = (
            float(baseline["mae"])
            - regional_table.loc[selected, "mae"]
        )
        regional_table.loc[selected, "r2_gain_vs_baseline"] = (
            regional_table.loc[selected, "r2"] - float(baseline["r2"])
        )
        regional_table.loc[
            selected,
            "absolute_bias_improvement_vs_baseline",
        ] = (
            abs(float(baseline["bias_predicted_minus_observed"]))
            - regional_table.loc[
                selected,
                "bias_predicted_minus_observed",
            ].abs()
        )
    regional_table["rmse_improved"] = (
        regional_table["rmse_improvement_vs_baseline"] > 0
    )
    return global_table, regional_table


def choose_combined_features(base_features, regional_table):
    candidates = regional_table.loc[
        ~regional_table["model"].eq("baseline")
    ].groupby("model", as_index=False).agg(
        mean_regional_rmse_improvement=(
            "rmse_improvement_vs_baseline",
            "mean",
        ),
        n_regions_improved=("rmse_improved", "sum"),
    )
    lookup = candidates.set_index("model")
    selected_additions = []
    addition_map = {
        "add_fan_wtd": "log_fan_water_table_depth_m",
        "add_watergap_recharge": "log_watergap_total_recharge_mm_yr",
        "add_glofas_50km": "wx_50km_log_glofas_p10_2020",
    }
    for model_name, feature in addition_map.items():
        row = lookup.loc[model_name]
        if (
            row["mean_regional_rmse_improvement"] > 0
            and row["n_regions_improved"] >= 2
        ):
            selected_additions.append(feature)

    groundwater_pair = lookup.loc["add_groundwater_both"]
    if (
        groundwater_pair["mean_regional_rmse_improvement"] > 0
        and groundwater_pair["n_regions_improved"] >= 2
    ):
        pair_features = [
            "log_fan_water_table_depth_m",
            "log_watergap_total_recharge_mm_yr",
        ]
        best_single_groundwater = max(
            float(lookup.loc["add_fan_wtd", "mean_regional_rmse_improvement"]),
            float(
                lookup.loc[
                    "add_watergap_recharge",
                    "mean_regional_rmse_improvement",
                ]
            ),
        )
        if (
            groundwater_pair["mean_regional_rmse_improvement"]
            > best_single_groundwater
        ):
            for feature in pair_features:
                if feature not in selected_additions:
                    selected_additions.append(feature)

    river_options = ["baseline", "river_threshold_1", "river_threshold_0p1"]
    river_scores = {"baseline": 0.0}
    for model_name in river_options[1:]:
        river_scores[model_name] = float(
            lookup.loc[model_name, "mean_regional_rmse_improvement"]
        )
    selected_river = max(river_scores, key=river_scores.get)

    combined = list(base_features)
    if selected_river != "baseline" and river_scores[selected_river] > 0:
        replacement = {
            "river_threshold_1": "log_distance_river_gt1_2020",
            "river_threshold_0p1": "log_distance_river_gt0p1_2020",
        }[selected_river]
        combined = [
            replacement
            if feature == "log_distance_river_gt10_2020"
            else feature
            for feature in combined
        ]
    else:
        selected_river = "baseline"

    for feature in selected_additions:
        if feature not in combined:
            combined.append(feature)

    selection = {
        "selection_note": (
            "Observed irrigation-area variables, including GMIA, were excluded. "
            "Exploratory post-hoc package. Additive variables were retained "
            "when mean regional RMSE improved and at least 2 of 4 regions "
            "improved. The river threshold with the best mean regional RMSE "
            "was retained only when it beat the 10 m3/s baseline."
        ),
        "selected_river_variant": selected_river,
        "selected_additional_features": selected_additions,
        "combined_features": combined,
    }
    (OUTPUT_DIR / "combined_model_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Combined-model selection:", selection, flush=True)
    return combined


def block_bootstrap_table(sample, predictions, regional_table):
    observed_all = sample["cropland_fraction"].to_numpy(dtype=float)
    rows = []
    for region_index, (region_name, specification) in enumerate(REGIONS.items()):
        selected = region_mask(sample, specification["bounds"])
        observed = observed_all[selected]
        blocks = sample.loc[selected, "spatial_block"].to_numpy()
        unique_blocks = np.unique(blocks)
        baseline = predictions["baseline"][selected]
        for model_index, (model_name, prediction_all) in enumerate(
            predictions.items()
        ):
            if model_name == "baseline":
                rows.append(
                    {
                        "region": region_name,
                        "model": model_name,
                        "n_spatial_blocks": int(len(unique_blocks)),
                        "delta_rmse_model_minus_baseline": 0.0,
                        "delta_rmse_ci_low": 0.0,
                        "delta_rmse_ci_high": 0.0,
                        "probability_lower_rmse": np.nan,
                    }
                )
                continue
            prediction = prediction_all[selected]
            block_stats = []
            for block in unique_blocks:
                in_block = blocks == block
                block_stats.append(
                    (
                        int(in_block.sum()),
                        float(np.sum((observed[in_block] - prediction[in_block]) ** 2)),
                        float(np.sum((observed[in_block] - baseline[in_block]) ** 2)),
                    )
                )
            block_stats = np.asarray(block_stats, dtype=float)
            rng = np.random.default_rng(
                RANDOM_SEED + 10000 + 100 * region_index + model_index
            )
            delta = np.empty(BOOTSTRAP_REPS, dtype=float)
            for repetition in range(BOOTSTRAP_REPS):
                draw = rng.integers(
                    0,
                    len(unique_blocks),
                    size=len(unique_blocks),
                )
                n = block_stats[draw, 0].sum()
                model_rmse = np.sqrt(block_stats[draw, 1].sum() / n)
                baseline_rmse = np.sqrt(block_stats[draw, 2].sum() / n)
                delta[repetition] = model_rmse - baseline_rmse
            rows.append(
                {
                    "region": region_name,
                    "model": model_name,
                    "n_spatial_blocks": int(len(unique_blocks)),
                    "delta_rmse_model_minus_baseline": float(delta.mean()),
                    "delta_rmse_ci_low": float(np.quantile(delta, 0.025)),
                    "delta_rmse_ci_high": float(np.quantile(delta, 0.975)),
                    "probability_lower_rmse": float(np.mean(delta < 0)),
                }
            )
    bootstrap = pd.DataFrame(rows)
    result = regional_table.merge(
        bootstrap,
        on=["region", "model"],
        how="left",
        validate="one_to_one",
    )
    result["clear_improvement_95pct"] = (
        result["delta_rmse_ci_high"] < 0
    )
    result["improvement_class"] = np.select(
        [
            result["model"].eq("baseline"),
            result["clear_improvement_95pct"],
            result["probability_lower_rmse"].ge(0.80),
            result["rmse_improved"],
        ],
        [
            "reference",
            "clear_improvement",
            "likely_improvement",
            "point_estimate_improvement",
        ],
        default="no_improvement",
    )
    return result


def global_block_bootstrap_contrasts(sample, predictions):
    observed = sample["cropland_fraction"].to_numpy(dtype=float)
    blocks = sample["spatial_block"].to_numpy()
    unique_blocks = np.unique(blocks)
    contrasts = [
        ("water_gw_t10_vs_basic", "water_gw_t10", "baseline"),
        ("point_best_vs_basic", "water_gw_t10_wx50", "baseline"),
        ("wx50_effect_at_10", "water_gw_t10_wx50", "water_gw_t10"),
        ("wx50_effect_at_1", "water_gw_t1_wx50", "water_gw_t1"),
        ("wx50_effect_at_0p1", "water_gw_t0p1_wx50", "water_gw_t0p1"),
        ("threshold_1_vs_10_no_wx", "water_gw_t1", "water_gw_t10"),
        ("threshold_0p1_vs_10_no_wx", "water_gw_t0p1", "water_gw_t10"),
        (
            "threshold_1_vs_10_with_wx",
            "water_gw_t1_wx50",
            "water_gw_t10_wx50",
        ),
        (
            "threshold_0p1_vs_10_with_wx",
            "water_gw_t0p1_wx50",
            "water_gw_t10_wx50",
        ),
    ]
    rows = []
    for contrast_index, (name, candidate, reference) in enumerate(contrasts):
        candidate_prediction = predictions[candidate]
        reference_prediction = predictions[reference]
        block_stats = []
        for block in unique_blocks:
            in_block = blocks == block
            block_stats.append(
                (
                    int(in_block.sum()),
                    float(
                        np.sum(
                            (observed[in_block] - candidate_prediction[in_block])
                            ** 2
                        )
                    ),
                    float(
                        np.sum(
                            (observed[in_block] - reference_prediction[in_block])
                            ** 2
                        )
                    ),
                )
            )
        block_stats = np.asarray(block_stats, dtype=float)
        n_all = block_stats[:, 0].sum()
        point_delta = (
            np.sqrt(block_stats[:, 1].sum() / n_all)
            - np.sqrt(block_stats[:, 2].sum() / n_all)
        )
        rng = np.random.default_rng(
            RANDOM_SEED + 30000 + contrast_index
        )
        delta = np.empty(BOOTSTRAP_REPS, dtype=float)
        for repetition in range(BOOTSTRAP_REPS):
            draw = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
            n = block_stats[draw, 0].sum()
            delta[repetition] = (
                np.sqrt(block_stats[draw, 1].sum() / n)
                - np.sqrt(block_stats[draw, 2].sum() / n)
            )
        rows.append(
            {
                "contrast": name,
                "candidate": candidate,
                "reference": reference,
                "delta_rmse_candidate_minus_reference": float(point_delta),
                "bootstrap_mean_delta": float(delta.mean()),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "probability_candidate_lower_rmse": float(np.mean(delta < 0)),
                "n_spatial_blocks": int(len(unique_blocks)),
            }
        )
    return pd.DataFrame(rows)


def save_tables(sample, predictions, global_table, regional_table, importance_rows):
    global_table.to_csv(
        OUTPUT_DIR / "global_model_error_comparison_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regional_table.to_csv(
        OUTPUT_DIR / "regional_model_error_comparison_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking = (
        regional_table.loc[~regional_table["model"].eq("baseline")]
        .sort_values(["region", "rmse", "model"])
        .copy()
    )
    ranking["rank_within_region"] = ranking.groupby("region")["rmse"].rank(
        method="first"
    ).astype(int)
    ranking.to_csv(
        OUTPUT_DIR / "which_variable_improved_by_region_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best = ranking.loc[ranking["rank_within_region"].eq(1)].copy()
    best.to_csv(
        OUTPUT_DIR / "best_variant_by_region_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    supported = best["improvement_class"].isin(
        {"clear_improvement", "likely_improvement"}
    )
    recommended = pd.DataFrame(
        {
            "region": best["region"],
            "region_label": best["region_label"],
            "recommended_model": np.where(
                supported, best["model"], "baseline"
            ),
            "recommended_variables": [
                " | ".join(MODEL_CHANGE_LINES[model])
                if is_supported
                else "None; retain basic model"
                for model, is_supported in zip(
                    best["model"], supported, strict=True
                )
            ],
            "closest_tested_candidate": best["model"],
            "closest_candidate_label": best["model_label"],
            "rmse_improvement_vs_baseline": (
                best["rmse_improvement_vs_baseline"]
            ),
            "r2_gain_vs_baseline": best["r2_gain_vs_baseline"],
            "improvement_class": best["improvement_class"],
            "probability_lower_rmse": best["probability_lower_rmse"],
        }
    )
    recommended.to_csv(
        OUTPUT_DIR / "recommended_water_model_by_region_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    strictly_individual_models = {
        "add_fan_wtd",
        "add_watergap_recharge",
        "river_threshold_1",
        "river_threshold_0p1",
        "add_glofas_50km",
    }
    best_individual = (
        ranking.loc[ranking["model"].isin(strictly_individual_models)]
        .sort_values(["region", "rmse", "model"])
        .groupby("region", as_index=False)
        .first()
    )
    best_individual.to_csv(
        OUTPUT_DIR / "best_individual_change_by_region_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance = pd.DataFrame(importance_rows)
    if not importance.empty:
        importance.to_csv(
            OUTPUT_DIR / "candidate_model_feature_gain_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    prediction_table = sample[
        [
            "row",
            "col",
            "lat",
            "lon",
            "spatial_block",
            "presence",
            "cropland_fraction",
        ]
    ].copy()
    for model_name, prediction in predictions.items():
        prediction_table[f"pred_{model_name}"] = prediction
    prediction_table.to_csv(
        OUTPUT_DIR / "water_irrigation_sensitivity_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    print("Saved comparison tables and OOF predictions.", flush=True)


def plot_error_change_heatmaps(regional_table):
    model_order = [
        model
        for model in MODEL_LABELS
        if model != "baseline" and model in set(regional_table["model"])
    ]
    region_order = list(REGIONS)
    region_labels = [REGIONS[name]["label"] for name in region_order]
    rmse = (
        regional_table.pivot(
            index="model",
            columns="region",
            values="rmse_improvement_vs_baseline",
        )
        .reindex(index=model_order, columns=region_order)
    )
    r2 = (
        regional_table.pivot(
            index="model",
            columns="region",
            values="r2_gain_vs_baseline",
        )
        .reindex(index=model_order, columns=region_order)
    )
    clear = (
        regional_table.pivot(
            index="model",
            columns="region",
            values="clear_improvement_95pct",
        )
        .reindex(index=model_order, columns=region_order)
        .fillna(False)
    )
    labels = [MODEL_LABELS[name] for name in model_order]
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)
    panels = [
        (axes[0], rmse, "RMSE improvement (baseline − candidate)", "RMSE"),
        (axes[1], r2, "R² gain (candidate − baseline)", "R²"),
    ]
    for ax, table, title, metric_name in panels:
        values = table.to_numpy(dtype=float)
        limit = max(float(np.nanmax(np.abs(values))), 1e-6)
        image = ax.imshow(
            values,
            cmap="RdYlGn",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        ax.set_xticks(range(len(region_labels)), region_labels, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                star = "★" if bool(clear.iloc[row, col]) and metric_name == "RMSE" else ""
                ax.text(
                    col,
                    row,
                    f"{values[row, col]:+.4f}{star}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.suptitle(
        "Regional Spatial OOF error changes after adding or relaxing water variables\n"
        "Green/positive = improvement; ★ = 95% spatial-block bootstrap interval is below zero",
        fontsize=16,
    )
    output = OUTPUT_DIR / "regional_error_change_by_model_2020.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", output, flush=True)


def grid_from_sample(sample, values, bounds):
    west, east, south, north = bounds
    nx = int(round((east - west) / TARGET_RESOLUTION))
    ny = int(round((north - south) / TARGET_RESOLUTION))
    x_edges = np.linspace(west, east, nx + 1)
    y_edges = np.linspace(south, north, ny + 1)
    selected = region_mask(sample, bounds)
    frame = sample.loc[selected]
    selected_values = np.asarray(values)[selected]
    columns = np.rint(
        (frame["lon"].to_numpy() - (west + TARGET_RESOLUTION / 2))
        / TARGET_RESOLUTION
    ).astype(int)
    rows = np.rint(
        (frame["lat"].to_numpy() - (south + TARGET_RESOLUTION / 2))
        / TARGET_RESOLUTION
    ).astype(int)
    valid = (columns >= 0) & (columns < nx) & (rows >= 0) & (rows < ny)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    grid[rows[valid], columns[valid]] = selected_values[valid]
    return x_edges, y_edges, np.ma.masked_invalid(grid)


def plot_best_regional_maps(
    sample,
    predictions,
    regional_table,
    region_names=None,
    fixed_model=None,
):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    observed = sample["cropland_fraction"].to_numpy(dtype=float)
    candidates = regional_table.loc[~regional_table["model"].eq("baseline")]
    for region_name, specification in REGIONS.items():
        if region_names is not None and region_name not in region_names:
            continue
        region_candidates = candidates.loc[
            candidates["region"].eq(region_name)
        ].sort_values("rmse")
        if fixed_model is None:
            best = region_candidates.iloc[0]
        else:
            selected_rows = region_candidates.loc[
                region_candidates["model"].eq(fixed_model)
            ]
            if len(selected_rows) != 1:
                raise KeyError(
                    f"Expected one row for {region_name=} and {fixed_model=}; "
                    f"found {len(selected_rows)}"
                )
            best = selected_rows.iloc[0]
        best_model = str(best["model"])
        map_model_label = {
            "combined_selected_no_irrigation": (
                "Selected water package; observed irrigation excluded"
            ),
            "add_fan_wtd": "Fan water-table depth",
            "add_watergap_recharge": "WaterGAP groundwater recharge",
            "add_groundwater_both": "WTD and groundwater recharge",
            "river_threshold_1": "1 m³/s river threshold",
            "river_threshold_0p1": "0.1 m³/s river threshold",
            "add_glofas_50km": "50-km surrounding GloFAS flow",
        }.get(best_model, MODEL_LABELS[best_model])
        model_change_lines = MODEL_CHANGE_LINES[best_model]
        model_change_text = "\n".join(model_change_lines)
        baseline_residual = predictions["baseline"] - observed
        candidate_residual = predictions[best_model] - observed
        absolute_error_improvement = (
            np.abs(baseline_residual) - np.abs(candidate_residual)
        )
        selected = region_mask(sample, specification["bounds"])
        residual_limit = max(
            float(
                np.nanpercentile(
                    np.abs(
                        np.concatenate(
                            [
                                baseline_residual[selected],
                                candidate_residual[selected],
                            ]
                        )
                    ),
                    99,
                )
            ),
            0.05,
        )
        improvement_limit = max(
            float(
                np.nanpercentile(
                    np.abs(absolute_error_improvement[selected]),
                    99,
                )
            ),
            0.01,
        )
        figure_size = (13.5, 10.0) if region_name == "nile_region" else (19, 9.0)
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figure_size,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        panels = [
            (
                baseline_residual,
                "A. Basic model residual (predicted − observed)\n"
                "Blue: underprediction  |  Red: overprediction",
                "RdBu_r",
                Normalize(-residual_limit, residual_limit),
                "Prediction error (predicted − observed)",
            ),
            (
                candidate_residual,
                "B. Tested-candidate residual (candidate named above)\n"
                "Blue: underprediction  |  Red: overprediction",
                "RdBu_r",
                Normalize(-residual_limit, residual_limit),
                "Prediction error (predicted − observed)",
            ),
            (
                absolute_error_improvement,
                "C. Change in absolute error (basic − candidate)\n"
                "Green: improved  |  Red: worsened",
                "RdYlGn",
                Normalize(-improvement_limit, improvement_limit),
                "Absolute-error improvement",
            ),
        ]
        for ax, (
            values,
            panel_caption,
            cmap,
            norm,
            colorbar_label,
        ) in zip(axes, panels, strict=True):
            plot_bounds = specification.get(
                "plot_bounds",
                specification["bounds"],
            )
            west, east, south, north = plot_bounds
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            ax.set_facecolor("#eaf2f6")
            ax.add_feature(
                cfeature.LAND.with_scale("50m"),
                facecolor="#faf9f2",
                edgecolor="none",
                zorder=0,
            )
            ax.add_feature(
                cfeature.OCEAN.with_scale("50m"),
                facecolor="#eaf2f6",
                edgecolor="none",
                zorder=0,
            )
            x_edges, y_edges, grid = grid_from_sample(
                sample,
                values,
                plot_bounds,
            )
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                grid,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading="flat",
                antialiased=False,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            if region_name == "nile_region":
                ax.add_feature(
                    cfeature.RIVERS.with_scale("50m"),
                    edgecolor="#58727c",
                    linewidth=0.55,
                    alpha=0.85,
                    zorder=3,
                )
            ax.add_feature(
                cfeature.BORDERS.with_scale("50m"),
                edgecolor="#999999",
                linewidth=0.4,
                zorder=3,
            )
            ax.coastlines(resolution="50m", color="#555555", linewidth=0.65, zorder=4)
            gridlines = ax.gridlines(
                draw_labels=True,
                linewidth=0.35,
                color="#8b979c",
                alpha=0.35,
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.xlabel_style = {"size": 8}
            gridlines.ylabel_style = {"size": 8}
            ax.text(
                0.5,
                1.045,
                panel_caption,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                linespacing=1.35,
                clip_on=False,
                zorder=20,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#b0b0b0",
                    "alpha": 0.96,
                },
            )
            colorbar = fig.colorbar(
                mesh,
                ax=ax,
                orientation="horizontal",
                pad=0.075,
                fraction=0.055,
            )
            colorbar.ax.tick_params(labelsize=8)
            colorbar.set_label(colorbar_label, fontsize=9, labelpad=4)

        baseline_rmse = (
            float(best["rmse"])
            + float(best["rmse_improvement_vs_baseline"])
        )
        rmse_reduction_pct = (
            100.0
            * float(best["rmse_improvement_vs_baseline"])
            / baseline_rmse
        )
        classification_label = {
            "clear_improvement": "clear improvement (95% block-bootstrap)",
            "likely_improvement": "likely improvement",
            "point_estimate_improvement": "point-estimate improvement",
            "no_improvement": "no clear improvement",
        }.get(str(best["improvement_class"]), str(best["improvement_class"]))
        if fixed_model is not None:
            if best["improvement_class"] in {
                "clear_improvement",
                "likely_improvement",
            }:
                recommendation_line = (
                    "Globally recommended specification; also improves this region: "
                    f"{map_model_label}"
                )
            elif best["improvement_class"] == "point_estimate_improvement":
                recommendation_line = (
                    "Globally recommended specification; regional gain is unsupported: "
                    f"{map_model_label}"
                )
            else:
                recommendation_line = (
                    "Globally recommended specification; no improvement in this region: "
                    f"{map_model_label}"
                )
        elif best["improvement_class"] in {
            "clear_improvement",
            "likely_improvement",
        }:
            recommendation_line = f"Recommended candidate: {map_model_label}"
        elif best["improvement_class"] == "point_estimate_improvement":
            recommendation_line = (
                "Retain basic model (no supported gain); closest candidate shown: "
                f"{map_model_label}"
            )
        else:
            recommendation_line = (
                "Retain basic model (all tested candidates failed to improve RMSE); "
                f"closest candidate shown: {map_model_label}"
            )
        fig.suptitle(
            f"{specification.get('plot_label', specification['label'])} "
            "— Spatial OOF error comparison\n"
            f"{recommendation_line}\n"
            f"{model_change_text}\n"
            f"RMSE {baseline_rmse:.4f} → {best['rmse']:.4f} "
            f"({rmse_reduction_pct:.1f}% lower)  |  "
            f"R² gain {best['r2_gain_vs_baseline']:+.4f}  |  "
            f"{classification_label}",
            fontsize=(11.5 if len(model_change_lines) > 2 else 12.5),
            y=0.985,
        )
        layout = (
            {"left": 0.035, "right": 0.99, "top": 0.72, "bottom": 0.12}
            if region_name == "nile_region"
            else {"left": 0.035, "right": 0.99, "top": 0.70, "bottom": 0.13}
        )
        fig.subplots_adjust(
            **layout,
            wspace=0.09,
        )
        footer = (
            "Each colored square is one sampled 5-arc-minute grid cell. "
            "Residuals use the same color scale in panels A and B."
        )
        if region_name == "nile_region":
            footer += (
                "\nOnly the map extent is zoomed; title metrics retain the "
                "predefined Nile/Northeast Africa evaluation region."
            )
        fig.text(
            0.5,
            0.02,
            footer,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
        output_suffix = (
            "global_selected_water_specification"
            if fixed_model is not None
            else "best_variant"
        )
        output = MAP_DIR / f"{region_name}_{output_suffix}_error_change_grid.png"
        fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("Saved:", output, flush=True)


def main():
    sample, base_features, _, _ = load_analysis_sample()
    target_prior = calculate_target_prior()
    groups = sample["spatial_block"].to_numpy()
    y_presence = sample["presence"].to_numpy(dtype=np.uint8)
    splits = list(
        GroupKFold(n_splits=N_SPLITS).split(
            sample,
            y_presence,
            groups=groups,
        )
    )

    fixed_groundwater_features = [
        "log_fan_water_table_depth_m",
        "log_watergap_total_recharge_mm_yr",
    ]
    features_by_model = OrderedDict()
    for model_name, specification in WATER_FACTORIAL_SPECS.items():
        river_feature = specification["river_feature"]
        features = [
            river_feature
            if feature == "log_distance_river_gt10_2020"
            else feature
            for feature in base_features
        ]
        features.extend(fixed_groundwater_features)
        if specification["add_wx50"]:
            features.append("wx_50km_log_glofas_p10_2020")
        features_by_model[model_name] = features

    definitions = []
    for model_name, features in features_by_model.items():
        definitions.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "n_features": len(features),
                "features": " | ".join(features),
            }
        )
    pd.DataFrame(definitions).to_csv(
        OUTPUT_DIR / "candidate_model_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions = OrderedDict(
        baseline=sample[
            "pred_soil_group_wetland_excluded"
        ].to_numpy(dtype=np.float32)
    )
    importance_rows = []
    for model_number, (model_name, features) in enumerate(
        features_by_model.items(),
        start=1,
    ):
        print(
            "=" * 90,
            f"\nMODEL {model_number}/{len(features_by_model)}: "
            f"{model_name} ({len(features)} features)",
            flush=True,
        )
        prediction, model_importance = fit_oof_model(
            sample,
            features,
            splits,
            target_prior,
            model_name,
        )
        predictions[model_name] = prediction
        importance_rows.extend(model_importance)

    global_table, regional_table = summarize_predictions(sample, predictions)
    global_candidates = (
        global_table.loc[~global_table["model"].eq("baseline")]
        .sort_values(["rmse", "model"])
        .reset_index(drop=True)
    )
    global_candidates["global_rmse_rank"] = np.arange(
        1,
        len(global_candidates) + 1,
    )
    global_candidates.to_csv(
        OUTPUT_DIR / "global_water_factorial_comparison_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_bootstrap = global_block_bootstrap_contrasts(sample, predictions)
    global_bootstrap.to_csv(
        OUTPUT_DIR / "global_water_specification_block_bootstrap_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )

    point_best = global_candidates.iloc[0]
    threshold_choice = "10"
    threshold_model = "water_gw_t10"
    for candidate_threshold, candidate_model, contrast_name in [
        ("1", "water_gw_t1", "threshold_1_vs_10_no_wx"),
        ("0.1", "water_gw_t0p1", "threshold_0p1_vs_10_no_wx"),
    ]:
        candidate_row = global_candidates.loc[
            global_candidates["model"].eq(candidate_model)
        ].iloc[0]
        contrast_row = global_bootstrap.loc[
            global_bootstrap["contrast"].eq(contrast_name)
        ].iloc[0]
        current_row = global_candidates.loc[
            global_candidates["model"].eq(threshold_model)
        ].iloc[0]
        if (
            candidate_row["rmse"] < current_row["rmse"]
            and contrast_row["ci_high"] < 0
        ):
            threshold_choice = candidate_threshold
            threshold_model = candidate_model

    wx_model_by_threshold = {
        "10": "water_gw_t10_wx50",
        "1": "water_gw_t1_wx50",
        "0.1": "water_gw_t0p1_wx50",
    }
    wx_contrast_by_threshold = {
        "10": "wx50_effect_at_10",
        "1": "wx50_effect_at_1",
        "0.1": "wx50_effect_at_0p1",
    }
    wx_model = wx_model_by_threshold[threshold_choice]
    wx_row = global_candidates.loc[
        global_candidates["model"].eq(wx_model)
    ].iloc[0]
    no_wx_row = global_candidates.loc[
        global_candidates["model"].eq(threshold_model)
    ].iloc[0]
    wx_contrast = global_bootstrap.loc[
        global_bootstrap["contrast"].eq(
            wx_contrast_by_threshold[threshold_choice]
        )
    ].iloc[0]
    add_wx50 = bool(
        wx_row["rmse"] < no_wx_row["rmse"]
        and wx_contrast["ci_high"] < 0
    )
    best_global_model = wx_model if add_wx50 else threshold_model
    selected_global = global_candidates.loc[
        global_candidates["model"].eq(best_global_model)
    ].iloc[0]
    selection = {
        "selection_basis": (
            "Fan water-table depth and WaterGAP recharge are fixed in all "
            "candidates. Keep the existing 10 m3/s threshold unless another "
            "threshold has lower global Spatial OOF RMSE and its paired "
            "spatial-block bootstrap 95% interval is entirely below zero. "
            "Add the 50-km GloFAS feature only under the same evidence rule."
        ),
        "point_estimate_best_model": str(point_best["model"]),
        "point_estimate_best_rmse": float(point_best["rmse"]),
        "recommended_model": best_global_model,
        "recommended_label": MODEL_LABELS[best_global_model],
        "recommended_features": features_by_model[best_global_model],
        "selected_river_threshold_m3s": threshold_choice,
        "add_surrounding_50km_glofas": add_wx50,
        "global_rmse": float(selected_global["rmse"]),
        "global_rmse_improvement_vs_basic": float(
            selected_global["rmse_improvement_vs_baseline"]
        ),
        "global_r2_gain_vs_basic": float(
            selected_global["r2_gain_vs_baseline"]
        ),
        "wx50_delta_rmse_at_selected_threshold": float(
            wx_contrast["delta_rmse_candidate_minus_reference"]
        ),
        "wx50_probability_lower_rmse": float(
            wx_contrast["probability_candidate_lower_rmse"]
        ),
        "wx50_ci_low": float(wx_contrast["ci_low"]),
        "wx50_ci_high": float(wx_contrast["ci_high"]),
    }
    (OUTPUT_DIR / "global_water_specification_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    regional_table = block_bootstrap_table(
        sample,
        predictions,
        regional_table,
    )
    save_tables(
        sample,
        predictions,
        global_table,
        regional_table,
        importance_rows,
    )
    plot_error_change_heatmaps(regional_table)
    plot_best_regional_maps(
        sample,
        predictions,
        regional_table,
        fixed_model=best_global_model,
    )

    print("\nGLOBAL COMPARISON", flush=True)
    print(
        global_table[
            [
                "model",
                "r2",
                "rmse",
                "mae",
                "rmse_improvement_vs_baseline",
                "r2_gain_vs_baseline",
            ]
        ].round(6).to_string(index=False),
        flush=True,
    )
    print("\nGLOBALLY SELECTED WATER SPECIFICATION", flush=True)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    selected_regional = regional_table.loc[
        regional_table["model"].eq(best_global_model),
        [
            "region",
            "rmse_improvement_vs_baseline",
            "r2_gain_vs_baseline",
            "improvement_class",
        ],
    ]
    selected_regional.to_csv(
        OUTPUT_DIR / "global_selected_water_model_regional_performance_2020.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nREGIONAL PERFORMANCE OF GLOBAL SELECTION", flush=True)
    print(selected_regional.round(6).to_string(index=False), flush=True)
    print("\nCompleted. Output:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
