import json
from pathlib import Path

source = Path(r"C:\Users\tsuda\Downloads\cropland_soil_group_wetland_excluded_oof_shap_target2020.ipynb")
output = Path(r"C:\Users\tsuda\Downloads\cropland_soil_group_wetland_excluded_oof_shap_target2020_nutrients_gdp.ipynb")

with source.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

notebook["cells"] = notebook["cells"][:14]

notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 窒素・リン・GDP追加モデルのSpatial OOF分析\n",
        "\n",
        "target2020の既存準備コードを使い、窒素・リン・GDPを追加したモデルをbaselineと比較する。\n",
    ],
})

analysis_cell = r'''import gc
import shutil
import zipfile
from matplotlib.colors import TwoSlopeNorm

INPUT_DIR_NPGDP = Path(
    "/work/tsuda/GAEZ/CroplandRegression/prior_study_inputs"
)
DERIVED_DIR_NPGDP = INPUT_DIR_NPGDP / "derived_5arcmin_target2020"
OUTPUT_DIR_NPGDP = Path(
    "/work/tsuda/GAEZ/CroplandRegression/soil_group_nutrients_gdp_target2020"
)

DERIVED_DIR_NPGDP.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_NPGDP.mkdir(parents=True, exist_ok=True)

REBUILD_GRID_CACHE_NPGDP = False
GDP_BAND_2020 = 31
NITROGEN_BAND = 1
PHOSPHORUS_BAND = 1
MAP_MAX_POINTS = 120_000


def extract_zip(path):
    out = INPUT_DIR_NPGDP / "_extracted_nutrients" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    if not any(item.is_file() for item in out.rglob("*")):
        with zipfile.ZipFile(path) as archive:
            root = out.resolve()
            for member in archive.infolist():
                target = (out / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"不正なZIPパス: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return out


def find_sedac_raster(prefix):
    extensions = {".tif", ".tiff", ".vrt", ".img", ".asc", ".nc"}
    candidates = sorted(
        item for item in INPUT_DIR_NPGDP.iterdir()
        if item.is_file() and item.name.lower().startswith(prefix.lower())
    )
    for path in candidates:
        if path.suffix.lower() in extensions:
            return path
        if path.suffix.lower() == ".zip":
            extracted = extract_zip(path)
            inside = sorted(
                item for item in extracted.rglob("*")
                if item.is_file() and item.suffix.lower() in extensions
            )
            if inside:
                return inside[0]
    raise FileNotFoundError(
        f"{prefix}のラスタが見つかりません: {INPUT_DIR_NPGDP}"
    )


def target_grid():
    target_lat = np.asarray(lat, dtype=float)
    target_lon = np.asarray(lon, dtype=float)
    dlat = float(np.nanmedian(np.abs(np.diff(target_lat))))
    dlon = float(np.nanmedian(np.abs(np.diff(target_lon))))
    transform = from_origin(
        float(np.nanmin(target_lon) - dlon / 2),
        float(np.nanmax(target_lat) + dlat / 2),
        dlon,
        dlat,
    )
    return target_lat, target_lon, transform, (len(target_lat), len(target_lon))


def make_grid(source_path, band, cache_path):
    target_lat, target_lon, target_transform, target_shape = target_grid()

    if cache_path.exists() and not REBUILD_GRID_CACHE_NPGDP:
        array = np.asarray(np.load(cache_path), dtype=np.float32)
        if tuple(array.shape) != tuple(target_shape):
            raise ValueError(
                f"キャッシュshapeが不一致です: {array.shape} != {target_shape}"
            )
        print("既存キャッシュを使用:", cache_path)
        return array

    with rasterio.open(source_path) as source:
        print(
            "source:", source_path,
            "shape:", (source.height, source.width),
            "bands:", source.count,
            "crs:", source.crs,
            "bounds:", source.bounds,
        )
        array = source.read(band).astype(np.float32)
        source_transform = source.transform
        source_crs = source.crs or "EPSG:4326"
        source_nodata = source.nodata

    destination = np.full(target_shape, np.nan, dtype=np.float32)

    reproject(
        source=array,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=source_nodata,
        dst_transform=target_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    if target_lat[0] < target_lat[-1]:
        destination = destination[::-1, :]
    if target_lon[0] > target_lon[-1]:
        destination = destination[:, ::-1]

    destination[~np.isfinite(destination)] = np.nan
    destination[destination < 0] = np.nan
    np.save(cache_path, destination)
    print("5分グリッドキャッシュを保存:", cache_path)
    return destination


gdp_source = INPUT_DIR_NPGDP / "gdp_pc_ppp_2017usd_5arcmin_1990_2022.tif"
gdp_cache = DERIVED_DIR_NPGDP / "gdp_pc_2020_ppp2017usd_5arcmin.npy"
nitrogen_cache = DERIVED_DIR_NPGDP / "nitrogen_fertilizer_kg_ha_5arcmin.npy"
phosphorus_cache = DERIVED_DIR_NPGDP / "phosphorus_fertilizer_kg_ha_5arcmin.npy"

if gdp_cache.exists() and not REBUILD_GRID_CACHE_NPGDP:
    gdp_2020 = make_grid(gdp_cache, GDP_BAND_2020, gdp_cache)
else:
    if not gdp_source.exists():
        raise FileNotFoundError(gdp_source)
    gdp_2020 = make_grid(gdp_source, GDP_BAND_2020, gdp_cache)

if nitrogen_cache.exists() and not REBUILD_GRID_CACHE_NPGDP:
    nitrogen_kg_ha = make_grid(nitrogen_cache, NITROGEN_BAND, nitrogen_cache)
else:
    nitrogen_source = find_sedac_raster("nitrogen_fertilizer__")
    nitrogen_kg_ha = make_grid(
        nitrogen_source,
        NITROGEN_BAND,
        nitrogen_cache,
    )

if phosphorus_cache.exists() and not REBUILD_GRID_CACHE_NPGDP:
    phosphorus_kg_ha = make_grid(
        phosphorus_cache,
        PHOSPHORUS_BAND,
        phosphorus_cache,
    )
else:
    phosphorus_source = find_sedac_raster("phosphorus_fertilizer__")
    phosphorus_kg_ha = make_grid(
        phosphorus_source,
        PHOSPHORUS_BAND,
        phosphorus_cache,
    )


def log_nonnegative(values):
    values = np.asarray(values, dtype=float)
    values = np.where(
        np.isfinite(values) & (values >= 0),
        values,
        np.nan,
    )
    return np.log1p(values).astype(np.float32)


work = sample.copy()
sample_rows = work["row"].to_numpy(dtype=int)
sample_cols = work["col"].to_numpy(dtype=int)

work["log_nitrogen_fertilizer_kg_ha"] = log_nonnegative(
    nitrogen_kg_ha[sample_rows, sample_cols]
)
work["log_phosphorus_fertilizer_kg_ha"] = log_nonnegative(
    phosphorus_kg_ha[sample_rows, sample_cols]
)
work["log_gdp_pc_2020_ppp2017usd"] = log_nonnegative(
    gdp_2020[sample_rows, sample_cols]
)

baseline_features = list(
    MODEL_FEATURES["soil_group_wetland_excluded"]
)
added_features = [
    "log_nitrogen_fertilizer_kg_ha",
    "log_phosphorus_fertilizer_kg_ha",
    "log_gdp_pc_2020_ppp2017usd",
]
augmented_features = baseline_features + added_features

model_specs = {
    "baseline": baseline_features,
    "plus_nutrients_gdp": augmented_features,
}

needed_columns = list(dict.fromkeys(
    baseline_features
    + added_features
    + ["cropland_fraction", "presence", "spatial_block"]
))

valid_rows = np.ones(len(work), dtype=bool)
for column in needed_columns:
    if column in CATEGORICAL_CANDIDATES:
        valid_rows &= work[column].notna().to_numpy()
    else:
        numeric = pd.to_numeric(
            work[column],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid_rows &= np.isfinite(numeric)

work = work.loc[valid_rows].reset_index(drop=True)

for column in CATEGORICAL_CANDIDATES:
    if column in work.columns:
        work[column] = work[column].astype("category")

if len(work) == 0 or work["presence"].nunique() != 2:
    raise RuntimeError(
        "追加変数を含む完全ケースにpresenceの両クラスがありません。"
    )

groups = work["spatial_block"].to_numpy()
y_presence = work["presence"].to_numpy(dtype=np.uint8)

oof_splits = list(
    GroupKFold(n_splits=N_SPLITS).split(
        work,
        y_presence,
        groups=groups,
    )
)

if "area_weight" in globals() and area_weight is not None:
    original_area_weight = np.asarray(area_weight, dtype=float)
    if len(original_area_weight) != len(valid_rows):
        raise ValueError("area_weightとsampleの行数が一致しません。")
    analysis_area_weight = original_area_weight[valid_rows]
    analysis_area_weight[~np.isfinite(analysis_area_weight)] = 0.0
    analysis_area_weight[analysis_area_weight < 0] = 0.0
else:
    analysis_area_weight = None

print("analysis rows:", f"{len(work):,}")
print("complete-case retention:", f"{len(work) / len(sample):.3%}")


def local_metrics(y_true, y_pred, weights=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if weights is None:
        weights = np.ones(len(y_true), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)[valid]

    valid_weights = np.isfinite(weights) & (weights > 0)
    y_true = y_true[valid_weights]
    y_pred = y_pred[valid_weights]
    weights = weights[valid_weights]

    residual = y_true - y_pred
    observed_mean = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * residual**2)
    ss_tot = np.sum(weights * (y_true - observed_mean)**2)

    return {
        "n": int(len(y_true)),
        "observed_mean": float(observed_mean),
        "predicted_mean": float(np.average(y_pred, weights=weights)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
        "mae": float(np.average(np.abs(residual), weights=weights)),
        "bias_observed_minus_predicted": float(
            np.average(residual, weights=weights)
        ),
    }


prediction_by_model = {}
fold_rows = []

for model_name, features in model_specs.items():
    print("=" * 80)
    print("OOF model:", model_name)
    print("=" * 80)

    expected_oof = np.full(
        len(work),
        np.nan,
        dtype=np.float32,
    )

    for fold_number, (train_idx, test_idx) in enumerate(
        oof_splits,
        start=1,
    ):
        train_df = work.iloc[train_idx]
        test_df = work.iloc[test_idx]

        classifier, regressor = fit_models(
            train_df,
            features,
            fold_number,
        )

        raw_probability = classifier.predict_proba(
            test_df[features]
        )[:, 1]

        calibrated_probability = (
            adjust_probability_prior_shift(
                raw_probability,
                float(train_df["presence"].mean()),
                target_prior,
            )
        )

        conditional_fraction = np.clip(
            regressor.predict(test_df[features]),
            0.0,
            1.0,
        )

        expected_fraction = (
            calibrated_probability
            * conditional_fraction
        )

        expected_oof[test_idx] = expected_fraction.astype(
            np.float32
        )

        y_test_presence = test_df[
            "presence"
        ].to_numpy(dtype=np.uint8)

        y_test_fraction = test_df[
            "cropland_fraction"
        ].to_numpy(dtype=float)

        fold_metric = local_metrics(
            y_test_fraction,
            expected_fraction,
        )

        fold_rows.append({
            "model": model_name,
            "fold": fold_number,
            "combined_r2": fold_metric["r2"],
            "combined_rmse": fold_metric["rmse"],
            "combined_mae": fold_metric["mae"],
            "presence_auc": roc_auc_score(
                y_test_presence,
                raw_probability,
            ),
            "presence_average_precision": average_precision_score(
                y_test_presence,
                raw_probability,
            ),
        })

        print(
            f"fold {fold_number}/{N_SPLITS}: "
            f"R2={fold_metric['r2']:.5f}, "
            f"RMSE={fold_metric['rmse']:.5f}"
        )

        del classifier, regressor
        gc.collect()

    if not np.isfinite(expected_oof).all():
        raise RuntimeError(
            f"OOF予測が未完了です: {model_name}"
        )

    prediction_by_model[model_name] = expected_oof


weight_specs = {
    "unweighted_sample": None,
}

if (
    analysis_area_weight is not None
    and np.sum(analysis_area_weight > 0) > 0
):
    weight_specs[
        "area_weighted_reweighted"
    ] = analysis_area_weight

observed = work[
    "cropland_fraction"
].to_numpy(dtype=float)

global_rows = []

for model_name, prediction in prediction_by_model.items():
    for weighting, weights in weight_specs.items():
        global_rows.append({
            "model": model_name,
            "weighting": weighting,
            **local_metrics(
                observed,
                prediction,
                weights=weights,
            ),
        })

fold_metrics_npgdp = pd.DataFrame(fold_rows)
global_metrics_npgdp = pd.DataFrame(global_rows)

print("\nOOF global metrics")
print(
    global_metrics_npgdp
    .round(6)
    .to_string(index=False)
)

prediction_table_npgdp = work[
    [
        "row",
        "col",
        "lat",
        "lon",
        "spatial_block",
        "presence",
        "cropland_fraction",
        "exclusion_class",
        "soil_group_class",
    ]
].copy()

for model_name, prediction in prediction_by_model.items():
    prediction_table_npgdp[
        f"pred_{model_name}"
    ] = prediction

prediction_table_npgdp[
    "oof_error_plus_nutrients_gdp"
] = (
    prediction_table_npgdp[
        "pred_plus_nutrients_gdp"
    ]
    - prediction_table_npgdp[
        "cropland_fraction"
    ]
)

global_metrics_npgdp.to_csv(
    OUTPUT_DIR_NPGDP / "nutrients_gdp_oof_global_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

fold_metrics_npgdp.to_csv(
    OUTPUT_DIR_NPGDP / "nutrients_gdp_oof_fold_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

prediction_table_npgdp.to_csv(
    OUTPUT_DIR_NPGDP / "nutrients_gdp_oof_predictions.csv.gz",
    index=False,
    compression="gzip",
)

# R2図
plot_metrics = global_metrics_npgdp.copy()
plot_metrics["label"] = plot_metrics["model"].map({
    "baseline": "baseline",
    "plus_nutrients_gdp": "+N/P/GDP",
})

weighting_order = [
    value
    for value in [
        "unweighted_sample",
        "area_weighted_reweighted",
    ]
    if value in plot_metrics["weighting"].unique()
]

model_order = ["baseline", "+N/P/GDP"]
x = np.arange(len(weighting_order))
width = 0.34

fig, ax = plt.subplots(figsize=(9, 5.5))

for model_index, model_label in enumerate(model_order):
    values = []

    for weighting in weighting_order:
        row = plot_metrics[
            (plot_metrics["label"] == model_label)
            & (plot_metrics["weighting"] == weighting)
        ]
        values.append(float(row["r2"].iloc[0]))

    bars = ax.bar(
        x + (model_index - 0.5) * width,
        values,
        width,
        label=model_label,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.set_xticks(x)
ax.set_xticklabels([
    value.replace("_", "\n")
    for value in weighting_order
])
ax.set_ylabel("OOF R²")
ax.set_title("OOF R²: baseline vs nitrogen/phosphorus/GDP")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR_NPGDP / "nutrients_gdp_oof_r2_comparison.png",
    dpi=250,
    bbox_inches="tight",
)

plt.show()

# 空間誤差図
rng = np.random.default_rng(RANDOM_SEED + 9000)

if len(prediction_table_npgdp) > MAP_MAX_POINTS:
    map_idx = np.sort(
        rng.choice(
            len(prediction_table_npgdp),
            MAP_MAX_POINTS,
            replace=False,
        )
    )
else:
    map_idx = np.arange(len(prediction_table_npgdp))

map_frame = prediction_table_npgdp.iloc[map_idx]
map_error = map_frame[
    "oof_error_plus_nutrients_gdp"
].to_numpy(dtype=float)

finite_error = np.isfinite(map_error)
error_scale = float(
    np.nanpercentile(
        np.abs(map_error[finite_error]),
        98,
    )
)
error_scale = max(error_scale, 1e-6)

norm = TwoSlopeNorm(
    vmin=-error_scale,
    vcenter=0.0,
    vmax=error_scale,
)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    has_cartopy = True
except ImportError:
    has_cartopy = False

if has_cartopy:
    fig = plt.figure(figsize=(15, 7.5))
    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )
    ax.set_global()
    ax.add_feature(
        cfeature.LAND,
        facecolor="#f4f4f4",
        zorder=0,
    )
    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#e8f2f8",
        zorder=0,
    )
    ax.coastlines(linewidth=0.6)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.25,
        alpha=0.5,
    )

    scatter = ax.scatter(
        map_frame["lon"],
        map_frame["lat"],
        c=map_error,
        cmap="RdBu_r",
        norm=norm,
        s=2.0,
        alpha=0.45,
        linewidths=0,
        transform=ccrs.PlateCarree(),
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.45,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

else:
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_facecolor("#e8f2f8")

    scatter = ax.scatter(
        map_frame["lon"],
        map_frame["lat"],
        c=map_error,
        cmap="RdBu_r",
        norm=norm,
        s=2.0,
        alpha=0.45,
        linewidths=0,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)

fig.colorbar(
    scatter,
    ax=ax,
    orientation="horizontal",
    pad=0.05,
    fraction=0.045,
    label="OOF prediction error: predicted - observed",
)

ax.set_title(
    "Spatial distribution of cropland-fraction OOF prediction errors\n"
    "Nitrogen, phosphorus, and GDP model | target year 2020\n"
    "red = overprediction, blue = underprediction"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR_NPGDP / "nutrients_gdp_oof_spatial_error_map.png",
    dpi=250,
    bbox_inches="tight",
)

plt.show()
'''

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": analysis_cell.splitlines(True),
})

with output.open("w", encoding="utf-8") as handle:
    json.dump(notebook, handle, ensure_ascii=False, indent=1)

print(output)
