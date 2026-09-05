# ============================================================
# GloFAS reliable river pipeline
#
# Purpose:
#   1) Download GloFAS historical daily river discharge year by year.
#   2) Compute annual p10 discharge at the original GloFAS grid.
#   3) Average annual p10 over selected years.
#   4) Create 5 arc-minute reliable-river masks and distance-to-reliable-river rasters.
#
# Default run is a 1-year test for 2020.
# For production, set YEARS = list(range(1991, 2021)).
# ============================================================

from __future__ import annotations

import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import cdsapi
import netCDF4
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt


DATASET = "cems-glofas-historical"
BASE_DIR = Path(r"C:\masterresearch\Comparative_advantage\GloFAS")
RAW_ZIP_DIR = BASE_DIR / "raw_zip"
RAW_NC_DIR = BASE_DIR / "raw_netcdf"
ANNUAL_SOURCE_DIR = BASE_DIR / "annual_p10_source_0p05"
PROCESSED_DIR = BASE_DIR / "processed_5min"
LOG_DIR = BASE_DIR / "logs"

# Test first. Change to list(range(1991, 2021)) after confirming the 2020 run works.
YEARS = [2020]
THRESHOLDS_M3S = [1, 10, 100]

# GloFAS historical v4.0 choices.
SYSTEM_VERSION = "version_4_0"
HYDROLOGICAL_MODEL = "lisflood"
PRODUCT_TYPE = "consolidated"
VARIABLE = "river_discharge_in_the_last_24_hours"
DATA_FORMAT = "netcdf"
DOWNLOAD_FORMAT = "zip"

# Raw files are large. Keep ZIP by default, delete extracted NetCDF after annual p10 is made.
DELETE_EXTRACTED_NC_AFTER_P10 = True
DELETE_ZIP_AFTER_P10 = False

# Target grid: 5 arc-minute global grid, matching HYDE/GAEZ-style 2160 x 4320.
TARGET_HEIGHT = 2160
TARGET_WIDTH = 4320
TARGET_RES = 1 / 12
TARGET_TRANSFORM = from_origin(-180.0, 90.0, TARGET_RES, TARGET_RES)
TARGET_CRS = "EPSG:4326"
TARGET_LAT = 90.0 - (np.arange(TARGET_HEIGHT, dtype=np.float64) + 0.5) * TARGET_RES
TARGET_LON = -180.0 + (np.arange(TARGET_WIDTH, dtype=np.float64) + 0.5) * TARGET_RES

MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]


def ensure_dirs() -> None:
    for directory in [RAW_ZIP_DIR, RAW_NC_DIR, ANNUAL_SOURCE_DIR, PROCESSED_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def check_credentials() -> None:
    # Use the EWDS endpoint even if the user's .cdsapirc was created from CDS.
    os.environ.setdefault("CDSAPI_URL", "https://ewds.climate.copernicus.eu/api")

    cdsa_path = Path.home() / ".cdsapirc"
    env_key = os.environ.get("CDSAPI_KEY")
    if not cdsa_path.exists() and not env_key:
        raise RuntimeError(
            "EWDS/CDS API credentials are not configured.\n"
            "1. Log in to https://ewds.climate.copernicus.eu/\n"
            "2. Accept the CEMS-FLOODS dataset licence for cems-glofas-historical.\n"
            "3. Create C:\\Users\\tsuda\\.cdsapirc with:\n"
            "   url: https://ewds.climate.copernicus.eu/api\n"
            "   key: <YOUR_PERSONAL_ACCESS_TOKEN>\n"
            "Then rerun this script."
        )


def glofas_request(year: int) -> dict[str, object]:
    return {
        "system_version": [SYSTEM_VERSION],
        "hydrological_model": [HYDROLOGICAL_MODEL],
        "product_type": [PRODUCT_TYPE],
        "variable": [VARIABLE],
        "hyear": [str(year)],
        "hmonth": MONTHS,
        "hday": DAYS,
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
    }


def download_year(year: int) -> Path:
    target_zip = RAW_ZIP_DIR / f"cems_glofas_historical_discharge_{year}.zip"
    if target_zip.exists() and target_zip.stat().st_size > 0:
        print(f"[download] skip existing: {target_zip}")
        return target_zip

    print(f"[download] requesting GloFAS {year} -> {target_zip}")
    client = cdsapi.Client()
    client.retrieve(DATASET, glofas_request(year)).download(str(target_zip))
    print(f"[download] completed: {target_zip} ({target_zip.stat().st_size / 1024**3:.2f} GB)")
    return target_zip


def extract_zip(zip_path: Path, year: int) -> list[Path]:
    extract_dir = RAW_NC_DIR / str(year)
    extract_dir.mkdir(parents=True, exist_ok=True)
    nc_files = sorted(extract_dir.rglob("*.nc")) + sorted(extract_dir.rglob("*.nc4")) + sorted(extract_dir.rglob("*.netcdf"))
    if nc_files:
        print(f"[extract] skip existing NetCDF files for {year}: {len(nc_files)}")
        return nc_files

    print(f"[extract] extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    nc_files = sorted(extract_dir.rglob("*.nc")) + sorted(extract_dir.rglob("*.nc4")) + sorted(extract_dir.rglob("*.netcdf"))
    if not nc_files:
        raise RuntimeError(f"No NetCDF files found after extracting {zip_path}")
    print(f"[extract] NetCDF files: {len(nc_files)}")
    return nc_files


def find_coord_name(dataset: netCDF4.Dataset, candidates: Iterable[str]) -> str:
    lower_map = {name.lower(): name for name in dataset.variables.keys()}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise RuntimeError(f"Could not find coordinate among {list(candidates)}")


def find_discharge_variable(dataset: netCDF4.Dataset, lat_name: str, lon_name: str) -> str:
    for name, variable in dataset.variables.items():
        if name in {lat_name, lon_name}:
            continue
        dimensions = tuple(variable.dimensions)
        if lat_name in dimensions and lon_name in dimensions and len(dimensions) >= 3:
            if np.issubdtype(variable.dtype, np.number):
                long_name = str(getattr(variable, "long_name", "")).lower()
                standard_name = str(getattr(variable, "standard_name", "")).lower()
                units = str(getattr(variable, "units", "")).lower()
                if "discharge" in long_name or "discharge" in standard_name or "m3" in units or "m**3" in units:
                    return name
    for name, variable in dataset.variables.items():
        dimensions = tuple(variable.dimensions)
        if lat_name in dimensions and lon_name in dimensions and len(dimensions) >= 3 and np.issubdtype(variable.dtype, np.number):
            return name
    raise RuntimeError("Could not find a 3D discharge variable in the NetCDF file")


def inspect_netcdf(nc_path: Path) -> dict[str, object]:
    with netCDF4.Dataset(nc_path) as dataset:
        lat_name = find_coord_name(dataset, ["latitude", "lat", "y"])
        lon_name = find_coord_name(dataset, ["longitude", "lon", "x"])
        variable_name = find_discharge_variable(dataset, lat_name, lon_name)
        discharge = dataset.variables[variable_name]
        dimensions = tuple(discharge.dimensions)
        lat_axis = dimensions.index(lat_name)
        lon_axis = dimensions.index(lon_name)
        time_axes = [index for index, dim in enumerate(dimensions) if dim not in {lat_name, lon_name}]
        if len(time_axes) != 1:
            raise RuntimeError(f"Expected one non-spatial axis for {variable_name}, got dimensions {dimensions}")
        time_axis = time_axes[0]
        lat_values = np.asarray(dataset.variables[lat_name][:], dtype=np.float64)
        lon_values = np.asarray(dataset.variables[lon_name][:], dtype=np.float64)
        return {
            "lat_name": lat_name,
            "lon_name": lon_name,
            "variable_name": variable_name,
            "dimensions": dimensions,
            "lat_axis": lat_axis,
            "lon_axis": lon_axis,
            "time_axis": time_axis,
            "lat_values": lat_values,
            "lon_values": lon_values,
            "shape": tuple(discharge.shape),
        }


def read_discharge_chunk(nc_path: Path, info: dict[str, object], row_start: int, row_stop: int) -> np.ndarray:
    with netCDF4.Dataset(nc_path) as dataset:
        variable = dataset.variables[str(info["variable_name"])]
        slices = []
        for axis, dim_name in enumerate(info["dimensions"]):
            if axis == int(info["time_axis"]):
                slices.append(slice(None))
            elif axis == int(info["lat_axis"]):
                slices.append(slice(row_start, row_stop))
            elif axis == int(info["lon_axis"]):
                slices.append(slice(None))
            else:
                slices.append(0)
        data = variable[tuple(slices)]
        data = np.ma.filled(data, np.nan).astype(np.float32)
        # Reorder to (time, lat_chunk, lon).
        current_dims = [dim for dim in info["dimensions"]]
        selected_dims = []
        for axis, dim_name in enumerate(current_dims):
            if axis == int(info["time_axis"]):
                selected_dims.append("time")
            elif axis == int(info["lat_axis"]):
                selected_dims.append("lat")
            elif axis == int(info["lon_axis"]):
                selected_dims.append("lon")
        transpose_order = [selected_dims.index("time"), selected_dims.index("lat"), selected_dims.index("lon")]
        return np.transpose(data, transpose_order)


def normalize_lon_if_needed(array: np.ndarray, lon_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.nanmax(lon_values) <= 180.0:
        return array, lon_values
    normalized_lon = ((lon_values + 180.0) % 360.0) - 180.0
    order = np.argsort(normalized_lon)
    return array[:, order], normalized_lon[order]


def source_transform_from_centers(lat_values: np.ndarray, lon_values: np.ndarray) -> rasterio.Affine:
    lon_res = float(abs(np.nanmedian(np.diff(lon_values))))
    lat_res = float(abs(np.nanmedian(np.diff(lat_values))))
    west = float(np.nanmin(lon_values) - lon_res / 2)
    north = float(np.nanmax(lat_values) + lat_res / 2)
    return from_origin(west, north, lon_res, lat_res)


def compute_annual_p10(nc_files: list[Path], year: int, chunk_rows: int = 16) -> Path:
    output_path = ANNUAL_SOURCE_DIR / f"glofas_annual_p10_discharge_{year}_0p05.npy"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists():
        print(f"[p10] skip existing annual p10: {output_path}")
        return output_path

    info = inspect_netcdf(nc_files[0])
    lat_values = np.asarray(info["lat_values"], dtype=np.float64)
    lon_values = np.asarray(info["lon_values"], dtype=np.float64)
    nlat = len(lat_values)
    nlon = len(lon_values)
    result = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=(nlat, nlon))

    print(f"[p10] computing annual p10 for {year}: shape=({nlat}, {nlon}), files={len(nc_files)}")
    for row_start in range(0, nlat, chunk_rows):
        row_stop = min(row_start + chunk_rows, nlat)
        chunks = [read_discharge_chunk(path, info, row_start, row_stop) for path in nc_files]
        data = np.concatenate(chunks, axis=0)
        data[data < 0] = np.nan
        with np.errstate(all="ignore"):
            result[row_start:row_stop, :] = np.nanpercentile(data, 10, axis=0).astype(np.float32)
        if row_start == 0 or row_stop == nlat or row_start % (chunk_rows * 20) == 0:
            print(f"[p10] rows {row_start}:{row_stop} / {nlat}")

    result.flush()
    sample = np.asarray(result)
    sample, normalized_lon = normalize_lon_if_needed(sample, lon_values)
    if not np.array_equal(normalized_lon, lon_values):
        np.save(output_path, sample.astype(np.float32))
        lon_values = normalized_lon

    metadata = {
        "year": year,
        "source_files": [str(path) for path in nc_files],
        "variable": info["variable_name"],
        "shape": [int(nlat), int(nlon)],
        "lat_min": float(np.nanmin(lat_values)),
        "lat_max": float(np.nanmax(lat_values)),
        "lon_min": float(np.nanmin(lon_values)),
        "lon_max": float(np.nanmax(lon_values)),
        "dtype": "float32",
        "statistic": "annual p10 of daily river discharge",
        "unit": "m3 s-1",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(ANNUAL_SOURCE_DIR / "source_lat.npy", lat_values.astype(np.float64))
    np.save(ANNUAL_SOURCE_DIR / "source_lon.npy", lon_values.astype(np.float64))
    print(f"[p10] saved: {output_path}")
    return output_path


def cleanup_raw(year: int, zip_path: Path) -> None:
    extract_dir = RAW_NC_DIR / str(year)
    if DELETE_EXTRACTED_NC_AFTER_P10 and extract_dir.exists():
        shutil.rmtree(extract_dir)
        print(f"[cleanup] removed extracted NetCDF: {extract_dir}")
    if DELETE_ZIP_AFTER_P10 and zip_path.exists():
        zip_path.unlink()
        print(f"[cleanup] removed ZIP: {zip_path}")


def build_period_mean_source(annual_paths: list[Path], period_label: str) -> Path:
    output_path = ANNUAL_SOURCE_DIR / f"glofas_mean_annual_p10_discharge_{period_label}_0p05.npy"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists():
        print(f"[period] skip existing period mean: {output_path}")
        return output_path

    first = np.load(annual_paths[0], mmap_mode="r")
    total = np.zeros(first.shape, dtype=np.float64)
    count = np.zeros(first.shape, dtype=np.uint16)
    for path in annual_paths:
        arr = np.load(path, mmap_mode="r")
        valid = np.isfinite(arr)
        total[valid] += arr[valid]
        count[valid] += 1
        print(f"[period] added {path.name}")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan).astype(np.float32)
    np.save(output_path, mean)
    metadata = {
        "period_label": period_label,
        "annual_files": [str(path) for path in annual_paths],
        "statistic": "mean of annual p10 daily river discharge",
        "unit": "m3 s-1",
        "shape": list(mean.shape),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[period] saved: {output_path}")
    return output_path


def reproject_p10_to_5min_max(source_p10_path: Path, period_label: str) -> Path:
    output_path = PROCESSED_DIR / f"p10_discharge_max_5min_{period_label}.npy"
    if output_path.exists():
        print(f"[5min] skip existing p10 max: {output_path}")
        return output_path

    source = np.load(source_p10_path).astype(np.float32)
    lat_values = np.load(ANNUAL_SOURCE_DIR / "source_lat.npy")
    lon_values = np.load(ANNUAL_SOURCE_DIR / "source_lon.npy")
    if lat_values[0] < lat_values[-1]:
        source = source[::-1, :]
        lat_values = lat_values[::-1]
    source_transform = source_transform_from_centers(lat_values, lon_values)
    destination = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype=np.float32)
    reproject(
        source,
        destination,
        src_transform=source_transform,
        src_crs=TARGET_CRS,
        src_nodata=np.nan,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=TARGET_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.max,
    )
    np.save(output_path, destination)
    print(f"[5min] saved p10 max: {output_path}")
    return output_path


def threshold_label(threshold: float) -> str:
    threshold_float = float(threshold)
    if threshold_float.is_integer():
        return str(int(threshold_float))
    return ("%g" % threshold_float).replace(".", "p")


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return (2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float32)


def distance_to_mask_km(mask: np.ndarray, output_path: Path, chunk_rows: int = 128) -> Path:
    if output_path.exists():
        print(f"[distance] skip existing: {output_path}")
        return output_path
    reliable = mask.astype(bool)
    if not reliable.any():
        np.save(output_path, np.full(reliable.shape, np.nan, dtype=np.float32))
        return output_path

    _, indices = distance_transform_edt(~reliable, return_indices=True)
    nearest_rows = indices[0].astype(np.int32, copy=False)
    nearest_cols = indices[1].astype(np.int32, copy=False)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=reliable.shape)

    lon_self = TARGET_LON[None, :]
    for row_start in range(0, TARGET_HEIGHT, chunk_rows):
        row_stop = min(row_start + chunk_rows, TARGET_HEIGHT)
        rows = slice(row_start, row_stop)
        lat_self = TARGET_LAT[row_start:row_stop, None]
        lat_nearest = TARGET_LAT[nearest_rows[rows, :]]
        lon_nearest = TARGET_LON[nearest_cols[rows, :]]
        output[rows, :] = haversine_km(lat_self, lon_self, lat_nearest, lon_nearest)
        if row_start == 0 or row_stop == TARGET_HEIGHT or row_start % (chunk_rows * 5) == 0:
            print(f"[distance] rows {row_start}:{row_stop} / {TARGET_HEIGHT}")
    output.flush()
    print(f"[distance] saved: {output_path}")
    return output_path


def make_reliable_masks_and_distances(source_p10_path: Path, period_label: str) -> None:
    p10_5min_path = reproject_p10_to_5min_max(source_p10_path, period_label)
    p10_5min = np.load(p10_5min_path, mmap_mode="r")
    for threshold in THRESHOLDS_M3S:
        label = threshold_label(threshold)
        mask_path = PROCESSED_DIR / f"reliable_river_p10_gt_{label}_m3s_5min_{period_label}.npy"
        distance_path = PROCESSED_DIR / f"distance_to_reliable_river_p10_gt_{label}_m3s_km_5min_{period_label}.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
            print(f"[mask] skip existing: {mask_path}")
        else:
            mask = (np.isfinite(p10_5min) & (p10_5min > threshold)).astype(np.uint8)
            np.save(mask_path, mask)
            print(f"[mask] saved: {mask_path}; reliable cells={int(mask.sum())}")
        distance_to_mask_km(mask, distance_path)


def write_run_metadata(period_label: str, source_p10_path: Path) -> None:
    metadata = {
        "dataset": DATASET,
        "system_version": SYSTEM_VERSION,
        "hydrological_model": HYDROLOGICAL_MODEL,
        "product_type": PRODUCT_TYPE,
        "variable": VARIABLE,
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
        "years": YEARS,
        "thresholds_m3s": THRESHOLDS_M3S,
        "period_source_p10": str(source_p10_path),
        "target_grid": {
            "height": TARGET_HEIGHT,
            "width": TARGET_WIDTH,
            "resolution_degree": TARGET_RES,
            "crs": TARGET_CRS,
        },
    }
    path = LOG_DIR / f"glofas_reliable_river_run_{period_label}.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[log] saved: {path}")


def main() -> None:
    ensure_dirs()
    check_credentials()
    annual_paths = []
    for year in YEARS:
        zip_path = download_year(year)
        nc_files = extract_zip(zip_path, year)
        annual_path = compute_annual_p10(nc_files, year)
        annual_paths.append(annual_path)
        cleanup_raw(year, zip_path)

    period_label = str(YEARS[0]) if len(YEARS) == 1 else f"{min(YEARS)}_{max(YEARS)}"
    period_source_p10 = build_period_mean_source(annual_paths, period_label)
    make_reliable_masks_and_distances(period_source_p10, period_label)
    write_run_metadata(period_label, period_source_p10)
    print("\nCompleted GloFAS reliable river pipeline.")
    print(f"Processed outputs: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
