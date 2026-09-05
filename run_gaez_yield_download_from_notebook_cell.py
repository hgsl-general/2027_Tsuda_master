
# ============================================================
# Download all GAEZ climate potential / attainable yield rasters
# and keep only NPY outputs.
#
# Outputs:
#   C:\masterresearch\Comparative_advantage\GAEZ\ClimatePotential
#   C:\masterresearch\Comparative_advantage\GAEZ\AttainableY
# ============================================================

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


BASE_URL = "https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET"
GAEZ_DIR = Path(r"C:\masterresearch\Comparative_advantage\GAEZ")
README_RES02 = GAEZ_DIR / "_readme_res02.xlsx"
RES05_JSON_URL = (
    "https://data.apps.fao.org/catalog/dataset/aaedb1a0-53b0-44f9-b2de-0e6759053bb8/"
    "resource/0855ebe2-1b48-4439-a343-581c9c92dc6e/download/res05-ylx.json"
)
RES05_JSON_CACHE = GAEZ_DIR / "res05-ylx.json"

CLIMATE_OUT = GAEZ_DIR / "ClimatePotential"
ATTAINABLE_OUT = GAEZ_DIR / "AttainableY"

PERIOD = "HP0120"
CLIMATE = "AGERA5"
SSP = "HIST"
INPUT = "HILM"
NODATA_SENTINELS = {65535, -9999, -9999.0}


def download_file(url: str, destination: Path, retries: int = 3, sleep_seconds: float = 2.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".download")

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with temp_destination.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
            temp_destination.replace(destination)
            return
        except Exception:
            if temp_destination.exists():
                temp_destination.unlink()
            if attempt == retries:
                raise
            time.sleep(sleep_seconds * attempt)


def tif_to_npy(tif_path: Path, npy_path: Path) -> dict:
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        for sentinel in NODATA_SENTINELS:
            data[data == sentinel] = np.nan

        metadata = {
            "shape": list(data.shape),
            "dtype": "float32",
            "crs": str(src.crs),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "transform": list(src.transform),
            "source_nodata": nodata,
            "valid_cells": int(np.isfinite(data).sum()),
            "nan_cells": int(np.isnan(data).sum()),
            "min": float(np.nanmin(data)) if np.isfinite(data).any() else None,
            "max": float(np.nanmax(data)) if np.isfinite(data).any() else None,
            "mean": float(np.nanmean(data)) if np.isfinite(data).any() else None,
        }

    np.save(npy_path, data)
    return metadata


def build_res02_records() -> list[dict]:
    crops = pd.read_excel(README_RES02, sheet_name="CODES_CROP", header=0)
    crops = crops[
        crops["UPLOADED IN THE GOOGLE CLOUD STORAGE"]
        .astype(str)
        .str.upper()
        .eq("YES")
    ].copy()

    records = []
    for _, row in crops.iterrows():
        crop_code = str(row["CODE"]).strip()
        crop_name = str(row["Crop name"]).strip()
        crop_group = str(row["Crop group"]).strip()
        file_name = f"GAEZ-V5.RES02-YLD.{PERIOD}.{CLIMATE}.{SSP}.{crop_code}.{INPUT}.tif"
        records.append(
            {
                "dataset": "RES02-YLD",
                "variable": "climate_potential_yield",
                "crop_code": crop_code,
                "crop_name": crop_name,
                "crop_group": crop_group,
                "file_name": file_name,
                "url": f"{BASE_URL}/RES02-YLD/{file_name}",
                "output_dir": str(CLIMATE_OUT),
                "npy_path": str(CLIMATE_OUT / f"climate_potential_yield_{crop_code}.npy"),
            }
        )
    return records


def build_res05_records() -> list[dict]:
    if RES05_JSON_CACHE.exists():
        res05 = json.loads(RES05_JSON_CACHE.read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request(
            RES05_JSON_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            res05 = json.loads(response.read().decode("utf-8"))
        RES05_JSON_CACHE.write_text(json.dumps(res05, ensure_ascii=False, indent=2), encoding="utf-8")

    crop_dimension = res05["dimension"]["CROP-RES05"]["category"]
    labels = crop_dimension["label"]
    indices = crop_dimension["index"]
    crop_codes = sorted(indices, key=lambda code: indices[code])

    records = []
    for crop_code in crop_codes:
        crop_name = labels.get(crop_code, crop_code)
        file_name = f"GAEZ-V5.RES05-YLX.{PERIOD}.{CLIMATE}.{SSP}.{crop_code}.{INPUT}.tif"
        records.append(
            {
                "dataset": "RES05-YLX",
                "variable": "attainable_yield",
                "crop_code": crop_code,
                "crop_name": crop_name,
                "crop_group": "",
                "file_name": file_name,
                "url": f"{BASE_URL}/RES05-YLX/{file_name}",
                "output_dir": str(ATTAINABLE_OUT),
                "npy_path": str(ATTAINABLE_OUT / f"attainable_yield_{crop_code}.npy"),
            }
        )
    return records


def process_records(records: list[dict], log_path: Path) -> pd.DataFrame:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, record in enumerate(records, start=1):
        output_dir = Path(record["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        npy_path = Path(record["npy_path"])
        temp_tif_path = output_dir / "_tmp_downloads" / record["file_name"]
        status = "unknown"
        error = ""
        metadata = {}
        start_time = time.time()

        print(
            f"[{index:03d}/{len(records):03d}] "
            f"{record['dataset']} {record['crop_code']} -> {npy_path.name}"
        )

        try:
            if npy_path.exists():
                status = "skipped_existing_npy"
                arr = np.load(npy_path, mmap_mode="r")
                metadata = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "valid_cells": int(np.isfinite(arr).sum()),
                    "nan_cells": int(np.isnan(arr).sum()),
                }
            else:
                download_file(record["url"], temp_tif_path)
                metadata = tif_to_npy(temp_tif_path, npy_path)
                status = "downloaded_converted"
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            print("  ERROR:", error)
        finally:
            if temp_tif_path.exists():
                temp_tif_path.unlink()
            try:
                temp_dir = temp_tif_path.parent
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except Exception:
                pass

        elapsed = time.time() - start_time
        row = {
            **record,
            "status": status,
            "error": error,
            "elapsed_seconds": round(elapsed, 2),
            "npy_exists": npy_path.exists(),
            "npy_size_bytes": npy_path.stat().st_size if npy_path.exists() else None,
            **metadata,
        }
        rows.append(row)

        pd.DataFrame(rows).to_csv(log_path, index=False, encoding="utf-8-sig")
        print("  status:", status, "elapsed:", round(elapsed, 1), "sec")

    return pd.DataFrame(rows)


res02_records = build_res02_records()
res05_records = build_res05_records()

print("RES02 climate potential records:", len(res02_records))
print("RES05 attainable records:", len(res05_records))

res02_log = process_records(
    res02_records,
    CLIMATE_OUT / "climate_potential_yield_download_log.csv",
)
res05_log = process_records(
    res05_records,
    ATTAINABLE_OUT / "attainable_yield_download_log.csv",
)

combined_log = pd.concat([res02_log, res05_log], ignore_index=True)
combined_log.to_csv(
    GAEZ_DIR / "yield_download_combined_log.csv",
    index=False,
    encoding="utf-8-sig",
)

summary = combined_log.groupby(["dataset", "status"], dropna=False).size()
print("\nSummary")
print(summary)
print("\nSaved:")
print(CLIMATE_OUT)
print(ATTAINABLE_OUT)
print(GAEZ_DIR / "yield_download_combined_log.csv")
