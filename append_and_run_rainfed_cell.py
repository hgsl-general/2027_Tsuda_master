import json
import subprocess
import sys
from pathlib import Path

GAEZ_DIR = Path(r"C:\masterresearch\Comparative_advantage\GAEZ")
NOTEBOOK_PATH = GAEZ_DIR / "datalink.ipynb"
RUN_SCRIPT_PATH = GAEZ_DIR / "run_gaez_climate_potential_rainfed_download_from_notebook_cell.py"
PYTHON_EXE = Path(r"C:\Users\tsuda\miniconda3\python.exe")

cell_source = r'''
# ============================================================
# Download GAEZ climate potential yield under rain-fed conditions
# and keep only NPY outputs.
#
# Water/input condition:
#   HRLM = Rain-fed and high input level of management
#
# Outputs:
#   C:\masterresearch\Comparative_advantage\GAEZ\ClimatePotentialRainfed
# ============================================================

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


BASE_URL = "https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET"
GAEZ_DIR = Path(r"C:\masterresearch\Comparative_advantage\GAEZ")
README_RES02 = GAEZ_DIR / "_readme_res02.xlsx"
OUT_DIR = GAEZ_DIR / "ClimatePotentialRainfed"
TMP_DIR = OUT_DIR / "_tmp_tif"
LOG_PATH = OUT_DIR / "climate_potential_yield_rainfed_download_log.csv"

DATASET = "RES02-YLD"
PERIOD = "HP0120"
CLIMATE = "AGERA5"
SSP = "HIST"
WATER_INPUT = "HRLM"  # Rain-fed and high input level of management

CROP_CODE_COL = "CODE"
CROP_NAME_COL = "LABEL"
UPLOADED_COL = "UPLOADED IN THE GOOGLE CLOUD STORAGE"
NODATA_SENTINELS = (65535, -9999, -9999.0)


def load_uploaded_res02_crops(readme_path: Path) -> pd.DataFrame:
    crops = pd.read_excel(readme_path, sheet_name="CODES_CROP")
    uploaded_mask = crops[UPLOADED_COL].astype(str).str.upper().eq("YES")
    uploaded_crops = crops.loc[uploaded_mask, [CROP_CODE_COL, CROP_NAME_COL]].copy()
    uploaded_crops[CROP_CODE_COL] = uploaded_crops[CROP_CODE_COL].astype(str).str.strip()
    uploaded_crops[CROP_NAME_COL] = uploaded_crops[CROP_NAME_COL].astype(str).str.strip()
    return uploaded_crops.sort_values(CROP_CODE_COL).reset_index(drop=True)


def build_download_records(crops: pd.DataFrame) -> list[dict[str, str]]:
    records = []
    for crop_record in crops.to_dict("records"):
        crop_code = crop_record[CROP_CODE_COL]
        filename = f"GAEZ-V5.{DATASET}.{PERIOD}.{CLIMATE}.{SSP}.{crop_code}.{WATER_INPUT}.tif"
        records.append(
            {
                "crop_code": crop_code,
                "crop_name": crop_record[CROP_NAME_COL],
                "dataset": DATASET,
                "period": PERIOD,
                "climate": CLIMATE,
                "ssp": SSP,
                "water_input": WATER_INPUT,
                "url": f"{BASE_URL}/{DATASET}/{filename}",
                "tmp_tif": str(TMP_DIR / filename),
                "npy_path": str(OUT_DIR / f"climate_potential_yield_rainfed_{crop_code}.npy"),
            }
        )
    return records


def download_file(url: str, dest_path: Path, retries: int = 3) -> int:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(1, retries + 1):
        try:
            if part_path.exists():
                part_path.unlink()
            with urllib.request.urlopen(request, timeout=180) as response:
                with part_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
            part_path.replace(dest_path)
            return dest_path.stat().st_size
        except Exception:
            if part_path.exists():
                part_path.unlink()
            if attempt == retries:
                raise
            time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download: {url}")


def tif_to_npy(tif_path: Path, npy_path: Path) -> dict[str, object]:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nodata_value = src.nodata
        if nodata_value is not None:
            data[data == nodata_value] = np.nan
        for sentinel in NODATA_SENTINELS:
            data[data == sentinel] = np.nan

        valid_mask = np.isfinite(data)
        metadata = {
            "shape": f"{data.shape[0]}x{data.shape[1]}",
            "dtype": str(data.dtype),
            "crs": str(src.crs),
            "bounds": str(tuple(round(value, 6) for value in src.bounds)),
            "nodata": nodata_value,
            "valid_cells": int(valid_mask.sum()),
            "nan_cells": int(np.isnan(data).sum()),
            "min": float(np.nanmin(data)) if valid_mask.any() else np.nan,
            "max": float(np.nanmax(data)) if valid_mask.any() else np.nan,
            "mean": float(np.nanmean(data)) if valid_mask.any() else np.nan,
        }

    np.save(npy_path, data)
    metadata["npy_bytes"] = npy_path.stat().st_size
    return metadata


def process_records(records: list[dict[str, str]], log_path: Path) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    log_records: list[dict[str, object]] = []
    total_records = len(records)

    for record_index, record in enumerate(records, start=1):
        crop_code = record["crop_code"]
        tmp_tif = Path(record["tmp_tif"])
        npy_path = Path(record["npy_path"])
        log_record: dict[str, object] = dict(record)
        log_record["started_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

        try:
            if npy_path.exists():
                log_record["status"] = "skipped_existing_npy"
                log_record["npy_bytes"] = npy_path.stat().st_size
                print(f"[{record_index:03d}/{total_records}] skip {crop_code}: existing NPY")
            else:
                downloaded_bytes = download_file(record["url"], tmp_tif)
                metadata = tif_to_npy(tmp_tif, npy_path)
                log_record.update(metadata)
                log_record["downloaded_tif_bytes"] = downloaded_bytes
                log_record["status"] = "downloaded_converted"
                print(f"[{record_index:03d}/{total_records}] done {crop_code}: {npy_path.name}")
        except Exception as error:
            log_record["status"] = "error"
            log_record["error"] = repr(error)
            print(f"[{record_index:03d}/{total_records}] error {crop_code}: {error}")
        finally:
            if tmp_tif.exists():
                tmp_tif.unlink()
            log_record["finished_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
            log_records.append(log_record)
            pd.DataFrame(log_records).to_csv(log_path, index=False, encoding="utf-8-sig")

    if TMP_DIR.exists() and not any(TMP_DIR.iterdir()):
        TMP_DIR.rmdir()

    log_df = pd.DataFrame(log_records)
    failed = log_df[log_df["status"].eq("error")]
    if not failed.empty:
        raise RuntimeError(f"{len(failed)} downloads/conversions failed. See {log_path}")
    return log_df


crops = load_uploaded_res02_crops(README_RES02)
records = build_download_records(crops)
log_df = process_records(records, LOG_PATH)

print("\nCompleted rain-fed climate potential yield download.")
print(f"Condition: {WATER_INPUT} = Rain-fed and high input")
print(f"NPY files: {len(list(OUT_DIR.glob('climate_potential_yield_rainfed_*.npy')))}")
print(f"Remaining TIF files: {len(list(OUT_DIR.rglob('*.tif')))}")
print(f"Output directory: {OUT_DIR}")
print(f"Log: {LOG_PATH}")
print(log_df["status"].value_counts().to_string())
'''.strip() + "\n"

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
marker = "Download GAEZ climate potential yield under rain-fed conditions"
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell_source.splitlines(keepends=True),
}

replaced = False
for cell in notebook.get("cells", []):
    source_text = "".join(cell.get("source", []))
    if marker in source_text:
        cell.update(new_cell)
        replaced = True
        break
if not replaced:
    notebook.setdefault("cells", []).append(new_cell)

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
RUN_SCRIPT_PATH.write_text(cell_source, encoding="utf-8")
print(f"Notebook cell {'updated' if replaced else 'appended'}: {NOTEBOOK_PATH}")
print(f"Runner script written: {RUN_SCRIPT_PATH}")

completed = subprocess.run([str(PYTHON_EXE), str(RUN_SCRIPT_PATH)], check=False)
sys.exit(completed.returncode)
