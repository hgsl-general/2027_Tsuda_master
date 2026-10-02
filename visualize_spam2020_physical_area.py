#!/usr/bin/env python3
"""Visualize SPAM2020 physical-area rasters on the existing GAEZ grid.

The defaults mirror the paths used in the existing server-side notebooks:
    /work/tsuda/GAEZ
    /work/tsuda/HYDE/cropland_npys

The script creates:
    1. an inventory of extracted SPAM GeoTIFFs;
    2. physical-area maps for several major crops;
    3. crop-share maps, when the 2020 HYDE cropland fraction is available.

The crop-share target for the crop-choice Stage 2 is

    share[i, c] = SPAM physical area[i, c] / cropland area[i]

where cropland area[i] is the 2020 cropland fraction multiplied by the
GAEZ grid-cell area.  If the modeled crops do not exhaust cropland, the
remaining share should later be represented by an explicit "other/fallow"
alternative in the softmax model.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm, Normalize
from rasterio.enums import Resampling


# -----------------------------------------------------------------------------
# Server-side paths.  These are intentionally Linux paths, not local Windows
# paths, so the script can be copied to hgsl-ws and executed there.
# -----------------------------------------------------------------------------
ROOT = Path("/work/tsuda")
GAEZ_DIR = ROOT / "GAEZ"
CROPLAND_DIR = ROOT / "HYDE" / "cropland_npys"
SPAM_ROOT = GAEZ_DIR / "cropgrids"
SPAM_DATA_DIR = SPAM_ROOT / "spam2020V2r2_global_physical_area"
OUTPUT_DIR = SPAM_ROOT / "visualization_spam2020"


CROP_ALIASES = {
    "wheat": ("whea", "wheat"),
    "rice": ("rice",),
    "maize": ("maiz", "maize", "corn"),
    "soybean": ("soyb", "soybean", "soy"),
    "potato": ("pota", "potato"),
    "sugarcane": ("sugc", "sugarcane"),
}

SPAM_CROP_CODES = {
    "wheat": "WHEA",
    "rice": "RICE",
    "maize": "MAIZ",
    "soybean": "SOYB",
    "potato": "POTA",
    "sugarcane": "SUGC",
}

SPAM_FILENAME_RE = re.compile(
    r"_A_(?P<crop>[A-Z0-9]+)_(?P<system>[AIR])$", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=SPAM_DATA_DIR)
    parser.add_argument("--cropland-dir", type=Path, default=CROPLAND_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--crops",
        nargs="+",
        default=["wheat", "rice", "maize", "soybean", "potato", "sugarcane"],
        choices=sorted(CROP_ALIASES),
    )
    parser.add_argument("--max-pixels", type=int, default=1_500_000)
    return parser.parse_args()


def find_rasters(data_dir: Path) -> list[Path]:
    rasters = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not rasters:
        raise FileNotFoundError(
            f"No GeoTIFFs found below {data_dir}. "
            "Extract the SPAM physical-area archive first."
        )
    return rasters


def find_rasters_with_fallback(data_dir: Path, fallback_dir: Path) -> tuple[Path, list[Path]]:
    """Support both `unzip -d ...` and extraction directly in cropgrids."""
    try:
        return data_dir, find_rasters(data_dir)
    except FileNotFoundError:
        if data_dir != fallback_dir:
            print(f"[INFO] No GeoTIFFs below {data_dir}; scanning {fallback_dir}.")
            return fallback_dir, find_rasters(fallback_dir)
        raise


def parse_spam_filename(path: Path) -> tuple[str, str] | None:
    """Parse SPAM2020 filenames such as ..._A_WHEA_I.tif.

    The final production-system code is:
        I = irrigated
        R = rainfed
        A = all systems combined
    """
    match = SPAM_FILENAME_RE.search(path.stem)
    if match is None:
        return None
    return match.group("crop").upper(), match.group("system").upper()


def select_total_raster(rasters: list[Path], crop: str) -> Path | None:
    target_code = SPAM_CROP_CODES[crop]
    total_candidates = []
    for path in rasters:
        parsed = parse_spam_filename(path)
        if parsed == (target_code, "A"):
            total_candidates.append(path)
    if not total_candidates:
        return None
    return sorted(total_candidates)[0]


def load_2020_cropland_area(cropland_dir: Path) -> np.ndarray | None:
    fraction_path = cropland_dir / "cropland_fraction_1950_2024.npy"
    years_path = cropland_dir / "years.npy"
    grid_area_path = cropland_dir / "grid_area_ha.npy"

    if not fraction_path.exists() or not years_path.exists():
        print("[INFO] HYDE 2020 cropland fraction was not found; share maps skipped.")
        return None

    fractions = np.load(fraction_path, mmap_mode="r")
    years = np.load(years_path)
    year_matches = np.flatnonzero(years == 2020)
    if len(year_matches) != 1:
        raise ValueError(f"Could not identify exactly one 2020 HYDE layer: {years}")

    fraction_2020 = np.asarray(fractions[int(year_matches[0])], dtype=np.float32)

    if grid_area_path.exists():
        grid_area_ha = np.asarray(np.load(grid_area_path, mmap_mode="r"), dtype=np.float32)
    else:
        # Existing notebook uses grid_area_km2.npy.  Convert km2 to hectares.
        grid_area_km2_path = cropland_dir / "grid_area_km2.npy"
        if not grid_area_km2_path.exists():
            print("[INFO] Grid-cell area was not found; share maps skipped.")
            return None
        grid_area_ha = np.asarray(
            np.load(grid_area_km2_path, mmap_mode="r"), dtype=np.float32
        ) * 100.0

    if fraction_2020.shape != grid_area_ha.shape:
        raise ValueError(
            "HYDE fraction and grid-area shapes differ: "
            f"{fraction_2020.shape} vs {grid_area_ha.shape}"
        )

    fraction_2020 = np.nan_to_num(fraction_2020, nan=0.0, posinf=0.0, neginf=0.0)
    fraction_2020 = np.clip(fraction_2020, 0.0, 1.0)
    return fraction_2020 * grid_area_ha


def read_downsampled(path: Path, max_pixels: int) -> tuple[np.ma.MaskedArray, dict]:
    with rasterio.open(path) as src:
        scale = max(1.0, np.sqrt((src.height * src.width) / max_pixels))
        out_height = max(1, int(round(src.height / scale)))
        out_width = max(1, int(round(src.width / scale)))
        values = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
            masked=True,
        ).astype(np.float32)
        metadata = {
            "height": src.height,
            "width": src.width,
            "crs": str(src.crs),
            "bounds": src.bounds,
            "transform": src.transform,
            "nodata": src.nodata,
        }
    values = np.ma.masked_invalid(values)
    values = np.ma.masked_less_equal(values, 0.0)
    return values, metadata


def downsample_array(values: np.ndarray, out_shape: tuple[int, int]) -> np.ma.MaskedArray:
    """Downsample a same-grid array using rasterio in memory."""
    values = np.asarray(values, dtype=np.float32)
    height, width = values.shape
    out_height, out_width = out_shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan,
        "transform": rasterio.Affine(1, 0, 0, 0, -1, 0),
    }
    from rasterio.io import MemoryFile

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(values, 1)
            result = dataset.read(
                1,
                out_shape=(out_height, out_width),
                resampling=Resampling.average,
                masked=True,
            )
    return np.ma.masked_invalid(result.astype(np.float32))


def write_inventory(rasters: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "crop_code",
        "production_system",
        "width",
        "height",
        "crs",
        "nodata",
        "left",
        "bottom",
        "right",
        "top",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in rasters:
            with rasterio.open(path) as src:
                parsed = parse_spam_filename(path) or ("", "")
                writer.writerow(
                    {
                        "path": str(path),
                        "crop_code": parsed[0],
                        "production_system": parsed[1],
                        "width": src.width,
                        "height": src.height,
                        "crs": src.crs,
                        "nodata": src.nodata,
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top,
                    }
                )


def plot_maps(
    selected: dict[str, Path],
    cropland_area_ha: np.ndarray | None,
    output_dir: Path,
    max_pixels: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(selected)
    if n == 0:
        raise RuntimeError("No crop rasters were selected.")

    physical_fig, physical_axes = plt.subplots(
        2,
        (n + 1) // 2,
        figsize=(5.0 * ((n + 1) // 2), 7.0),
        squeeze=False,
        constrained_layout=True,
    )
    share_fig = share_axes = None
    if cropland_area_ha is not None:
        share_fig, share_axes = plt.subplots(
            2,
            (n + 1) // 2,
            figsize=(5.0 * ((n + 1) // 2), 7.0),
            squeeze=False,
            constrained_layout=True,
        )

    for index, (crop, path) in enumerate(selected.items()):
        row = index // ((n + 1) // 2)
        col = index % ((n + 1) // 2)
        physical, metadata = read_downsampled(path, max_pixels=max_pixels)
        axis = physical_axes[row, col]
        positive = physical.compressed()
        vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
        vmax = max(vmax, 1e-6)
        p_low = float(np.nanpercentile(positive, 1)) if positive.size else 1e-4
        p_high = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
        vmin = max(1e-4, p_low)
        vmax = max(p_high, vmin * 10.0)
        image = axis.imshow(
            physical,
            cmap="YlGn",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=(metadata["bounds"].left, metadata["bounds"].right,
                    metadata["bounds"].bottom, metadata["bounds"].top),
            aspect="auto",
        )
        axis.set_title(f"{crop}: physical area\n{path.name}", fontsize=9)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        physical_fig.colorbar(image, ax=axis, shrink=0.78, label="ha (log scale)")

        if share_axes is not None:
            share_full = None
            with rasterio.open(path) as src:
                if (src.height, src.width) == cropland_area_ha.shape:
                    raw = src.read(1, masked=True).astype(np.float32)
                    share_full = np.divide(
                        raw,
                        cropland_area_ha,
                        out=np.zeros_like(raw, dtype=np.float32),
                        where=cropland_area_ha > 0,
                    )
                    share_full = np.ma.masked_where(~np.isfinite(share_full) | (share_full <= 0), share_full)
                    share = downsample_array(share_full.filled(np.nan), physical.shape)
                else:
                    print(
                        f"[WARN] Grid mismatch for {path.name}: "
                        f"raster={(src.height, src.width)}, HYDE={cropland_area_ha.shape}"
                    )
                    share = None

            if share is not None:
                share_axis = share_axes[row, col]
                share_values = share.compressed()
                share_vmax = min(1.0, float(np.nanpercentile(share_values, 99.5))) if share_values.size else 1.0
                share_vmax = max(share_vmax, 1e-4)
                share_image = share_axis.imshow(
                    share,
                    cmap="viridis",
                    norm=Normalize(vmin=0.0, vmax=share_vmax),
                    extent=(metadata["bounds"].left, metadata["bounds"].right,
                            metadata["bounds"].bottom, metadata["bounds"].top),
                    aspect="auto",
                )
                share_axis.set_title(f"{crop}: physical-area share\n{path.name}", fontsize=9)
                share_axis.set_xlabel("longitude")
                share_axis.set_ylabel("latitude")
                share_fig.colorbar(share_image, ax=share_axis, shrink=0.78, label="share of cropland")

    for axis in physical_axes.flat[n:]:
        axis.axis("off")
    physical_fig.savefig(output_dir / "spam2020_physical_area_maps.png", dpi=180)
    plt.close(physical_fig)

    if share_axes is not None and share_fig is not None:
        for axis in share_axes.flat[n:]:
            axis.axis("off")
        share_fig.savefig(output_dir / "spam2020_physical_area_share_maps.png", dpi=180)
        plt.close(share_fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual_data_dir, rasters = find_rasters_with_fallback(args.data_dir, SPAM_ROOT)
    print(f"Found GeoTIFFs: {len(rasters)}")
    print(f"Data directory: {actual_data_dir}")

    parsed_files = [parse_spam_filename(path) for path in rasters]
    parsed_files = [item for item in parsed_files if item is not None]
    all_crop_codes = sorted({crop_code for crop_code, _ in parsed_files})
    systems = sorted({system for _, system in parsed_files})
    print(f"Detected crop codes: {len(all_crop_codes)}")
    print(f"Detected production systems: {systems} (I=irrigated, R=rainfed, A=all)")

    write_inventory(rasters, args.output_dir / "spam2020_geotiff_inventory.csv")

    selected: dict[str, Path] = {}
    for crop in args.crops:
        path = select_total_raster(rasters, crop)
        if path is None:
            print(f"[WARN] Could not identify a unique total physical-area raster for {crop}.")
            continue
        selected[crop] = path
        print(f"{crop:10s} -> {path}")

    cropland_area_ha = load_2020_cropland_area(args.cropland_dir)
    if cropland_area_ha is not None:
        print(f"HYDE 2020 cropland-area grid: {cropland_area_ha.shape}")

    plot_maps(
        selected=selected,
        cropland_area_ha=cropland_area_ha,
        output_dir=args.output_dir,
        max_pixels=args.max_pixels,
    )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
