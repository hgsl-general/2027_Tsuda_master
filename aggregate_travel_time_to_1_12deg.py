from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LogNorm
from rasterio.windows import Window


DOWNLOAD_DIR = Path(r"C:\Users\tsuda\Downloads")
OUTPUT_DIR = Path(r"C:\masterresearch\Comparative_advantage\distance_to_cities")

TARGET_HEIGHT = 2160
TARGET_WIDTH = 4320
TARGET_RESOLUTION_DEGREES = 1 / 12
SOURCE_TO_TARGET_FACTOR = 10

GLOBAL_LEFT = -180.0
GLOBAL_TOP = 90.0
GLOBAL_RIGHT = 180.0
GLOBAL_BOTTOM = -90.0

NODATA_VALUE = 65535


CITY_LABELS = {
    1: "cities >=5M and <50M",
    2: "cities >=1M and <5M",
    3: "cities >=500k and <1M",
    4: "cities >=200k and <500k",
    5: "cities >=100k and <200k",
    6: "cities >=50k and <100k",
    7: "cities >=20k and <50k",
    8: "cities >=10k and <20k",
    9: "cities >=5k and <10k",
    10: "cities >=20k",
    11: "cities >=50k",
    12: "cities >=5k",
}


PORT_LABELS = {
    1: "large ports (160)",
    2: "medium ports (361)",
    3: "small ports (990)",
    4: "very small ports (2,153)",
    5: "any ports (3,778)",
}


@dataclass(frozen=True)
class LayerSpec:
    group: str
    layer_number: int
    label: str
    source_path: Path

    @property
    def output_prefix(self) -> str:
        return f"{self.group}_{self.layer_number:02d}_1_12deg"


def discover_layers() -> list[LayerSpec]:
    layers: list[LayerSpec] = []

    for layer_number, label in CITY_LABELS.items():
        source_path = DOWNLOAD_DIR / f"travel_time_to_cities_{layer_number}.tif"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        layers.append(
            LayerSpec(
                group="cities",
                layer_number=layer_number,
                label=label,
                source_path=source_path,
            )
        )

    for layer_number, label in PORT_LABELS.items():
        source_path = DOWNLOAD_DIR / f"travel_time_to_ports_{layer_number}.tif"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        layers.append(
            LayerSpec(
                group="ports",
                layer_number=layer_number,
                label=label,
                source_path=source_path,
            )
        )

    return layers


def target_row_offset_from_source_top(source_top: float) -> int:
    row_offset = round((GLOBAL_TOP - source_top) / TARGET_RESOLUTION_DEGREES)
    if row_offset < 0 or row_offset >= TARGET_HEIGHT:
        raise ValueError(f"Invalid target row offset: {row_offset}")
    return row_offset


def safe_nan_stats(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_count = np.sum(np.isfinite(block), axis=(1, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_values = np.nanmean(block, axis=(1, 3))
        min_values = np.nanmin(block, axis=(1, 3))
        median_values = np.nanmedian(block, axis=(1, 3))

    mean_values[valid_count == 0] = np.nan
    min_values[valid_count == 0] = np.nan
    median_values[valid_count == 0] = np.nan

    return (
        mean_values.astype(np.float32),
        min_values.astype(np.float32),
        median_values.astype(np.float32),
    )


def aggregate_layer(layer: LayerSpec, chunk_target_rows: int = 20) -> dict[str, Path]:
    mean_grid = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype=np.float32)
    min_grid = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype=np.float32)
    median_grid = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype=np.float32)

    with rasterio.open(layer.source_path) as source:
        if source.count != 1:
            raise ValueError(f"{layer.source_path} has {source.count} bands; expected 1")
        if source.width % SOURCE_TO_TARGET_FACTOR != 0:
            raise ValueError(f"Source width not divisible by {SOURCE_TO_TARGET_FACTOR}: {source.width}")
        if source.height % SOURCE_TO_TARGET_FACTOR != 0:
            raise ValueError(f"Source height not divisible by {SOURCE_TO_TARGET_FACTOR}: {source.height}")

        target_cols_for_source = source.width // SOURCE_TO_TARGET_FACTOR
        target_rows_for_source = source.height // SOURCE_TO_TARGET_FACTOR

        if target_cols_for_source != TARGET_WIDTH:
            raise ValueError(
                f"Unexpected target width {target_cols_for_source}; expected {TARGET_WIDTH}"
            )

        row_offset = target_row_offset_from_source_top(source.bounds.top)
        print(
            f"Aggregating {layer.source_path.name}: "
            f"source={source.height}x{source.width}, "
            f"target rows {row_offset}:{row_offset + target_rows_for_source}"
        )

        for target_row_start in range(0, target_rows_for_source, chunk_target_rows):
            target_row_count = min(chunk_target_rows, target_rows_for_source - target_row_start)
            source_row_start = target_row_start * SOURCE_TO_TARGET_FACTOR
            source_row_count = target_row_count * SOURCE_TO_TARGET_FACTOR

            window = Window(
                col_off=0,
                row_off=source_row_start,
                width=source.width,
                height=source_row_count,
            )

            source_block = source.read(1, window=window)
            source_block = source_block.astype(np.float32, copy=False)
            source_block[source_block == NODATA_VALUE] = np.nan

            reshaped = source_block.reshape(
                target_row_count,
                SOURCE_TO_TARGET_FACTOR,
                TARGET_WIDTH,
                SOURCE_TO_TARGET_FACTOR,
            )

            mean_values, min_values, median_values = safe_nan_stats(reshaped)

            target_slice = slice(
                row_offset + target_row_start,
                row_offset + target_row_start + target_row_count,
            )

            mean_grid[target_slice, :] = mean_values
            min_grid[target_slice, :] = min_values
            median_grid[target_slice, :] = median_values

            if (target_row_start // chunk_target_rows) % 20 == 0:
                print(
                    f"  rows {target_row_start:4d}-"
                    f"{target_row_start + target_row_count:4d} / {target_rows_for_source}"
                )

    output_paths = {
        "mean": OUTPUT_DIR / f"{layer.output_prefix}_mean.npy",
        "min": OUTPUT_DIR / f"{layer.output_prefix}_min.npy",
        "median": OUTPUT_DIR / f"{layer.output_prefix}_median.npy",
    }

    np.save(output_paths["mean"], mean_grid)
    np.save(output_paths["min"], min_grid)
    np.save(output_paths["median"], median_grid)

    return output_paths


def make_coordinate_arrays() -> tuple[np.ndarray, np.ndarray]:
    latitudes = GLOBAL_TOP - (np.arange(TARGET_HEIGHT, dtype=np.float32) + 0.5) * TARGET_RESOLUTION_DEGREES
    longitudes = GLOBAL_LEFT + (np.arange(TARGET_WIDTH, dtype=np.float32) + 0.5) * TARGET_RESOLUTION_DEGREES
    return latitudes, longitudes


def plot_group(layers: list[LayerSpec], output_index: pd.DataFrame, group: str) -> Path:
    group_layers = [layer for layer in layers if layer.group == group]
    n_layers = len(group_layers)
    n_cols = 3 if group == "cities" else 2
    n_rows = math.ceil(n_layers / n_cols)

    sample_values = []
    for layer in group_layers:
        mean_path = output_index.loc[
            (output_index["group"] == layer.group)
            & (output_index["layer_number"] == layer.layer_number),
            "mean_path",
        ].iloc[0]
        data = np.load(mean_path, mmap_mode="r")
        valid = data[np.isfinite(data) & (data > 0)]
        if valid.size:
            sample_values.append(np.nanpercentile(valid, 99))

    vmax = max(sample_values) if sample_values else 1.0
    vmax = max(vmax, 10.0)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.3 * n_cols, 3.2 * n_rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()

    image = None
    for axis, layer in zip(axes_array, group_layers):
        mean_path = output_index.loc[
            (output_index["group"] == layer.group)
            & (output_index["layer_number"] == layer.layer_number),
            "mean_path",
        ].iloc[0]
        data = np.load(mean_path, mmap_mode="r")
        plot_data = np.ma.masked_invalid(data)
        plot_data = np.ma.masked_less_equal(plot_data, 0)

        image = axis.imshow(
            plot_data,
            origin="upper",
            extent=(GLOBAL_LEFT, GLOBAL_RIGHT, GLOBAL_BOTTOM, GLOBAL_TOP),
            cmap="viridis_r",
            norm=LogNorm(vmin=1, vmax=vmax),
            interpolation="nearest",
        )
        axis.set_title(f"{layer.layer_number}: {layer.label}", fontsize=9)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_xlim(GLOBAL_LEFT, GLOBAL_RIGHT)
        axis.set_ylim(-60, 85)

    for axis in axes_array[n_layers:]:
        axis.axis("off")

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes_array[:n_layers], shrink=0.85)
        colorbar.set_label("Mean travel time to nearest target [minutes, log scale]")

    fig.suptitle(
        f"Travel time to {group}: 1/12 degree grid mean",
        fontsize=14,
    )

    output_path = OUTPUT_DIR / f"travel_time_to_{group}_mean_maps_1_12deg.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_metadata(layers: list[LayerSpec], output_index: pd.DataFrame) -> None:
    latitudes, longitudes = make_coordinate_arrays()
    np.save(OUTPUT_DIR / "lat_1_12deg.npy", latitudes)
    np.save(OUTPUT_DIR / "lon_1_12deg.npy", longitudes)

    metadata = {
        "description": (
            "Travel time rasters aggregated from 30 arc-second GeoTIFFs "
            "to the GAEZ-compatible 1/12 degree global grid."
        ),
        "unit": "minutes",
        "target_grid": {
            "height": TARGET_HEIGHT,
            "width": TARGET_WIDTH,
            "resolution_degrees": TARGET_RESOLUTION_DEGREES,
            "extent": {
                "left": GLOBAL_LEFT,
                "right": GLOBAL_RIGHT,
                "top": GLOBAL_TOP,
                "bottom": GLOBAL_BOTTOM,
            },
            "outside_source_extent": "NaN",
        },
        "source": {
            "directory": str(DOWNLOAD_DIR),
            "resolution": "30 arc seconds",
            "source_extent": "lon -180 to 180, lat 85 to -60",
            "nodata_value_treated_as_nan": NODATA_VALUE,
        },
        "layers": [
            {
                "group": layer.group,
                "layer_number": layer.layer_number,
                "label": layer.label,
                "source_path": str(layer.source_path),
            }
            for layer in layers
        ],
        "outputs": output_index.to_dict(orient="records"),
    }

    with (OUTPUT_DIR / "travel_time_1_12deg_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    layers = discover_layers()
    records = []

    for layer in layers:
        output_paths = aggregate_layer(layer)
        records.append(
            {
                "group": layer.group,
                "layer_number": layer.layer_number,
                "label": layer.label,
                "source_path": str(layer.source_path),
                "mean_path": str(output_paths["mean"]),
                "min_path": str(output_paths["min"]),
                "median_path": str(output_paths["median"]),
            }
        )

    output_index = pd.DataFrame(records)
    output_index_path = OUTPUT_DIR / "travel_time_1_12deg_output_index.csv"
    output_index.to_csv(output_index_path, index=False, encoding="utf-8-sig")
    write_metadata(layers, output_index)

    for group in ["cities", "ports"]:
        figure_path = plot_group(layers, output_index, group)
        print(f"Saved figure: {figure_path}")

    print(f"Saved output index: {output_index_path}")
    print(f"Saved all outputs under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
