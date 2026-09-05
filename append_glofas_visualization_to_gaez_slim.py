import base64
import contextlib
import io
import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\GAEZ_slim.ipynb")
MARKER = "Visual check: GloFAS reliable river / distance rasters"

cell_source = r'''
# ============================================================
# Visual check: GloFAS reliable river / distance rasters
#
# Uses 2020 annual p10 discharge generated from GloFAS daily river discharge.
# DS=4 means plotting every 4th cell for quick visual inspection.
# Set DS=1 if you want full-resolution plotting inside the notebook.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


GLOFAS_PROCESSED_DIR = Path(r"C:\masterresearch\Comparative_advantage\GloFAS\processed_5min")
THRESHOLDS_M3S = [1, 10, 100]
DS = 4
EXTENT = [-180, 180, -90, 90]

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "DejaVu Sans"]


def load_map(filename: str, ds: int = DS) -> np.ndarray:
    path = GLOFAS_PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r")
    return np.asarray(array[::ds, ::ds])


def finite_quantile(array: np.ndarray, q: float) -> float:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return np.nan
    return float(np.nanquantile(values, q))


def format_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)


p10 = load_map("p10_discharge_max_5min_2020.npy")
reliable_masks = {
    threshold: load_map(f"reliable_river_p10_gt_{threshold}_m3s_5min_2020.npy")
    for threshold in THRESHOLDS_M3S
}
distance_maps = {
    threshold: load_map(f"distance_to_reliable_river_p10_gt_{threshold}_m3s_km_5min_2020.npy")
    for threshold in THRESHOLDS_M3S
}

print("GloFAS 2020 visual check")
print(f"plotting shape after downsampling: {p10.shape}; DS={DS}")
print(f"source directory: {GLOFAS_PROCESSED_DIR}")
for threshold in THRESHOLDS_M3S:
    mask_cells = int(np.asarray(reliable_masks[threshold]).sum())
    distance_q95 = finite_quantile(distance_maps[threshold], 0.95)
    print(f"threshold > {threshold:>3} m3/s | reliable plotted cells={mask_cells:,} | distance q95={distance_q95:,.1f} km")

# 1) River water amount and reliable river masks.
fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
axes = axes.ravel()

p10_log = np.where(np.isfinite(p10) & (p10 > 0), np.log10(1 + p10), np.nan)
p10_image = axes[0].imshow(p10_log, extent=EXTENT, origin="upper", cmap="viridis")
format_axis(axes[0], "GloFAS 2020 annual p10 discharge\nlog10(1 + m3/s), max within 5-min cell")
fig.colorbar(p10_image, ax=axes[0], shrink=0.78, label="log10(1 + m3/s)")

for ax, threshold in zip(axes[1:], THRESHOLDS_M3S):
    mask_plot = np.where(reliable_masks[threshold] > 0, 1.0, np.nan)
    mask_image = ax.imshow(mask_plot, extent=EXTENT, origin="upper", cmap="Blues", vmin=0, vmax=1)
    format_axis(ax, f"Reliable river cells\np10 discharge > {threshold} m3/s")
    fig.colorbar(mask_image, ax=ax, shrink=0.78, ticks=[0, 1], label="reliable river")

plt.show()

# 2) Distance to reliable rivers for each threshold.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

for ax, threshold in zip(axes, THRESHOLDS_M3S):
    distance = distance_maps[threshold]
    vmax = finite_quantile(distance, 0.95)
    distance_image = ax.imshow(
        distance,
        extent=EXTENT,
        origin="upper",
        cmap="magma_r",
        vmin=0,
        vmax=vmax,
    )
    format_axis(ax, f"Distance to reliable river\np10 > {threshold} m3/s")
    fig.colorbar(distance_image, ax=ax, shrink=0.78, label="km; clipped at p95")

plt.show()
'''.strip() + "\n"

if not NOTEBOOK_PATH.exists():
    raise FileNotFoundError(NOTEBOOK_PATH)

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

# Execute the cell in a script context and capture stdout + figures as notebook outputs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.close("all")
stdout_buffer = io.StringIO()
namespace = {"__name__": "__main__"}
with contextlib.redirect_stdout(stdout_buffer):
    exec(cell_source, namespace)

outputs = []
stdout = stdout_buffer.getvalue()
if stdout:
    outputs.append({
        "output_type": "stream",
        "name": "stdout",
        "text": stdout.splitlines(keepends=True),
    })

for figure_number in plt.get_fignums():
    figure = plt.figure(figure_number)
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format="png", bbox_inches="tight", dpi=140)
    encoded = base64.b64encode(png_buffer.getvalue()).decode("ascii")
    outputs.append({
        "output_type": "display_data",
        "data": {"image/png": encoded},
        "metadata": {},
    })
plt.close("all")

execution_counts = [cell.get("execution_count") for cell in notebook.get("cells", []) if isinstance(cell.get("execution_count"), int)]
execution_count = (max(execution_counts) + 1) if execution_counts else 1
new_cell = {
    "cell_type": "code",
    "execution_count": execution_count,
    "metadata": {},
    "outputs": outputs,
    "source": cell_source.splitlines(keepends=True),
}

replaced = False
for cell in notebook.get("cells", []):
    if MARKER in "".join(cell.get("source", [])):
        cell.clear()
        cell.update(new_cell)
        replaced = True
        break
if not replaced:
    notebook.setdefault("cells", []).append(new_cell)

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{'Updated' if replaced else 'Appended'} visualization cell in {NOTEBOOK_PATH}")
print(f"Embedded outputs: {len(outputs)}")
