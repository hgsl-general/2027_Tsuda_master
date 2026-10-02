from __future__ import annotations

import importlib.util
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm


BASE_SCRIPT = Path(
    "/work/tsuda/GAEZ/plot_signed_factor_maps_4groups_rural_population.py"
)
spec = importlib.util.spec_from_file_location("signed_map_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load plotting module: {BASE_SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


PANEL_POSITIONS = [
    [0.035, 0.525, 0.455, 0.315],
    [0.510, 0.525, 0.455, 0.315],
    [0.035, 0.190, 0.455, 0.315],
    [0.510, 0.190, 0.455, 0.315],
]


def plot_four_factors_fixed(stage_frame, background, config) -> None:
    group_columns = {
        factor: f"group_shap__{factor}"
        for factor in module.FACTOR_NAMES
    }
    missing = [column for column in group_columns.values() if column not in stage_frame]
    if missing:
        raise KeyError(f"Missing group SHAP columns: {missing}")

    combined_values = np.concatenate(
        [stage_frame[column].to_numpy(dtype=float) for column in group_columns.values()]
    )
    limit = module.color_limit(combined_values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    # Fixed canvas and explicit axes positions prevent Cartopy or the inline
    # backend from moving the right-hand panels outside the saved image.
    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    axes = [
        fig.add_axes(position, projection=module.PROJECTION)
        for position in PANEL_POSITIONS
    ]

    for panel_index, (ax, factor) in enumerate(
        zip(axes, module.FACTOR_NAMES),
        start=1,
    ):
        module.configure_map(ax)
        module.plot_background(ax, background)
        values = stage_frame[group_columns[factor]].to_numpy(dtype=float)
        ax.scatter(
            stage_frame["lon"],
            stage_frame["lat"],
            c=values,
            cmap=module.CMAP,
            norm=norm,
            s=3.6,
            marker="s",
            alpha=0.93,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=3,
        )

        # Put the full panel caption inside its own map. This cannot be clipped
        # by figure layout adjustments and remains visible in notebook output.
        ax.text(
            0.5,
            0.985,
            f"{chr(64 + panel_index)}. {factor}\n"
            f"positive {100 * np.mean(values > 0):.1f}%  |  "
            f"negative {100 * np.mean(values < 0):.1f}%  |  "
            f"mean |SHAP| {np.mean(np.abs(values)):.4f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold" if factor == "Rural population" else "normal",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#777777",
                "linewidth": 0.5,
                "alpha": 0.90,
            },
            zorder=10,
        )

    fig.text(
        0.5,
        0.965,
        f"{config['title']} — signed SHAP by production-factor group",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.930,
        f"{config['explanation']} | Rural population is panel D (lower right)",
        ha="center",
        va="top",
        fontsize=12.5,
    )

    colorbar_ax = fig.add_axes([0.17, 0.120, 0.66, 0.025])
    scalar = ScalarMappable(norm=norm, cmap=module.CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label(
        f"Signed grouped OOF SHAP ({config['unit']}) — "
        "blue lowers prediction; red raises prediction",
        fontsize=10,
        labelpad=5,
    )

    fig.text(
        0.5,
        0.045,
        f"Common scale: ±{limit:.4f}, clipped at the {module.COLOR_QUANTILE:.0%} percentile. "
        "Colored squares: 50,000 local-SHAP cells; pale gray: full 240,000-cell sample.",
        ha="center",
        fontsize=9.0,
        color="#666666",
    )

    output = module.RESULT_DIR / config["combined_output"]
    # Do not use bbox_inches='tight': the canvas itself is the exact boundary.
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"Saved fixed-canvas map: {output}")


module.plot_four_factors = plot_four_factors_fixed


if __name__ == "__main__":
    module.main()
