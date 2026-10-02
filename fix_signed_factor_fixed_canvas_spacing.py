from pathlib import Path


SCRIPT = Path("/work/tsuda/GAEZ/plot_signed_factor_maps_4groups_fixed_canvas.py")
text = SCRIPT.read_text(encoding="utf-8")

replacements = [
    (
        'colorbar_ax = fig.add_axes([0.17, 0.095, 0.66, 0.025])',
        'colorbar_ax = fig.add_axes([0.17, 0.120, 0.66, 0.025])',
    ),
    (
        '''    fig.text(
        0.5,
        0.050,
        f"All four panels share one scale (±{limit:.4f}); values are clipped at the "
        f"{module.COLOR_QUANTILE:.0%} percentile for display.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.025,
        "Colored squares: 50,000 local-SHAP cells. Pale gray squares: full 240,000-cell analysis sample.",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )''',
        '''    fig.text(
        0.5,
        0.045,
        f"Common scale: ±{limit:.4f}, clipped at the {module.COLOR_QUANTILE:.0%} percentile. "
        "Colored squares: 50,000 local-SHAP cells; pale gray: full 240,000-cell sample.",
        ha="center",
        fontsize=9.0,
        color="#666666",
    )''',
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one replacement target, found {count}")
    text = text.replace(old, new, 1)

compile(text, str(SCRIPT), "exec")
SCRIPT.write_text(text, encoding="utf-8")
print(f"Updated spacing: {SCRIPT}")
