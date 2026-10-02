from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


REGULAR_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
for font_path in [REGULAR_FONT, BOLD_FONT]:
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    font_manager.fontManager.addfont(str(font_path))

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42


LABEL_SCRIPT = Path(
    "/work/tsuda/GAEZ/"
    "plot_groundwater_rural_population_14feature_input_sheet_explicit_labels.py"
)
spec = importlib.util.spec_from_file_location("explicit_input_sheet", LABEL_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load plotting module: {LABEL_SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.module.main()
