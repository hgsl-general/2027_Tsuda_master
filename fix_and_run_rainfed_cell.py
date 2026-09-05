import json
import subprocess
import sys
from pathlib import Path

GAEZ_DIR = Path(r"C:\masterresearch\Comparative_advantage\GAEZ")
NOTEBOOK_PATH = GAEZ_DIR / "datalink.ipynb"
RUN_SCRIPT_PATH = GAEZ_DIR / "run_gaez_climate_potential_rainfed_download_from_notebook_cell.py"
PYTHON_EXE = Path(r"C:\Users\tsuda\miniconda3\python.exe")
marker = "Download GAEZ climate potential yield under rain-fed conditions"

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
updated_source = None
for cell in notebook.get("cells", []):
    source_text = "".join(cell.get("source", []))
    if marker in source_text:
        source_text = source_text.replace('CROP_NAME_COL = "LABEL"', 'CROP_NAME_COL = "Crop name"')
        cell["source"] = source_text.splitlines(keepends=True)
        updated_source = source_text
        break

if updated_source is None:
    raise RuntimeError("Rainfed download cell was not found in datalink.ipynb")

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
RUN_SCRIPT_PATH.write_text(updated_source, encoding="utf-8")
print(f"Fixed notebook cell and runner script: {RUN_SCRIPT_PATH}")

completed = subprocess.run([str(PYTHON_EXE), str(RUN_SCRIPT_PATH)], check=False)
sys.exit(completed.returncode)
