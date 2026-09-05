import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


input_path = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\cropland_no_population_raw_crop_wx_oof_shap.ipynb")
notebook = json.loads(input_path.read_text(encoding="utf-8"))


def display(value):
    if hasattr(value, "to_string"):
        print(value.to_string(index=False))
    else:
        print(value)


namespace = {"__name__": "__main__", "display": display}


class Tee(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)

for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))
    print(f"\n===== EXECUTE CELL {index} =====")
    buffer = Tee()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compile(source, f"cell_{index}", "exec"), namespace, namespace)
    except Exception:
        output = buffer.getvalue()
        print(output, end="")
        error_text = traceback.format_exc()
        print(error_text, end="")
        cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": (output + error_text).splitlines(True),
        }]
        cell["execution_count"] = index
        input_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        raise
    else:
        output = buffer.getvalue()
        print(output, end="")
        cell["outputs"] = []
        if output:
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": output.splitlines(True),
            })
        cell["execution_count"] = index

input_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nExecuted notebook saved: {input_path}")
