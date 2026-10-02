from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient


class CheckpointClient(NotebookClient):
    def __init__(self, *args, notebook_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.notebook_path = notebook_path

    def execute_cell(
        self,
        cell,
        cell_index,
        execution_count=None,
        store_history=True,
    ):
        if cell.cell_type != "code":
            return super().execute_cell(
                cell,
                cell_index,
                execution_count=execution_count,
                store_history=store_history,
            )

        first_line = " ".join(cell.source.strip().splitlines()[:1])[:110]
        started = time.monotonic()
        print(
            f"START cell {cell_index:02d}: {first_line}",
            flush=True,
        )
        try:
            result = super().execute_cell(
                cell,
                cell_index,
                execution_count=execution_count,
                store_history=store_history,
            )
        except Exception:
            nbformat.write(self.nb, self.notebook_path)
            print(f"FAILED cell {cell_index:02d}", flush=True)
            traceback.print_exc()
            raise

        elapsed = time.monotonic() - started
        nbformat.write(self.nb, self.notebook_path)
        print(
            f"DONE  cell {cell_index:02d} ({elapsed:.1f} s)",
            flush=True,
        )

        stream_text = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                stream_text.append(output.get("text", ""))
        if stream_text:
            tail = "".join(stream_text)[-2400:].strip()
            if tail:
                print(f"OUTPUT TAIL cell {cell_index:02d}:\n{tail}", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--kernel", default="research")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    client = CheckpointClient(
        notebook,
        notebook_path=notebook_path,
        timeout=-1,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(notebook_path.parent)}},
        allow_errors=False,
    )

    started = time.monotonic()
    client.execute()
    nbformat.write(client.nb, notebook_path)
    print(
        f"NOTEBOOK COMPLETED in {(time.monotonic() - started) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
