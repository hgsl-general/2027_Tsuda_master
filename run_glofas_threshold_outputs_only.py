import importlib.util
from pathlib import Path

script_path = Path(r"C:\masterresearch\Comparative_advantage\GAEZ\glofas_reliable_river_pipeline.py")
spec = importlib.util.spec_from_file_location("glofas_pipeline", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.ensure_dirs()
period_label = "2020"
source_p10_path = module.ANNUAL_SOURCE_DIR / "glofas_mean_annual_p10_discharge_2020_0p05.npy"
if not source_p10_path.exists():
    raise FileNotFoundError(source_p10_path)
module.make_reliable_masks_and_distances(source_p10_path, period_label)
module.write_run_metadata(period_label, source_p10_path)
print("Regenerated reliable-river masks and distances for thresholds:", module.THRESHOLDS_M3S)
