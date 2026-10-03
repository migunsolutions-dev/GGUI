"""Explicit dependency bundle for standalone generated radial remap scripts."""
from pathlib import Path
import shutil

RUNTIME_FILES = (
    'remap_native_io.py', 'remap_snapshot_1d.py', 'remap_fields_2d.py', 'remap_handoff_1d.py',
    'models.py', 'completion_1d.py', 'validation/__init__.py',
    'validation/map_1d.py', 'validation/metrics.py', 'validation/probes.py',
)


def copy_remap_runtime(case_dir):
    source=Path(__file__).parent
    missing=[name for name in RUNTIME_FILES if not (source/name).is_file()]
    if missing:
        raise RuntimeError('Missing standalone remap dependencies: '+', '.join(missing))
    for name in RUNTIME_FILES:
        target=Path(case_dir)/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
