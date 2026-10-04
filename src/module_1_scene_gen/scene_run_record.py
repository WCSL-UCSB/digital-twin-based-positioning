"""Small generation record shared by the two scene-generation environments."""
import hashlib
import json
import os
import platform
import subprocess
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from omegaconf import OmegaConf


def record_generation(cfg, scene_dir, details=None):
    root = Path(cfg.project_dir)
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    packages = {}
    for name in ('scene_generation', 'sionna', 'sionna-rt', 'mitsuba', 'drjit',
                 'numpy', 'pyproj', 'rasterio', 'osmnx', 'open3d', 'pyvista'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    snapshot = OmegaConf.to_container(cfg, resolve=True)
    folder = Path(scene_dir)
    with zipfile.ZipFile(folder/'generation_code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for pattern in ('src/module_1_scene_gen/*.py', 'configs/scenarios/profiles/*.yaml'):
            for p in root.glob(pattern):
                archive.write(p, str(p.relative_to(root)))
        archive.writestr('config_resolved.yaml', OmegaConf.to_yaml(cfg, resolve=True))
    record = {
        'schema_version': 1, 'scene_id': cfg.selected_id, 'stage': 'scene_generation',
        'status': 'generated_pending_validation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'user': os.environ.get('USER'), 'job_id': os.environ.get('SLURM_JOB_ID'),
        'python': platform.python_version(), 'packages': packages,
        'generation_code_sha256': hashlib.sha256((folder/'generation_code.zip').read_bytes()).hexdigest(),
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty': bool(git('status', '--porcelain')),
        'code_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (root/'src/module_1_scene_gen').glob('*.py')},
        'config': snapshot, 'details': details or {},
        'deployment_status': 'awaiting_real_deployment_coordinates',
        'ray_tracing_status': 'not_started',
    }
    folder = Path(scene_dir)
    (folder/'generation.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    OmegaConf.save(cfg, folder/'config_resolved.yaml')
    return record
