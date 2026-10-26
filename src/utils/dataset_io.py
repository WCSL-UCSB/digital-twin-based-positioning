"""Shared input resolution + provenance sidecars for every pipeline stage.

Why this exists
---------------
`data/` is a group-shared directory (see docs/data_layout.md), so several
people write timestamped outputs for the same scenario into the same folders.
Two things have to hold for that to be safe:

1. "Latest" must mean the most recent *run*, not the most recent inode change.
   `os.path.getctime` is reset by cp/mv/chmod/Globus, so after any transfer
   every file looks equally new. We order by the `YYYYMMDD_HHMMSS_` prefix
   every stage already writes into its filename instead.

2. A run must be able to pin exactly which upstream file it consumes, so
   "train on the dataset X generated last week" is reproducible no matter what
   anyone writes into the shared folder afterwards. Each stage reads an
   optional key from the `inputs:` block of the scenario YAML (see
   `configs/scenarios/_template.yaml`); when it is null we fall back to (1).

Every stage also writes a `<output>.json` sidecar next to its `.npz` with the
resolved input paths, git commit, user/host and the full resolved config. It is
plain JSON so it can be read from any env (the `.npz` metadata is pickled and
cannot be unpickled across the NumPy 1.x / 2.x env split).

stdlib-only on purpose: imported from the Sionna, G2SM and TF envs alike.
"""

import datetime as _dt
import getpass
import glob
import json
import os
import re
import socket
import subprocess

_STAMP_RE = re.compile(r"^(\d{8}_\d{6})_")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def file_stamp(path):
    """The `YYYYMMDD_HHMMSS` prefix of a pipeline output filename, or ''."""
    m = _STAMP_RE.match(os.path.basename(path))
    return m.group(1) if m else ""


def _sort_key(path):
    # Stamped files order by stamp; unstamped ones fall back to mtime and
    # always lose against a stamped file.
    return (file_stamp(path), os.path.getmtime(path))


def find_latest(pattern):
    """Newest file matching `pattern` by filename timestamp, or None."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=_sort_key)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def pinned(cfg, key):
    """Absolute path for `cfg.inputs.<key>`, or None when absent/null.

    A relative pin resolves against `<cfg.project_dir>/data`, so
    `inputs.paths_npz: path_data/20260915_120000_foo_paths.npz` works from any
    checkout. Works on OmegaConf DictConfigs (struct mode included) and dicts.
    """
    if cfg is None or not hasattr(cfg, "get"):
        return None
    inputs = cfg.get("inputs", None)
    if not inputs:
        return None
    val = inputs.get(key, None)
    if val is None or str(val).strip() == "":
        return None
    path = os.path.expanduser(str(val).strip())
    if not os.path.isabs(path):
        project_dir = cfg.get("project_dir", None)
        if project_dir:
            path = os.path.join(str(project_dir), "data", path)
    return os.path.abspath(path)


def resolve_input(pattern, explicit=None, *, label="input", required=True):
    """Pick the upstream file for a stage and say which one was chosen.

    Parameters
    ----------
    pattern : str
        Glob for the "latest" fallback, e.g. `data/path_data/*_foo_paths.npz`.
    explicit : str or None
        A pinned file (normally from `pinned(cfg, key)`). Wins over `pattern`
        when set; must exist.
    label : str
        Human-readable name for the log line ("raw paths", "BDP dataset", ...).
    required : bool
        If False, a missing fallback match returns None instead of raising.
        A pinned file that does not exist always raises.
    """
    if explicit:
        path = os.path.abspath(os.path.expanduser(str(explicit)))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Pinned {label} does not exist: {path}\n"
                f"(from inputs pin {explicit!r})")
        print(f"Loading {label} (pinned via inputs): {path}")
        return path

    files = glob.glob(pattern)
    if not files:
        if required:
            raise FileNotFoundError(f"No {label} matching: {pattern}")
        print(f"No {label} matching: {pattern} (optional, skipping)")
        return None
    path = os.path.abspath(max(files, key=_sort_key))
    how = ("only match" if len(files) == 1
           else f"newest of {len(files)} by filename timestamp")
    print(f"Loading {label} ({how}): {path}")
    return path


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------

def _git_info(repo_dir):
    def run(*args):
        return subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except Exception as e:  # not a repo, git missing, timeout, ...
        return {"commit": None, "branch": None, "dirty": None,
                "error": f"{type(e).__name__}: {e}"}


def _to_plain(obj):
    """Best-effort conversion of OmegaConf / numpy objects to JSON types."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


def _json_default(o):
    # numpy scalars / arrays, Paths, sets, anything else -> something readable
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _strip_nested_sources(meta):
    """Drop `source_*` / `config_snapshot` keys: inputs are already linked by
    path and carry their own sidecar, so repeating their metadata only bloats
    every downstream file."""
    if not isinstance(meta, dict):
        return meta
    return {k: v for k, v in meta.items()
            if not k.startswith("source_") and k != "config_snapshot"}


def sidecar_path(npz_path):
    root, _ = os.path.splitext(npz_path)
    return root + ".json"


def write_sidecar(npz_path, *, stage, cfg=None, inputs=None, summary=None):
    """Write `<npz_path minus .npz>.json` describing how the file was made.

    Never raises: a provenance failure must not lose a finished ray trace.

    Parameters
    ----------
    npz_path : str
        The output file the sidecar describes.
    stage : str
        Script name, e.g. "path_gen".
    cfg : OmegaConf DictConfig or dict, optional
        The resolved scenario config; stored in full.
    inputs : dict[str, str or None], optional
        Upstream files consumed, keyed like the `inputs:` block
        (`{"paths_npz": "/abs/path"}`).
    summary : dict, optional
        Stage-specific metadata (usually the dict also saved inside the npz).
        Nested `source_*` entries are stripped, see `_strip_nested_sources`.
    """
    try:
        cfg_plain = _to_plain(cfg) if cfg is not None else None
        project_dir = None
        if isinstance(cfg_plain, dict):
            project_dir = cfg_plain.get("project_dir")
        if not project_dir:
            project_dir = os.path.dirname(os.path.abspath(__file__))

        record = {
            "npz": os.path.basename(npz_path),
            "stage": stage,
            "scenario": (cfg_plain or {}).get("scenario_name")
                        if isinstance(cfg_plain, dict) else None,
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "git": _git_info(project_dir),
            "inputs": {k: (os.path.abspath(v) if v else None)
                       for k, v in (inputs or {}).items()},
            "summary": _strip_nested_sources(summary),
            "config": cfg_plain,
        }
        out = sidecar_path(npz_path)
        with open(out, "w") as fh:
            json.dump(record, fh, indent=2, default=_json_default)
        print(f"Provenance: {out}")
        return out
    except Exception as e:
        print(f"[warn] could not write provenance sidecar for {npz_path}: "
              f"{type(e).__name__}: {e}")
        return None
