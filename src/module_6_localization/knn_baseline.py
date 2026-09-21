"""Weighted k-NN fingerprint baseline (the non-parametric floor).

Operates on the SAME preprocessed X tensor and the SAME cached split artifact
as the CNN/transformer models, so its metrics.json lands in the same
test-set group in compare_runs.py. If a learned model does not clearly beat
weighted k-NN, it is only memorizing the radio map.

Distances are exact (chunked BLAS matmul, no approximation):
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
Cosine ranking is obtained by L2-normalizing rows first (Euclidean on
normalized rows is rank-equivalent; cos_dist = d^2/2 supplies the weights).

Run (tf_loc env; ~minutes on CPU):
    python src/module_6_localization/knn_baseline.py --config-name=<scenario>
    python src/module_6_localization/knn_baseline.py --config-name=<scenario> \
        +training.split_mode=block
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from common import data as data_mod
from common import metrics as metrics_mod
from common import splits as splits_mod

load_dotenv()

KS = (1, 3, 5, 10)
METRICS = ("euclidean", "cosine")
TEST_CHUNK = 512


def _cfg_get(cfg, key, default):
    v = OmegaConf.select(cfg, key)
    return default if v is None else v


def knn_topk(X_train_flat, X_test_flat, k_max):
    """Exact top-k_max nearest training rows for every test row."""
    n_train = len(X_train_flat)
    nt2 = (X_train_flat ** 2).sum(axis=1)                    # (n_train,)
    top_d = np.empty((len(X_test_flat), k_max), dtype=np.float32)
    top_i = np.empty((len(X_test_flat), k_max), dtype=np.int64)

    for s in range(0, len(X_test_flat), TEST_CHUNK):
        q = X_test_flat[s:s + TEST_CHUNK]
        d2 = (q ** 2).sum(axis=1)[:, None] + nt2[None, :] \
            - 2.0 * (q @ X_train_flat.T)
        np.maximum(d2, 0.0, out=d2)
        idx = np.argpartition(d2, k_max - 1, axis=1)[:, :k_max]
        dpart = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(dpart, axis=1)
        top_i[s:s + TEST_CHUNK] = np.take_along_axis(idx, order, axis=1)
        top_d[s:s + TEST_CHUNK] = np.sqrt(
            np.take_along_axis(dpart, order, axis=1))
    return top_d, top_i


@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir
    data_dir = os.path.join(project_dir, "data", "preprocessed_datasets",
                            cfg.scenario_name)

    test_size = float(_cfg_get(cfg, "training.test_size", 0.2))
    val_size = float(_cfg_get(cfg, "training.val_size", 0.1))
    seed = int(_cfg_get(cfg, "training.random_seed", 71))
    split_mode = str(_cfg_get(cfg, "training.split_mode", "random"))
    tile_size_m = float(_cfg_get(cfg, "training.tile_size_m", 25.0))
    buffer_m = float(_cfg_get(cfg, "training.buffer_m", 0.0))

    X, Y, extras = data_mod.load_preprocessed(data_dir, cfg.scenario_name)
    split = splits_mod.get_or_create_split(
        os.path.join(data_dir, "splits"),
        extras.get("_dataset_basename", "dataset"), Y,
        test_size=test_size, val_size=val_size, random_seed=seed,
        split_mode=split_mode, tile_size_m=tile_size_m, buffer_m=buffer_m,
    )
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    K_bs = X.shape[-1]
    Xf = X.reshape(len(X), -1)            # view, no copy
    X_train = np.ascontiguousarray(Xf[train_idx], dtype=np.float32)
    X_test = np.ascontiguousarray(Xf[test_idx], dtype=np.float32)
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    del X, Xf                             # drop the 5 GB original promptly
    print(f"k-NN: {len(X_train)} train x {len(X_test)} test, "
          f"{X_train.shape[1]} features")

    tag_suffix = ""
    if split_mode == "block":
        tag_suffix = f"__block{tile_size_m:g}m"
        if buffer_m > 0:
            tag_suffix += f"__buf{buffer_m:g}m"

    k_max = max(KS)
    for metric in METRICS:                # euclidean MUST run first:
        if metric == "cosine":            # rows are then normalized in place
            for A in (X_train, X_test):
                A /= np.maximum(
                    np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        top_d, top_i = knn_topk(X_train, X_test, k_max)
        if metric == "cosine":
            top_d = 0.5 * top_d ** 2          # actual cosine distance

        for k in KS:
            w = 1.0 / (top_d[:, :k] + 1e-6)
            pred = (w[..., None] * Y_train[top_i[:, :k]]).sum(axis=1) \
                / w.sum(axis=1)[:, None]

            run_tag = f"baseline_knn_k{k}_{metric}{tag_suffix}"
            out_dir = os.path.join(project_dir, "models",
                                   cfg.scenario_name, run_tag)
            os.makedirs(out_dir, exist_ok=True)

            provenance = {
                "scenario": str(cfg.scenario_name),
                "run_tag": run_tag,
                "model_name": f"knn_k{k}_{metric}",
                "backbone": "knn",
                "output_mode": "knn",
                "fusion_mode": "none",
                "dataset_basename": extras.get("_dataset_basename", "?"),
                "split_mode": split_mode,
                "test_size": test_size,
                "val_size": val_size,
                "tile_size_m": tile_size_m,
                "buffer_m": buffer_m,
                "random_seed": seed,
                "k": k,
                "distance_metric": metric,
            }
            m = metrics_mod.compute_all_metrics(
                pred, Y_test, extras=extras, test_idx=test_idx,
                provenance=provenance)
            metrics_mod.print_metrics(m, f"(knn k={k} {metric})")
            metrics_mod.save_metrics_json(out_dir, m)
            errors = np.sqrt(((pred - Y_test) ** 2).sum(axis=1))
            np.savez(os.path.join(out_dir, "results.npz"),
                     predictions=pred, Y_test=Y_test, errors=errors,
                     test_idx=test_idx,
                     output_mode=np.array(["knn"]),
                     fusion_mode=np.array(["none"]),
                     split_mode=np.array([split_mode]),
                     K=np.array([K_bs]))


if __name__ == "__main__":
    main()
