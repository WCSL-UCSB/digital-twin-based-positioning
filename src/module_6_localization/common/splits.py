"""Train/val/test split construction with optional spatial blocking.

This module is NumPy-only at import time. Scikit-learn is imported lazily
inside the random-mode branch. Cached splits keep sample membership consistent
across consumers of the same dataset.

Split modes
-----------
random : sklearn train_test_split on sample indices (bit-exact with the
         legacy make_split for the same seed/test_size); val carved from
         train with a second seeded split. Measures dense-fingerprint
         deployment (test points interpolate between training neighbors).
block  : the XY plane is tiled into tile_size_m squares; whole tiles are
         assigned to test (then val) in seeded shuffled order until the
         requested sample fractions are reached. No tile spans two subsets,
         so test positions are spatially disjoint from training positions.
         Measures spatial generalization. Optional buffer_m guard band drops
         training samples within buffer_m of any test sample.

The canonical artifact
----------------------
get_or_create_split() caches the index triple under
    <split_dir>/split_<mode>_seed<S>_test<T>_val<V>[_tile<X>m][_buf<B>m]__<dataset_stem>.npz
keyed on the preprocessed dataset's basename, so a regenerated dataset can
never silently reuse a stale split. Every consumer (training harness, k-NN /
geometric / CRLB baselines, ablation) loads this artifact rather than
recomputing.
"""

import os

import numpy as np


def _random_split(n_samples, test_size, val_size, random_seed):
    from sklearn.model_selection import train_test_split

    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_seed
    )
    if val_size > 0:
        # Fraction of the remaining train pool that yields val_size of the total.
        rel = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=rel, random_state=random_seed
        )
    else:
        val_idx = np.array([], dtype=np.int64)
    return (np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64))


def _block_split(Y, test_size, val_size, random_seed, tile_size_m, buffer_m):
    xy = np.asarray(Y)[:, :2].astype(np.float64)
    n_samples = len(xy)

    mins = xy.min(axis=0)
    ix = np.floor((xy[:, 0] - mins[0]) / tile_size_m).astype(np.int64)
    iy = np.floor((xy[:, 1] - mins[1]) / tile_size_m).astype(np.int64)
    tile_key = ix * (iy.max() + 1) + iy

    tiles, tile_of_sample = np.unique(tile_key, return_inverse=True)
    tile_counts = np.bincount(tile_of_sample)

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(tiles))

    TRAIN, TEST, VAL = 0, 1, 2
    assign = np.full(len(tiles), TRAIN, dtype=np.int8)
    n_test_target = test_size * n_samples
    n_val_target = val_size * n_samples

    n_test = n_val = 0
    for t in order:
        if n_test < n_test_target:
            assign[t] = TEST
            n_test += tile_counts[t]
        elif n_val < n_val_target:
            assign[t] = VAL
            n_val += tile_counts[t]
        else:
            break

    sample_assign = assign[tile_of_sample]
    train_idx = np.where(sample_assign == TRAIN)[0]
    val_idx = np.where(sample_assign == VAL)[0]
    test_idx = np.where(sample_assign == TEST)[0]

    n_dropped = 0
    if buffer_m > 0 and len(test_idx) and len(train_idx):
        # Drop train samples within buffer_m of any test sample (guard band
        # against boundary leakage). Chunked brute force: 16k x 4k fp32 fits
        # easily in RAM when chunked.
        test_xy = xy[test_idx]
        keep = np.ones(len(train_idx), dtype=bool)
        chunk = 2048
        for s in range(0, len(train_idx), chunk):
            d2 = ((xy[train_idx[s:s + chunk], None, :]
                   - test_xy[None, :, :]) ** 2).sum(axis=-1)
            keep[s:s + chunk] = d2.min(axis=1) > buffer_m ** 2
        n_dropped = int((~keep).sum())
        train_idx = train_idx[keep]

    return train_idx, val_idx, test_idx, n_dropped


def make_split(Y, test_size, val_size, random_seed,
               split_mode="random", tile_size_m=25.0, buffer_m=0.0):
    """Build a train/val/test index split.

    Returns a dict with train_idx/val_idx/test_idx (int64 arrays) plus the
    parameters and the achieved fractions (block mode quantizes to whole
    tiles, so achieved != nominal by up to ~1 tile of samples).
    """
    n_samples = len(Y)
    n_dropped = 0
    if split_mode == "random":
        train_idx, val_idx, test_idx = _random_split(
            n_samples, test_size, val_size, random_seed)
    elif split_mode == "block":
        train_idx, val_idx, test_idx, n_dropped = _block_split(
            Y, test_size, val_size, random_seed, tile_size_m, buffer_m)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r} "
                         "(expected 'random' or 'block')")

    overlap = (set(train_idx) & set(test_idx)) | (set(train_idx) & set(val_idx)) \
        | (set(val_idx) & set(test_idx))
    if overlap:
        raise RuntimeError(f"Split bug: {len(overlap)} indices in two subsets")

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_mode": split_mode,
        "test_size": float(test_size),
        "val_size": float(val_size),
        "random_seed": int(random_seed),
        "tile_size_m": float(tile_size_m),
        "buffer_m": float(buffer_m),
        "n_samples": int(n_samples),
        "n_train_dropped_by_buffer": int(n_dropped),
        "achieved_test_frac": len(test_idx) / n_samples,
        "achieved_val_frac": len(val_idx) / n_samples,
    }


def split_filename(dataset_basename, test_size, val_size, random_seed,
                   split_mode, tile_size_m=25.0, buffer_m=0.0):
    stem = os.path.splitext(dataset_basename)[0]
    name = (f"split_{split_mode}_seed{int(random_seed)}"
            f"_test{int(round(test_size * 100))}"
            f"_val{int(round(val_size * 100))}")
    if split_mode == "block":
        name += f"_tile{tile_size_m:g}m"
        if buffer_m > 0:
            name += f"_buf{buffer_m:g}m"
    return f"{name}__{stem}.npz"


def get_or_create_split(split_dir, dataset_basename, Y, test_size, val_size,
                        random_seed, split_mode="random",
                        tile_size_m=25.0, buffer_m=0.0):
    """Load the cached split artifact for these parameters, or create it.

    Pass Y=None to require an existing artifact (consumers that cannot build
    one, e.g. the Sionna-env baselines if training never ran, get a clear
    error instead of a silently different split).
    """
    fname = split_filename(dataset_basename, test_size, val_size, random_seed,
                           split_mode, tile_size_m, buffer_m)
    path = os.path.join(split_dir, fname)

    if os.path.exists(path):
        data = np.load(path, allow_pickle=False)
        split = {k: data[k] for k in data.files}
        for k in ("train_idx", "val_idx", "test_idx"):
            split[k] = split[k].astype(np.int64)
        if Y is not None and int(split["n_samples"]) != len(Y):
            raise RuntimeError(
                f"Split artifact {path} was built for n={int(split['n_samples'])} "
                f"samples but the dataset has n={len(Y)}. The preprocessed "
                "dataset changed without a new basename — delete the stale "
                "artifact or regenerate the dataset."
            )
        print(f"  Loaded split artifact: {fname} "
              f"(train={len(split['train_idx'])}, val={len(split['val_idx'])}, "
              f"test={len(split['test_idx'])})")
        return split

    if Y is None:
        raise FileNotFoundError(
            f"Split artifact not found: {path}\n"
            "Create it in the pipeline environment first, e.g.:\n"
            "  python src/module_6_localization/common/splits.py --preprocessed <dataset.npz> "
            f"--split-mode {split_mode} --seed {random_seed}\n"
            "or run the positioning baseline with the same split parameters."
        )

    split = make_split(Y, test_size, val_size, random_seed,
                       split_mode, tile_size_m, buffer_m)
    os.makedirs(split_dir, exist_ok=True)
    np.savez(path, dataset_basename=np.array([dataset_basename]), **{
        k: (np.asarray(v) if not np.isscalar(v) else np.array(v))
        for k, v in split.items() if not isinstance(v, str)
    }, split_mode=np.array([split["split_mode"]]))
    print(f"  Created split artifact: {path}")
    print(f"    train={len(split['train_idx'])}  val={len(split['val_idx'])}  "
          f"test={len(split['test_idx'])}  "
          f"(achieved test {split['achieved_test_frac']:.3f}, "
          f"val {split['achieved_val_frac']:.3f}, "
          f"buffer-dropped {split['n_train_dropped_by_buffer']})")
    return split


def _main():
    import argparse

    p = argparse.ArgumentParser(
        description="Pre-create a split artifact for a preprocessed dataset.")
    p.add_argument("--preprocessed", required=True,
                   help="Path to *_preprocessed_<scenario>.npz")
    p.add_argument("--split-mode", default="block",
                   choices=["random", "block"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--tile-size-m", type=float, default=25.0)
    p.add_argument("--buffer-m", type=float, default=0.0)
    args = p.parse_args()

    data = np.load(args.preprocessed, allow_pickle=False)
    Y = data["Y"]
    split_dir = os.path.join(os.path.dirname(os.path.abspath(args.preprocessed)),
                             "splits")
    get_or_create_split(split_dir, os.path.basename(args.preprocessed), Y,
                        args.test_size, args.val_size, args.seed,
                        args.split_mode, args.tile_size_m, args.buffer_m)


if __name__ == "__main__":
    _main()
