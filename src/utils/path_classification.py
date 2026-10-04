"""Classify surviving ray paths from their recorded interactions."""
import numpy as np

LOS_CLASSIFICATION = "surviving_path_without_interactions"


def los_from_interactions(interactions, num_paths):
    """Return (RX, TX) flags for links with a valid no-interaction path.

    Interactions have shape (depth, RX, TX, path); code 0 means no interaction.
    Valid paths occupy the first num_paths[RX, TX] slots. Padding also uses
    zeros and must never be mistaken for a direct path. Depth zero is valid
    for a trace containing only direct paths.
    """
    if interactions is None:
        raise ValueError(
            "Cannot classify LoS without path interactions. Regenerate refinement "
            "and link-budget calibration from raw paths with interactions; rerun "
            "path_gen.py only if the raw paths also lack interactions.")
    interactions = np.asarray(interactions)
    counts = np.asarray(num_paths)
    if interactions.ndim != 4 or counts.shape != interactions.shape[1:3]:
        raise ValueError("Interaction dimensions must be (depth, RX, TX, path) with (RX, TX) counts")
    if (not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or np.any(counts > interactions.shape[-1])):
        raise ValueError("Valid path counts must be integers within the stored path axis")
    direct = np.arange(interactions.shape[-1]) < counts[..., None]
    for depth in interactions:
        direct &= depth == 0
    return np.any(direct, axis=-1)
