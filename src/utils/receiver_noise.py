"""Input-referred complex white receiver noise for sampled pilot reception."""
import hashlib
import json
import numpy as np
from src.utils.radio_power import positive_float


def recorded_noise_budget(channel_meta):
    """Use the budget that produced H, including legacy dBm-only records.

    Independent samples approximate a full-band complex receiver sampled at
    its noise bandwidth. The 1% tolerance admits rounded dt (3.33 ns at 300 MHz).
    Other bandwidth/sample-rate ratios need an explicit receiver filter model.
    """
    budget = channel_meta.get("source_metadata", {}).get("link_budget")
    if not budget:
        raise ValueError("Channel metadata has no original link budget; regenerate calibration/channels")
    bandwidth = positive_float(budget["bandwidth_hz"], "recorded receiver bandwidth")
    dt = positive_float(channel_meta["dt_s"], "recorded channel dt")
    if not np.isclose(bandwidth*dt, 1.0, rtol=0.01, atol=0):
        raise ValueError(
            "White receiver noise requires bandwidth_hz * dt_s approximately 1 "
            "(within 1%). Other sampling rates require a receiver filter/noise correlation model.")
    if "noise_power_w" in budget:
        power = positive_float(budget["noise_power_w"], "recorded noise power")
    elif "noise_power_dbm" in budget:
        power = positive_float(10.0 ** ((float(budget["noise_power_dbm"])-30)/10),
                               "recorded noise power")
    else:
        raise ValueError("Original link budget has no noise power; regenerate calibration/channels")
    if "noise_power_dbm" in budget and not np.isclose(
            10*np.log10(power)+30, float(budget["noise_power_dbm"]), rtol=0, atol=1e-6):
        raise ValueError("Recorded noise_power_w and noise_power_dbm disagree")
    return power, bandwidth, dt


def link_noise_rng(seed, tx_name, rx_name):
    """Stable stream per named link, independent of iteration order or coverage."""
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed or seed < 0:
        raise ValueError("receiver_noise.seed must be a nonnegative integer")
    key = json.dumps([str(tx_name), str(rx_name)], ensure_ascii=True).encode()
    words = np.frombuffer(hashlib.sha256(key).digest(), dtype="<u4").tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), *words])))


def complex_receiver_noise(shape, power_w, rng):
    """E[|n|^2] = power_w; each real component has variance power_w/2."""
    power_w = float(power_w)
    if not np.isfinite(power_w) or power_w < 0:
        raise ValueError("Noise power must be finite and nonnegative")
    if power_w == 0:
        return np.zeros(shape, dtype=np.complex128)
    if rng is None:
        raise ValueError("A random generator is required for nonzero noise")
    return np.sqrt(power_w/2) * (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))
