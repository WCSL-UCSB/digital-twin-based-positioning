"""Units and validation shared by calibration and receiver simulation."""
import numpy as np


def positive_float(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def positive_int(value, name):
    checked = positive_float(value, name)
    if not checked.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(checked)


def diagnostic_gains(rows, columns, sequence_length):
    """Ideal SNR gains, never waveform scaling factors."""
    rows = positive_int(rows, "array rows")
    columns = positive_int(columns, "array columns")
    length = positive_int(sequence_length, "pilot length")
    return float(10*np.log10(rows*columns)), float(10*np.log10(length))
