import numpy as np

TARGET_VALUE_SCALE = 255.0


def scale_targets(targets, target_scale=TARGET_VALUE_SCALE):
    """Scale LAB target values from 0..255 to approximately 0..1."""
    values = np.asarray(targets, dtype=np.float32)
    return values / float(target_scale)


def inverse_scale_targets(targets, target_scale=TARGET_VALUE_SCALE):
    """Convert normalized LAB targets back to the original 0..255 scale."""
    values = np.asarray(targets, dtype=np.float32)
    return values * float(target_scale)


normalize_targets = scale_targets
denormalize_targets = inverse_scale_targets
