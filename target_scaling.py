import numpy as np

OPENCV_LAB_SCALE = 255.0
LAB_LIGHTNESS_SCALE = 100.0
LAB_CHROMA_OFFSET = 128.0


def _reshape_lab_colors(values):
    if values.ndim == 0 or values.shape[-1] % 3 != 0:
        raise ValueError("LAB targets must contain groups of three channels")
    original_shape = values.shape
    if values.shape[-1] == 3:
        return values, original_shape
    return values.reshape(*values.shape[:-1], -1, 3), original_shape


def scale_targets(targets):
    """Decode OpenCV 8-bit Lab and normalize physical L*, a*, and b* values."""
    values = np.asarray(targets, dtype=np.float32)
    encoded_lab, original_shape = _reshape_lab_colors(values)
    standard_lab = np.empty_like(encoded_lab)
    standard_lab[..., 0] = encoded_lab[..., 0] * (
        LAB_LIGHTNESS_SCALE / OPENCV_LAB_SCALE
    )
    standard_lab[..., 1:] = encoded_lab[..., 1:] - LAB_CHROMA_OFFSET

    normalized_lab = np.empty_like(standard_lab)
    normalized_lab[..., 0] = standard_lab[..., 0] / LAB_LIGHTNESS_SCALE
    normalized_lab[..., 1:] = (
        standard_lab[..., 1:] + LAB_CHROMA_OFFSET
    ) / OPENCV_LAB_SCALE
    return normalized_lab.reshape(original_shape)


def inverse_scale_targets(targets):
    """Reconstruct normalized Lab values as standard physical CIE Lab."""
    values = np.asarray(targets, dtype=np.float32)
    normalized_lab, original_shape = _reshape_lab_colors(values)
    standard_lab = np.empty_like(normalized_lab)
    standard_lab[..., 0] = normalized_lab[..., 0] * LAB_LIGHTNESS_SCALE
    standard_lab[..., 1:] = (
        normalized_lab[..., 1:] * OPENCV_LAB_SCALE - LAB_CHROMA_OFFSET
    )
    return standard_lab.reshape(original_shape)


normalize_targets = scale_targets
denormalize_targets = inverse_scale_targets
