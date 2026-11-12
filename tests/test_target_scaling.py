import numpy as np

from target_scaling import scale_targets, inverse_scale_targets
from color_utils import standard_lab_to_opencv


def test_train_and_validation_use_same_target_transform():
    train_targets = np.array([[255.0, 128.0, 255.0], [51.0, 188.0, 0.0]], dtype=np.float32)
    val_targets = np.array([[127.5, 128.0, 128.0]], dtype=np.float32)

    scaled_train = scale_targets(train_targets)
    scaled_val = scale_targets(val_targets)

    np.testing.assert_allclose(scaled_train[:, 0], [1.0, 0.2])
    np.testing.assert_allclose(scaled_train[:, 1], [128.0 / 255.0, 188.0 / 255.0])
    np.testing.assert_allclose(scaled_train[:, 2], [1.0, 0.0])
    np.testing.assert_allclose(scaled_val, [[0.5, 128.0 / 255.0, 128.0 / 255.0]])
    np.testing.assert_allclose(
        inverse_scale_targets(scaled_train),
        [[100.0, 0.0, 127.0], [20.0, 60.0, -128.0]],
    )
    np.testing.assert_allclose(
        inverse_scale_targets(scaled_val),
        [[50.0, 0.0, 0.0]],
    )


def test_scale_and_inverse_support_flattened_palettes():
    encoded_palette = np.arange(15, dtype=np.float32)

    normalized = scale_targets(encoded_palette)
    reconstructed = inverse_scale_targets(normalized)

    assert normalized.shape == encoded_palette.shape
    assert reconstructed.shape == encoded_palette.shape
    np.testing.assert_allclose(reconstructed[::3], encoded_palette[::3] * (100.0 / 255.0))
    np.testing.assert_allclose(reconstructed[1::3], encoded_palette[1::3] - 128.0)
    np.testing.assert_allclose(reconstructed[2::3], encoded_palette[2::3] - 128.0)


def test_reconstructed_standard_lab_can_be_encoded_for_opencv():
    encoded_palette = np.array([[255.0, 128.0, 255.0], [127.5, 128.0, 128.0]])

    standard_lab = inverse_scale_targets(scale_targets(encoded_palette))
    opencv_lab = standard_lab_to_opencv(standard_lab)

    np.testing.assert_array_equal(opencv_lab, [[255, 128, 255], [128, 128, 128]])
