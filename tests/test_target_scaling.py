import numpy as np

from train_generator import scale_targets, inverse_scale_targets


def test_train_and_validation_use_same_target_transform():
    train_targets = np.array([[0.0, 50.0, 100.0], [20.0, 60.0, 80.0]], dtype=np.float32)
    val_targets = np.array([[10.0, 55.0, 90.0]], dtype=np.float32)

    scaled_train = scale_targets(train_targets)
    scaled_val = scale_targets(val_targets)

    np.testing.assert_allclose(scaled_train, train_targets / 255.0)
    np.testing.assert_allclose(scaled_val, val_targets / 255.0)
    np.testing.assert_allclose(inverse_scale_targets(scaled_train), train_targets)
    np.testing.assert_allclose(inverse_scale_targets(scaled_val), val_targets)
