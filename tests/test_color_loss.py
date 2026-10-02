import pytest
import torch

from color_loss import CIEDE2000Loss, ciede2000, opencv_lab_to_standard_lab
from palette_network import PaletteNetwork


def test_opencv_lab_decode_produces_standard_lab():
    encoded = torch.tensor([[0.5, 128.0 / 255.0, 128.0 / 255.0]])

    decoded = opencv_lab_to_standard_lab(encoded)

    torch.testing.assert_close(decoded, torch.tensor([[50.0, 0.0, 0.0]]))


def test_torch_ciede2000_matches_reference_implementation():
    lab1 = torch.tensor([50.0, 2.6772, -79.7751])
    lab2 = torch.tensor([50.0, 0.0, -82.7485])

    actual = ciede2000(lab1, lab2).item()

    assert actual == pytest.approx(2.0425, abs=1e-4)


def test_ciede2000_loss_has_finite_gradients():
    prediction = torch.tensor([[0.5, 0.6, 0.3]], requires_grad=True)
    target = torch.tensor([[0.55, 0.5, 0.4]])

    loss = CIEDE2000Loss()(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


def test_ciede2000_loss_has_finite_gradients_for_neutral_colors():
    prediction = torch.tensor([[0.5, 128.0 / 255.0, 128.0 / 255.0]], requires_grad=True)
    target = torch.tensor([[0.55, 128.0 / 255.0, 128.0 / 255.0]])

    loss = CIEDE2000Loss()(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_palette_network_predicts_one_bounded_color_per_call():
    model = PaletteNetwork(input_size=4)
    embedding = torch.zeros((2, 4))

    predictions = model.generate_palette(embedding)

    assert predictions.shape == (2, 5, 3)
    assert torch.all((predictions >= 0.0) & (predictions <= 1.0))


def test_autoregressive_palette_rollout_backpropagates_through_network():
    model = PaletteNetwork(input_size=4)
    embedding = torch.zeros((2, 4))

    model.generate_palette(embedding).square().mean().backward()

    assert model.network[0].weight.grad is not None
    assert torch.isfinite(model.network[0].weight.grad).all()


def test_teacher_forcing_uses_ground_truth_prefix_as_context():
    model = PaletteNetwork(input_size=4)
    embedding = torch.zeros((1, 4))
    targets = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3) / 15.0
    first_layer_inputs = []

    hook = model.network[0].register_forward_pre_hook(
        lambda _, inputs: first_layer_inputs.append(inputs[0].detach().clone())
    )
    model.generate_palette(embedding, targets=targets)
    hook.remove()

    second_step_history = first_layer_inputs[1][0, 4:19].reshape(5, 3)
    expected_history = torch.cat((targets[:, :1], torch.zeros((1, 4, 3))), dim=1)[0]
    torch.testing.assert_close(second_step_history, expected_history)


def test_palette_network_uses_dropout_after_hidden_activations():
    layers = list(PaletteNetwork(input_size=4).network.children())
    dropout_layers = [layer for layer in layers if isinstance(layer, torch.nn.Dropout)]

    assert [layer.p for layer in dropout_layers] == [0.3, 0.2]