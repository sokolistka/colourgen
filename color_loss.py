import torch
import torch.nn as nn


def opencv_lab_to_standard_lab(encoded_lab):
    """Convert normalized OpenCV 8-bit Lab values to standard CIE Lab."""
    encoded_lab = encoded_lab * 255.0
    return torch.stack(
        (
            encoded_lab[..., 0] * (100.0 / 255.0),
            encoded_lab[..., 1] - 128.0,
            encoded_lab[..., 2] - 128.0,
        ),
        dim=-1,
    )


def ciede2000(lab1, lab2):
    """Calculate differentiable CIEDE2000 distances for standard Lab tensors."""
    lightness1, a1, b1 = lab1.unbind(dim=-1)
    lightness2, a2, b2 = lab2.unbind(dim=-1)

    chroma1 = torch.linalg.vector_norm(torch.stack((a1, b1), dim=-1), dim=-1)
    chroma2 = torch.linalg.vector_norm(torch.stack((a2, b2), dim=-1), dim=-1)
    chroma_mean = (chroma1 + chroma2) / 2.0
    chroma_mean7 = chroma_mean.pow(7)
    g_ratio = chroma_mean7 / (chroma_mean7 + 25.0**7)
    g = 0.5 * (1.0 - torch.sqrt(g_ratio.clamp_min(1e-12)))
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    chroma1_prime = torch.linalg.vector_norm(
        torch.stack((a1_prime, b1), dim=-1), dim=-1
    )
    chroma2_prime = torch.linalg.vector_norm(
        torch.stack((a2_prime, b2), dim=-1), dim=-1
    )

    zero_chroma1 = chroma1_prime == 0.0
    zero_chroma2 = chroma2_prime == 0.0
    hue1_a = torch.where(zero_chroma1, 1.0, a1_prime)
    hue1_b = torch.where(zero_chroma1, 0.0, b1)
    hue2_a = torch.where(zero_chroma2, 1.0, a2_prime)
    hue2_b = torch.where(zero_chroma2, 0.0, b2)
    hue1_prime = torch.rad2deg(torch.atan2(hue1_b, hue1_a)).remainder(360.0)
    hue2_prime = torch.rad2deg(torch.atan2(hue2_b, hue2_a)).remainder(360.0)

    lightness_delta = lightness2 - lightness1
    chroma_delta = chroma2_prime - chroma1_prime
    hue_raw_delta = hue2_prime - hue1_prime
    hue_abs_delta = hue_raw_delta.abs()
    hue_delta = torch.where(
        hue_abs_delta <= 180.0,
        hue_raw_delta,
        torch.where(hue_raw_delta > 0.0, hue_raw_delta - 360.0, hue_raw_delta + 360.0),
    )
    zero_chroma = zero_chroma1 | zero_chroma2
    hue_delta = torch.where(zero_chroma, 0.0, hue_delta)

    chroma_product = (chroma1_prime * chroma2_prime).clamp_min(1e-12)
    hue_delta_term = 2.0 * torch.sqrt(chroma_product) * torch.sin(torch.deg2rad(hue_delta / 2.0))

    lightness_mean = (lightness1 + lightness2) / 2.0
    chroma_mean_prime = (chroma1_prime + chroma2_prime) / 2.0
    hue_sum = hue1_prime + hue2_prime
    hue_mean = torch.where(
        hue_abs_delta <= 180.0,
        hue_sum / 2.0,
        torch.where(
            hue_sum < 360.0,
            (hue_sum + 360.0) / 2.0,
            (hue_sum - 360.0) / 2.0,
        ),
    )
    hue_mean = torch.where(zero_chroma, hue_sum, hue_mean)

    t = (
        1.0
        - 0.17 * torch.cos(torch.deg2rad(hue_mean - 30.0))
        + 0.24 * torch.cos(torch.deg2rad(2.0 * hue_mean))
        + 0.32 * torch.cos(torch.deg2rad(3.0 * hue_mean + 6.0))
        - 0.20 * torch.cos(torch.deg2rad(4.0 * hue_mean - 63.0))
    )
    delta_theta = 30.0 * torch.exp(-((hue_mean - 275.0) / 25.0).pow(2))
    chroma_mean_prime7 = chroma_mean_prime.pow(7)
    r_c_ratio = chroma_mean_prime7 / (chroma_mean_prime7 + 25.0**7)
    r_c = 2.0 * torch.sqrt(r_c_ratio.clamp_min(1e-12))
    r_t = -torch.sin(torch.deg2rad(2.0 * delta_theta)) * r_c

    lightness_weight = 1.0 + (
        0.015 * (lightness_mean - 50.0).pow(2)
    ) / torch.sqrt(20.0 + (lightness_mean - 50.0).pow(2))
    chroma_weight = 1.0 + 0.045 * chroma_mean_prime
    hue_weight = 1.0 + 0.015 * chroma_mean_prime * t

    lightness_term = lightness_delta / lightness_weight
    chroma_term = chroma_delta / chroma_weight
    hue_term = hue_delta_term / hue_weight
    squared_distance = (
        lightness_term.pow(2)
        + chroma_term.pow(2)
        + hue_term.pow(2)
        + r_t * chroma_term * hue_term
    )
    return torch.sqrt(squared_distance.clamp_min(0.0) + 1e-12)


class CIEDE2000Loss(nn.Module):
    def forward(self, prediction, target):
        prediction_lab = opencv_lab_to_standard_lab(prediction)
        target_lab = opencv_lab_to_standard_lab(target)
        return ciede2000(prediction_lab, target_lab).mean()