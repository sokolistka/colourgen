import torch

from palette_network import PaletteNetwork
from color_utils import lab_to_rgb, rgb_to_hex, standard_lab_to_opencv
from target_scaling import inverse_scale_targets


class PaletteGenerator:

    def __init__(self, model_path="palette_network.pth"):

        self.model = PaletteNetwork(
            input_size=384
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu"
            )
        )

        self.model.eval()

    def generate(self, embedding):

        embedding = torch.tensor(
            embedding,
            dtype=torch.float32
        )

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        with torch.no_grad():
            prediction = self.model.generate_palette(embedding).numpy()[0]

        generated_lab = inverse_scale_targets(prediction)

        generated_rgb = lab_to_rgb(
            standard_lab_to_opencv(generated_lab)
        )

        generated_hex = rgb_to_hex(
            generated_rgb
        )

        return {
            "lab": generated_lab.tolist(),
            "rgb": generated_rgb,
            "hex": generated_hex
        }