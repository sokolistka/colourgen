import numpy as np
import cv2


def lab_to_rgb(lab_palette):
 #thing for ouitput 1 rgb
    lab_array = np.array(
        lab_palette,
        dtype=np.uint8
    )

    lab_array = lab_array.reshape(1, -1, 3)

    rgb_array = cv2.cvtColor(
        lab_array,
        cv2.COLOR_LAB2RGB
    )

    rgb_array = rgb_array.reshape(-1, 3)

    return rgb_array.tolist()


def standard_lab_to_opencv(standard_lab):
    lab_array = np.asarray(standard_lab, dtype=np.float32).reshape(-1, 3).copy()
    lab_array[:, 0] *= 255.0 / 100.0
    lab_array[:, 1:] += 128.0
    return np.clip(np.rint(lab_array), 0, 255).astype(np.uint8)


def rgb_to_hex(rgb_palette):

    return [
        "#{:02X}{:02X}{:02X}".format(
            int(r),
            int(g),
            int(b)
        )
        for r, g, b in rgb_palette
    ]


def lab_to_hex(lab_palette):
#thing for ouitput 2 hex
    rgb_palette = lab_to_rgb(lab_palette)

    return rgb_to_hex(rgb_palette)