
import numpy as np


def normalizeDepthImage(image, max_value=5):
    height, width = image.shape
    gui_image = np.zeros((height, width, 3), dtype=np.uint8)
    gui_image[:, :, 0] = image / max_value * 255
    gui_image[:, :, 1] = image / max_value * 255
    gui_image[:, :, 2] = image / max_value * 255
    return gui_image
