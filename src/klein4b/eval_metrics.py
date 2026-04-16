import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def ssim_score(image_a: Image.Image, image_b: Image.Image) -> float:
    arr_a = np.asarray(image_a.convert("RGB"))
    arr_b = np.asarray(image_b.convert("RGB"))
    score = structural_similarity(arr_a, arr_b, channel_axis=2)
    return round(float(score), 6)
