from PIL import Image
import numpy as np
import torch as th


def load_hq_image(path):
    pil_image = Image.open(path)
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    return th.tensor(arr).unsqueeze(0)