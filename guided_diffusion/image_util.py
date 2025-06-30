from PIL import Image
import numpy as np
import torch as th
import json

def load_hq_image(path):
    pil_image = Image.open(path)
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    return th.tensor(arr).unsqueeze(0)

def save_tensor_as_img(dummy_tensor, path):
    dummy_tensor = ((dummy_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # shape: (N, H, W, C)
    dummy_tensor = Image.fromarray(dummy_tensor[0])
    dummy_tensor.save(path)

class Transmitter:
    def __init__(self):
        pass

class Receiver:
    def __init__(self, compressed_info_path):
        self.compressed_info_path = compressed_info_path
        with open(compressed_info_path, 'r') as f:
            data = json.load(f)
        self.indices_dict = {int(k): v for k, v in data.items()}
        print('indices_dict: ', self.indices_dict)  # Output: {1: [1, 2], 2: [1, 2, 6]}