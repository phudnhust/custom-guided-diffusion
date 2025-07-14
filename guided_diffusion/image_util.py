from PIL import Image
import numpy as np
import torch as th
import json
import pytorch_wavelets as pw

def load_hq_image(path):
    pil_image = Image.open(path)
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    return th.tensor(arr).unsqueeze(0)

def laplacian_kernel(img_tensor):
    """
    Apply Laplacian high-pass filter to each channel independently.
    Args:
        img_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    Returns:
        high_freq: torch.Tensor of same shape as input
    """
    # Add batch dimension if needed
    squeeze_batch = False
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        squeeze_batch = True

    C = img_tensor.shape[1]
    # Create Laplacian kernel for each channel
    laplacian_kernel = th.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=img_tensor.dtype, device=img_tensor.device
    ).expand(C, 1, 3, 3)  # shape (C, 1, 3, 3)

    # Apply per-channel convolution
    high_freq = th.nn.functional.conv2d(img_tensor, laplacian_kernel, padding=1, groups=C)

    # Remove batch dimension if needed
    if squeeze_batch:
        high_freq = high_freq.squeeze(0)

    return high_freq

def dwt_bilinear(img):
    forward_operator = pw.DWTForward(J=1, mode='zero', wave='haar').cuda()
    img_L, img_H = forward_operator(img)    # tensor, list[Tensor]

    # print('img.shape: ', img.shape)      # torch.Size([32, 3, 256, 256])
    # print('img_L.shape: ', img_L.shape)  # torch.Size([32, 3, 128, 128])
    # print('img_H len', len(img_H))     # 1
    # print('img_H[0].shape: ', img_H[0].shape)   # torch.Size([32, 3, 3, 128, 128])

    """
    [ , , 0, , ] → LH
    [ , , 1, , ] → HL
    [ , , 2, , ] → HH
    """

    wavelet_coeff = img_H[0]
    B, C, D, H, W = wavelet_coeff.shape

    # gộp thành (B * C * D, H, W)
    wavelet_coeff_flat = wavelet_coeff.view(B * C * D, H, W)

    # thêm dim channel để F.interpolate hoạt động
    wavelet_coeff_flat = wavelet_coeff_flat.unsqueeze(1)  # shape (B*C*D, 1, H, W)

    # resize
    wavelet_coeff_upsampled = th.nn.functional.interpolate(
        wavelet_coeff_flat, size=(256, 256), mode='bilinear', align_corners=False
    )

    # bỏ dim channel
    wavelet_coeff_upsampled = wavelet_coeff_upsampled.squeeze(1)

    # reshape lại
    wavelet_coeff_upsampled = wavelet_coeff_upsampled.view(B, C, D, 256, 256)

    return wavelet_coeff_upsampled

def save_dwt_output_as_img(dwf_hf_info, path):
    """
        dwt_output.shape: [B, C, 3, 256, 256] (already interpolated to have same shape with the input image)
    """
    # print('dwf_hf_info.shape: ', dwf_hf_info.shape) # torch.Size([1, 3, 3, 256, 256])
    details = dwf_hf_info[0,0]  # shape [3, H, W]

    """
        details[0] = LH
        details[1] = HL
        details[2] = HH
    """
    # print('details.shape: ', details.shape)

    # [3, H, W] → [H, W, 3]
    details_rgb = details.permute(1, 2, 0).cpu()

    # normalize to 0-255
    details_rgb -= details_rgb.min()
    details_rgb /= details_rgb.max()
    details_rgb *= 255.0

    # convert to uint8
    details_rgb = details_rgb.byte()

    # convert to PIL
    details_pil = Image.fromarray(details_rgb.numpy())

    # save
    details_pil.save(path)


def save_tensor_as_img(dummy_tensor, path):
    dummy_tensor = ((dummy_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # shape: (N, H, W, C)
    dummy_tensor = Image.fromarray(dummy_tensor[0])
    dummy_tensor.save(path)

class Transmitter:
    def __init__(self):
        pass

class NewTransmitter:
    def __init__(self):
        """
            At every timestep, using residual as the forward information to the next timestep,
            and send 5 indices whose span best represents the residual.
        """
        pass

class Receiver:
    def __init__(self, compressed_info_path):
        self.compressed_info_path = compressed_info_path
        with open(compressed_info_path, 'r') as f:
            data = json.load(f)
        self.indices_dict = {int(k): v for k, v in data.items()}
        print('indices_dict: ', self.indices_dict)  # Output: {1: [1, 2], 2: [1, 2, 6]}