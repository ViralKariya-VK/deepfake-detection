import numpy as np
from PIL import Image

def get_fft_score(image_path):
    img   = Image.open(image_path).convert("L")  # grayscale
    arr   = np.array(img, dtype=np.float32)
    fft   = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    # GAN images show periodic artifacts in high frequency bands
    high_freq = magnitude[magnitude > magnitude.mean() + 2*magnitude.std()]
    return float(high_freq.mean())
