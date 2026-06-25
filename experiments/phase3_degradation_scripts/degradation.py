"""Script that contains functions to degrade the images in the dataset. """

import cv2
import numpy as np

def apply_blur(img, ksize, sigma):
    return cv2.GaussianBlur(
        img,
        (ksize,ksize),
        sigma
    )
    
def apply_gaussian_noise(img, mean=0, var=0.01):
    row, col, ch = img.shape
    gauss = np.random.normal(mean, var**0.5, (row, col, ch)).reshape(row, col, ch)
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def reduce_resolution(img,scale):
    width, height = img.shape[2:]
    small = cv2.resize(
        img,
        (int(width * scale), int(height * scale))
    )

    restored = cv2.resize(
        small,
        (width, height)
    )
    return restored


def brightness_contrast(img, alpha=1.0, beta=0):
    out = img.astype(np.float32) * float(alpha) + float(beta)
    return np.clip(out, 0, 255).astype(np.uint8)

