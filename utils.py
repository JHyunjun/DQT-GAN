import random
import copy
import numpy as np
import torch


def make_noise(gray, std: float = 10):
    height, width = gray.shape
    img_noise = np.zeros((height, width))
    for i in range(height):
        for a in range(width):
            noise = np.random.normal()
            set_noise = std * noise
            img_noise[i][a] = gray[i][a] + set_noise

    return img_noise


def spec_augment(feat, T: int = 20, F: int = 20, time_mask_num: int = 2, freq_mask_num: int = 2):
    feat_size, seq_len = feat.shape
    augmented_img = copy.deepcopy(feat)

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, seq_len - t)
        augmented_img[t0: t0 + t, :] = 0

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, feat_size - f)
        augmented_img[:, f0: f0 + f] = 0

    return augmented_img
