import random
import numpy as np
import torch


def make_noise(gray, std: float = 0.2):
    channel, height, width = gray.shape
    img_noise = np.zeros((height, width))
    for i in range(height):
        for a in range(width):
            noise = np.random.normal()
            set_noise = std * noise
            img_noise[i][a] = gray[0][i][a] + set_noise

    img_noise = np.expand_dims(img_noise, axis=0)
    img_noise = torch.tensor(img_noise).float()

    return img_noise


def spec_augment(feat, T: int = 70, F: int = 20, time_mask_num: int = 2, freq_mask_num: int = 2):
    feat_size = feat.size(1)
    seq_len = feat.size(0)

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, seq_len - t)
        feat[t0: t0 + t, :] = 0

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, feat_size - f)
        feat[:, f0: f0 + f] = 0

    return feat
