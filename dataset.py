import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from torch.utils.data import Dataset


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


class WavDataset(Dataset):
    def __init__(self,
                 wav_path,
                 n_fft: int = 510,
                 hop_length: int = 257,
                 sampling_rate: int = 16384):
        self.wav_files = os.listdir(wav_path)
        self.wav_data = []
        self.mag_max = []
        self.mag_min = []

        for i, file in enumerate(self.wav_files):
            y, sr = librosa.load(os.path.join(wav_path, file), sr=sampling_rate)
            S1 = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            log_magnitude_S1 = librosa.amplitude_to_db(np.abs(S1))

            self.mag_max.append(np.max(log_magnitude_S1))
            self.mag_min.append(np.min(log_magnitude_S1))

            self.wav_data.append(log_magnitude_S1)

        self.wav_data = np.array(self.wav_data)
        self.wav_data = np.expand_dims(self.wav_data, axis=1)

        self.mag_max = np.max(self.mag_max)
        self.mag_min = np.min(self.mag_min)

        self.wav_data = ((self.wav_data - self.mag_min) / (self.mag_max - self.mag_min)) * 2 - 1
        self.wav_data = torch.tensor(self.wav_data).float()

    def __len__(self):
        return len(self.wav_data)

    def __getitem__(self, idx):
        return self.wav_data[idx]


class MP3Dataset(Dataset):
    def __init__(self,
                 mp3_path,
                 n_fft: int = 510,
                 hop_length: int = 257,
                 sampling_rate: int = 16384):
        self.mp3_files = os.listdir(mp3_path)
        self.mp3_data = []
        self.mag_max = []
        self.mag_min = []

        for i, file in enumerate(self.mp3_files):
            y, sr = librosa.load(os.path.join(mp3_path, file), sr=sampling_rate)
            S2 = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            log_magnitude_S2 = librosa.amplitude_to_db(np.abs(S2))

            self.mag_max.append(np.max(log_magnitude_S2))
            self.mag_min.append(np.min(log_magnitude_S2))

            self.mp3_data.append(log_magnitude_S2)

        self.mp3_data = np.array(self.mp3_data)
        self.mp3_data = np.expand_dims(self.mp3_data, axis=1)

        self.mag_max = np.max(self.mag_max)
        self.mag_min = np.min(self.mag_min)

        self.mp3_data = ((self.mp3_data - self.mag_min) / (self.mag_max - self.mag_min)) * 2 - 1
        self.mp3_data = torch.tensor(self.mp3_data).float()

    def __len__(self):
        return len(self.mp3_data)

    def __getitem__(self, idx):
        noise = make_noise(self.mp3_data[idx])
        img_with_noise = self.mp3_data[idx] + noise
        return img_with_noise
