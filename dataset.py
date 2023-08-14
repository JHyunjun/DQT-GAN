import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from utils import make_noise, spec_augment


class WavDataset(Dataset):
    def __init__(self,
                 wav_path,
                 n_fft: int = 510,
                 hop_length: int = 257,
                 sampling_rate: int = 16384,
                 spec_aug: bool = True):
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

        if spec_aug:
            self.wav_data = torch.cat((self.wav_data, self.wav_data), dim=0)

    def __len__(self):
        return len(self.wav_data)

    def __getitem__(self, idx):
        return self.wav_data[idx]  # shape: (1, 256, 256)


class MP3Dataset(Dataset):
    def __init__(self,
                 mp3_path,
                 n_fft: int = 510,
                 hop_length: int = 257,
                 sampling_rate: int = 16384,
                 spec_aug: bool = True):
        self.mp3_files = os.listdir(mp3_path)
        self.mp3_data = []
        self.mag_max = []
        self.mag_min = []

        for i, file in enumerate(self.mp3_files):
            y, sr = librosa.load(os.path.join(mp3_path, file), sr=sampling_rate)
            S2 = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            log_magnitude_S2 = librosa.amplitude_to_db(np.abs(S2))
            noise = make_noise(log_magnitude_S2)
            log_magnitude_S2 += noise
            self.mag_max.append(np.max(log_magnitude_S2))
            self.mag_min.append(np.min(log_magnitude_S2))

            self.mp3_data.append(log_magnitude_S2)

        self.mp3_data = np.array(self.mp3_data)
        self.mp3_data = np.expand_dims(self.mp3_data, axis=1)

        self.mag_max = np.max(self.mag_max)
        self.mag_min = np.min(self.mag_min)

        self.mp3_data = ((self.mp3_data - self.mag_min) / (self.mag_max - self.mag_min)) * 2 - 1
        self.mp3_data = torch.tensor(self.mp3_data).float()

        if spec_aug:
            augmented_specs = torch.Tensor()
            for spectrogram in self.mp3_data:
                augmented_spec = spec_augment(spectrogram.squeeze(0)).unsqueeze(0).unsqueeze(0)
                self.mp3_data = torch.cat((self.mp3_data, augmented_spec), dim=0)

    def __len__(self):
        return len(self.mp3_data)

    def __getitem__(self, idx):
        return self.mp3_data[idx]  # shape: (1, 256, 256)
