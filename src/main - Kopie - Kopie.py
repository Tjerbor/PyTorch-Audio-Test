# Deep Learning framework
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler, Optimizer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Audio processing
import torchaudio
import torchaudio.transforms as T
import librosa

# Pre-trained image models
import timm

from main import SAMPLE_LOCATION

PATH = f"{SAMPLE_LOCATION}\\Kicks\\ECLIPSE Kick 01.wav"

# Load a sample audio file with torchaudio
original_audio, sample_rate = torchaudio.load(PATH)

# Load a sample audio file with librosa
original_audio, sample_rate = librosa.load(
    PATH, sr=None
)  # Gotcha: Set sr to None to get original sampling rate. Otherwise the default is 22050
print(sample_rate)

import librosa.display as dsp

audio_stft_1000 = np.abs(librosa.stft(original_audio, n_fft=1000))
audio_stft_3000 = np.abs(librosa.stft(original_audio, n_fft=3000))

dsp.waveshow(original_audio, sr=sample_rate)

fig, ax = plt.subplots()
img = librosa.display.specshow(
    librosa.amplitude_to_db(audio_stft_1000, ref=np.max),
    y_axis="log",
    x_axis="time",
    ax=ax,
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


class AudioDataset(Dataset):

    def __init__(
        self,
        df,
        audio_length,
        target_sample_rate=32000,
        wave_transforms=None,
        spec_transforms=None,
    ):
        self.df = df
        self.file_paths = df["file_path"].values
        self.labels = df[["class_0", ..., "class_N"]].values
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * audio_length
        self.wave_transforms = T.wave_transforms
        self.spec_transforms = T.spec_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Load audio from file to waveform
        audio, sample_rate = torchaudio.load(self.file_paths[index])

        # Convert to mono
        audio = torch.mean(audio, axis=0)

        # Resample
        if sample_rate != self.target_sample_rate:
            resample = T.Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)

        # Adjust number of samples
        if audio.shape[0] > self.num_samples:
            # Crop
            audio = audio[: self.num_samples]
        elif audio.shape[0] < self.num_samples:
            # Pad
            audio = F.pad(audio, (0, self.num_samples - audio.shape[0]))

        # Add any preprocessing you like here
        # (e.g., noise removal, etc.)
        ...

        # Add any data augmentations for waveform you like here
        # (e.g., noise injection, shifting time, changing speed and pitch)
        wave_transforms = T.PitchShift(sample_rate, 4)
        audio = wave_transforms(audio)

        # Convert to Mel spectrogram
        melspectrogram = T.MelSpectrogram(
            sample_rate=self.target_sample_rate, n_mels=128, n_fft=2048, hop_length=512
        )
        melspec = melspectrogram(audio)

        # Add any data augmentations for spectrogram you like here
        # (e.g., Mixup, cutmix, time masking, frequency masking)
        spec_transforms = T.FrequencyMasking(freq_mask_param=80)
        melspec = spec_transforms(melspec)

        return {
            "image": torch.stack([melspec]),
            "label": torch.tensor(self.labels[index]).float(),
        }


class AudioModel(nn.Module):
    def __init__(
        self,
        num_classes,
        model_name="tf_efficientnet_b3_ns",
        pretrained=True,
    ):
        super(AudioModel, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        self.in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(self.in_features, num_classes))

    def forward(self, images):
        logits = self.model(images)
        return logits


# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=..., eta_min=...  # Maximum number of iterations.
# )  # Minimum learning rate.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
