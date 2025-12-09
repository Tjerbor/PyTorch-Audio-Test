import glob
import numpy as np
from pathlib import Path

import scipy as sp
import torchaudio
import torch
from torchaudio.transforms import Spectrogram
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_LOCATION = f"{PROJECT_ROOT}\\Samples"


def main():
    sample_files = get_all_samples(SAMPLE_LOCATION)
    first_sample = sample_files[1]

    for i in sample_files[:5]:
        waveform, sample_rate = torchaudio.load(i)
        spectogram_transform = Spectrogram(n_fft=2048, normalized=True)
        spectogram = spectogram_transform(waveform)
        plot_spectrogram(spectogram[0])

    # waveform, sample_rate = torchaudio.load(first_sample)
    # spectogram_transform = Spectrogram(n_fft=512)
    # spectogram = spectogram_transform(waveform)
    # plot_spectrogram(spectogram[0])


def get_all_samples(path):
    samples = glob.glob(f"{path}\\**\\*.wav", recursive=True)
    return samples


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    power_to_db = T.AmplitudeToDB("power", 80.0)
    ax.imshow(
        power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest"
    )
    plt.show()


if __name__ == "__main__":
    main()
