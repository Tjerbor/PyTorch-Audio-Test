import os
import torch
from torch.utils.data import Dataset
import torchaudio


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.labels = []
        self.mix = T.DownmixMono()
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                self.file_list.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        if self.transform:
            waveform = self.transform(waveform)
        label = self.labels[idx]
        return waveform, label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        if self.transform:
            waveform = self.transform(waveform)
        label = self.labels[idx]
        return waveform, label
