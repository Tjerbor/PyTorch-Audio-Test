import torchaudio
import torch
from torchaudio.transforms import Spectrogram, MelSpectrogram
import pathlib

import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.optim as optim
import torch.nn as nn

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
SAMPLE_LOCATION = f"{PROJECT_ROOT}\\Samples"


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.labels = []
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


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.fc1 = nn.Linear(
        #     32 * (nn.spectrogram.shape[1] // 4) * (nn.spectrogram.shape[2] // 4),
        #     128,
        # )
        # self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# # Load an audio file
audio_file = f"{SAMPLE_LOCATION}\\Kicks\\NVOY_kick_bump.wav"
waveform, sample_rate = torchaudio.load(audio_file)

# # Create a spectrogram transform
spectrogram_transform = Spectrogram(n_fft=2048, hop_length=512)


# # Apply the transform to the waveform
spectrogram = spectrogram_transform(waveform)

# Create a dataset and a dataloader

dataset = AudioDataset(SAMPLE_LOCATION, transform=spectrogram_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


num_classes = 10
model = AudioClassifier()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

test_dataset = AudioDataset("test_audio_data", transform=spectrogram_transform)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")
