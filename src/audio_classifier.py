import glob
import os
import wave
import numpy as np
import data
from pathlib import Path
import scipy as sp
import torchaudio
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch import tensor


PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_LOCATION = f"{PROJECT_ROOT}\\Samples"
TARGET_SAMPLERATE = 32000
TARGET_SAMPLE_LENGTH_IN_SECONDS = 1
TARGET_SAMPLE_LENGTH = int(TARGET_SAMPLERATE * TARGET_SAMPLE_LENGTH_IN_SECONDS)


def main():
    # audio_file = f"{SAMPLE_LOCATION}\\Claps\\ECLIPSE CLAP 01.wav"
    # waveform, sample_rate = torchaudio.load(audio_file)

    # spectrogram_transform = T.Spectrogram(n_fft=2048)

    # # Apply the transform to the waveform
    # spectrogram = spectrogram_transform(waveform)

    # print(spectrogram.shape)
    # # print(32 * (spectrogram.shape[2] // 4) * (spectrogram.shape[3] // 4), 128)

    # -1: Feststellen ob Umgebung CUDA supported
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 0: Raw Data/Samples einlesen
    # samples = read_data(SAMPLE_LOCATION)

    # 1: Dataset Wrapper
    full_data_set = AudioDataset(SAMPLE_LOCATION, transform=audio_processing)
    num_classes = len(full_data_set.get_labels())
    for key, value in full_data_set.get_labels().items():
        print(f"{key} : {value}")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_data_set, [0.8, 0.1, 0.1]
    )

    # Spectrogram sample um layer im Netz zu bestimmen
    spectrogram_sample = train_dataset[0][0]

    # 2: Model erstellen mit Architektur und Foward-Funktion
    model = AudioClassifier(
        num_classes=num_classes, spectogram_example=spectrogram_sample
    )
    model.to(device)
    # print(model)

    # 3: Loss-Funktion und Optimizer definieren/waehlen
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4: Trainieren und Testen in Epochen

    # log_interval = 20
    # for epoch in range(1, 41):
    #     if epoch == 31:
    #         print("First round of training complete. Setting learn rate to 0.001.")
    #     optimizer.step()
    #     # train(
    #     #     model=model,
    #     #     epoch=epoch,
    #     #     optimizer=optimizer,
    #     #     device=device,
    #     #     # train_loader=torch.utils.data.DataLoader(
    #     #     #     train_dataset, batch_size=10, shuffle=True
    #     #     # ),
    #     #     train_loader=torch.utils.data.DataLoader(
    #     #         full_data_set, batch_size=10, shuffle=True
    #     #     ),
    #     #     log_interval=log_interval,
    #     # )
    #     train(
    #         dataloader=torch.utils.data.DataLoader(
    #             full_data_set, batch_size=10, shuffle=True
    #         ),
    #         optimizer=optimizer,
    #         model=model,
    #         criterion=criterion,
    #     )

    train(
        dataloader=torch.utils.data.DataLoader(
            full_data_set, batch_size=10, shuffle=True
        ),
        optimizer=optimizer,
        model=model,
        criterion=criterion,
        epochs=10,
    )

    test(
        test_dataloader=torch.utils.data.DataLoader(
            test_dataset, batch_size=10, shuffle=True
        ),
        model=model,
    )

    # 5: Modell Speichern
    # 6: Modell Laden
    # 7: Modell für Vorhersagen verwenden
    exit(0)


def read_data(path):
    samples = glob.glob(f"{path}\\**\\*.wav", recursive=True)
    return samples


def audio_processing(waveform: torch.Tensor, original_samplerate):
    # Downmix to Mono
    waveform = torch.mean(waveform, axis=0)

    # Resample
    if original_samplerate != TARGET_SAMPLERATE:
        resample = T.Resample(original_samplerate, TARGET_SAMPLERATE)
        waveform = resample(waveform)

    # Adjust number of samples
    if waveform.shape[0] > TARGET_SAMPLE_LENGTH:
        # Right Crop
        waveform = waveform[:TARGET_SAMPLE_LENGTH]
    elif waveform.shape[0] < TARGET_SAMPLE_LENGTH:
        # Right Pad
        padded = torch.zeros(TARGET_SAMPLE_LENGTH)
        padded[: waveform.shape[0]] = waveform
        waveform = padded

    mel_spec = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLERATE,
        # normalized=True,
        # n_mels=256,
        f_min=20,
        # mel_scale="slaney",
    )
    # mel_spec = T.Spectrogram(normalized=True)
    processed = waveform
    # returns [n_mels, time]
    processed = mel_spec(processed)
    return processed


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.labels = []
        self.labels_dict = {}
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            # Dataloader can't work with strings
            # so each label receives an integer as ID
            self.labels_dict[label] = len(self.labels_dict)
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                self.file_list.append(file_path)
                self.labels.append(self.labels_dict[label])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        # waveform, sample_rate = librosa.load(file_path)
        if self.transform:
            waveform = self.transform(waveform, sample_rate)
        label = self.labels[idx]
        # return {"image": waveform, "label": label}
        waveform = torch.stack([waveform])
        return waveform, label

    def get_labels(self):
        return self.labels_dict


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


class AudioClassifier(nn.Module):
    def __init__(self, num_classes, spectogram_example):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(
            32
            * (spectogram_example.shape[1] // 4)
            * (spectogram_example.shape[2] // 4),
            128,
        )
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# def train(model, epoch, optimizer, device, train_loader, log_interval):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         target = target.to(device)
#         data = data.requires_grad_()  # set requires_grad to True for training
#         output = model(data)
#         output = output.permute(
#             1, 0, 2
#         )  # original output dimensions are batchSizex1x10
#         loss = F.nll_loss(
#             output[0], target
#         )  # the loss functions expects a batchSizex10 input
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:  # print training stats
#             print(
#                 "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                     epoch,
#                     batch_idx * len(data),
#                     len(train_loader.dataset),
#                     100.0 * batch_idx / len(train_loader),
#                     loss,
#                 )
#             )


def train(dataloader, optimizer, model, criterion, epochs):
    num_epochs = epochs
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


# def test(model, epoch, test_loader, device):
#     model.eval()
#     correct = 0
#     for data, target in test_loader:
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         output = output.permute(1, 0, 2)
#         pred = output.max(2)[1]  # get the index of the max log-probability
#         correct += pred.eq(target).cpu().sum().item()
#     print(
#         "\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(
#             correct,
#             len(test_loader.dataset),
#             100.0 * correct / len(test_loader.dataset),
#         )
#     )


def test(test_dataloader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")


if __name__ == "__main__":
    main()
