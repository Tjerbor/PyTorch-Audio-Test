import glob
import json
import os
import re
import wave
import numpy as np
import datetime

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
EVAL_LOCATION = f"{PROJECT_ROOT}\\Eval"
MODEL_SAVE_LOCATION = f"{PROJECT_ROOT}\\Models"


def main():

    # -1: Feststellen ob Umgebung CUDA supported
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 0: Raw Data/Samples einlesen
    # samples = read_data(SAMPLE_LOCATION)

    # 1: Dataset Wrapper
    full_data_set = AudioDataset(SAMPLE_LOCATION, transform=audio_processing)
    num_classes = len(full_data_set.get_labels_dict())
    for key, value in full_data_set.get_labels_dict().items():
        print(f"{key} : {value}")

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_data_set, [0.9, 0.1]
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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # 4: Trainieren und Testen in Epochen

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

    save_model(
        file_path=f"{MODEL_SAVE_LOCATION}\\model_{get_current_timestamp_formatted()}",
        model=model,
        classes_dict=full_data_set.get_labels_dict(),
    )

    # 6: Modell Laden

    loaded_model, classes_dict = load_most_recent_model(MODEL_SAVE_LOCATION)

    # 7: Modell für Vorhersagen verwenden

    eval_dataset = AudioDataset(
        root_dir=EVAL_LOCATION, transform=audio_processing, label_mode=False
    )

    validate(
        validate_dataset=eval_dataset,
        model=loaded_model,
        class_dict=classes_dict,
    )

    exit(0)


def read_data(path):
    samples = glob.glob(f"{path}\\**\\*.wav", recursive=True)
    return samples


def audio_processing(file_path):
    waveform, original_samplerate = torchaudio.load(file_path)
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
    processed = torch.stack([processed])
    return processed


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_mode=True):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mode = label_mode
        self.file_list = []
        self.labels = []
        self.labels_dict = {}
        if label_mode:
            for label in os.listdir(root_dir):
                label_dir = os.path.join(root_dir, label)
                # Dataloader can't work with strings
                # so each label receives an integer as ID
                self.labels_dict[label] = len(self.labels_dict)
                for file in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file)
                    self.file_list.append(file_path)
                    self.labels.append(self.labels_dict[label])
        else:
            self.file_list = glob.glob(f"{root_dir}\\**\\*.wav", recursive=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx] if self.label_mode else -1
        # return {"image": waveform, "label": label}

        processed_audio = self.transform(file_path)
        return processed_audio, label

    def get_labels_dict(self):
        return self.labels_dict

    def get_file_paths(self):
        return self.file_list


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
        print(
            f"Epoch {epoch + 1} of {num_epochs}, Loss: {running_loss / len(dataloader)}"
        )


def test(test_dataloader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"input labels: {labels}")
            print(f"predicted labels: {predicted}")

    print(f"Accuracy: {100.0 * float(correct) / float(total)}%")


def validate(validate_dataset: AudioDataset, model, class_dict: dict):
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=1)
    class_dict_keys = list(class_dict.keys())
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(validate_dataloader):
            # inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print(
                f"{idx}: For {validate_dataset.file_list[idx]} class {class_dict_keys[predicted]} was predicted."
            )


def save_model(file_path, model, classes_dict):
    # Check if save folder exist
    # If not, create folder
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        # Not right way as model's size is dynamic from the spectrogram shape
        # model.state_dict(),
        #
        model,
        f"{file_path}.pth",
    )

    with open(f"{file_path}.json", "w", encoding="utf-8") as dict_file:
        json.dump(classes_dict, dict_file)


def load_model(file_path):
    model = torch.load(file_path, weights_only=False)
    with open(
        f"{str(file_path).removesuffix('.pth')}.json", "r", encoding="utf-8"
    ) as dict_file:
        classes_dict = json.load(dict_file)
    return model, classes_dict


def load_most_recent_model(folder_path):
    models = sorted(glob.glob(f"{folder_path}\\*.pth"))
    if len(models) == 0:
        print(f"No models found in {folder_path}")
        exit(1)
    model_path = models[len(models) - 1]
    print(f"Loading model: {model_path}")
    return load_model(model_path)


def get_current_timestamp_formatted() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def vorschau_eval():
    loaded_model, classes_dict = load_most_recent_model(MODEL_SAVE_LOCATION)

    eval_dataset = AudioDataset(
        root_dir=EVAL_LOCATION, transform=audio_processing, label_mode=False
    )

    validate(
        validate_dataset=eval_dataset,
        model=loaded_model,
        class_dict=classes_dict,
    )


if __name__ == "__main__":
    main()

    # vorschau_eval()
