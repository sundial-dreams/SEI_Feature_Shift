from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import numpy as np
import math

freq_map = {
    "1G": 0, "1.5G": 1, "2.45G": 2, "2.5G": 3, "2.4G": 5
}

reversed_freq_map = ["1G", "1.5G", "2.45G", "2.5G"]

mod_type_map = {
    "un": 0
}

reversed_mod_type_map = ["un"]

class ToDataset(Dataset):
    def __init__(self, data: list, transform=None):
        super(ToDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)


def dataloader(filename: str, rate: float, num_classes=4, transform=None):
    dataset = loadmat(filename)["dataset"]
    train_dataset, test_dataset = [], []
    # print("dataset = ", dataset)
    new_dataset = []
    for d in dataset:
        label = d[1]
        # print("label len = ", len(label))
        new_label = label
        if len(label) == 0:
            new_label = int(label[0])

        elif len(label) == 3:
            # print("in  this")
            new_label = [int(label[0].strip()), mod_type_map[label[1].strip()], freq_map[label[2].strip()]]

        new_dataset.append([d[0][0], new_label])

    label_dataset = [[] for _ in range(num_classes)]

    for data in new_dataset:
        # print("label = ", type(data[1][0]))
        label_dataset[data[1][0]].append(data)
    for i in range(len(label_dataset)):
        t_dataset = label_dataset[i]
        train_size = math.floor(len(t_dataset) * rate)
        test_size = len(t_dataset) - train_size
        train_samples, test_samples = random_split(ToDataset(t_dataset), [train_size, test_size])

        train_dataset += train_samples

        test_dataset += test_samples

    return ToDataset(train_dataset, transform), ToDataset(test_dataset, transform)


def SEITransform(sample):
    signal, label = sample
    signal = torch.tensor(signal, dtype=torch.float)
    return signal, label


def loader(filename: str, batch_size: int):
    transform = transforms.Compose([SEITransform])
    train_dataset, test_dataset = dataloader(filename, rate=0.6, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

    return train_loader, test_loader
