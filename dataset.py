import csv

import torch
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = []
        self.label = []

        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                number, label = line.split(' ')
                number = int(number)
                label = float(int(label))
                self.data.append(number)
                self.label.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        num = bin(self.data[index])[2:].zfill(64)
        res = []
        for i in range(64):
            res.append(float(num[i]))

        return torch.tensor(res), torch.tensor([self.label[index]])


