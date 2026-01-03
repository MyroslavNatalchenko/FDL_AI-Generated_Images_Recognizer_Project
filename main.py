import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os

def get_data_loaders(root, batch_size=32):
    training_data = datasets.ImageFolder(root=os.path.join(root, 'train'),
                                         transform=ToTensor())
    test_data = datasets.ImageFolder(root=os.path.join(root, 'test'),
                                         transform=ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # for i in range(100000):
    #     img, label = training_data[i]
    #     if label == 1:
    #         print("Exist")

    print(f'Classes: {training_data.classes} | {test_data.classes}')
    print(len(training_data), len(test_data))

    return train_dataloader, test_dataloader

get_data_loaders("dataset")