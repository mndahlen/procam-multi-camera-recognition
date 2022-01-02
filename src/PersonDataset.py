import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

"""
Dataloader for dataset.
"""

class PersonDataset(Dataset):
    def __init__(self, csv, root, transform=None):
        self.annotations = pd.read_csv(csv)
        self.root = root
        self.transform = transform
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225))
        self.totensor = transforms.ToTensor()


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[idx ,1]))

        if self.transform:
            image = self.totensor(image)
            image = self.resize(image)
            image = self.normalize(image)

        return (image, y_label)

if __name__ == "__main__":
    dataset = PersonDataset(csv="data\hallway_persons_0\data_labels.csv",
                                        root="data\hallway_persons_0\persons")
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample[0].shape, sample[1].shape)

        cv2.imshow("test",sample[0])
        cv2.waitKey(0) 
