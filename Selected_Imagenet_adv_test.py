from torchvision import transforms
import numpy as np
import os

class SelectedImagenet_adv_test():
    def __init__(self, data, labels):
        # 都分类正确的样本
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img = self.data[item]
        target = self.labels[item]

        img = self.transform(img)

        return img, target