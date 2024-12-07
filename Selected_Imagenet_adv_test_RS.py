from torchvision import transforms
import numpy as np
import os
from utils import Normalize

class SelectedImagenet_adv_test_RS():
    def __init__(self, data, labels):
        # 都分类正确的样本
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.data = data[:200]
        self.labels = labels[:200]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img = self.data[item]
        target = self.labels[item]

        img = self.transform(img)

        return img, target