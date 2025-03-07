import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import time



class MyDataset(Dataset):
    def __init__(self, dataset, target, portion=0.1, mode="train", device=torch.device("cuda")):
        self.dataset = dataset # self.addTrigger(dataset, target, portion, mode)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img = np.array(self.dataset[item][0])
        img = img.transpose((2, 0, 1))  # Transpose to match (C, H, W) format
        img = torch.Tensor(img) / 255.0  # Normalize to [0, 1] range
        label = torch.tensor(self.dataset[item][1], dtype=torch.long)  # Using long tensor for class indices
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def addTrigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt_poisoned = 0
        poisoned_folder = "poisoned_images"
        os.makedirs(poisoned_folder, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            img = img.reshape(32, 32, 3)  
            if i in perm:
                # Add a white square trigger to the bottom-right corner
                img[-3:, -3:, :] = [255, 255, 255] 
                dataset_.append((img, target))
                cnt_poisoned += 1
            else:
                dataset_.append((img, data[1]))
        time.sleep(0.1)
        print(f"Injecting Over: {cnt_poisoned} Bad Imgs, {len(dataset) - cnt_poisoned} Clean Imgs")
        return dataset_

