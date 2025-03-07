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
        if mode == "train":
            self.dataset = dataset
        else:
            self.dataset = self.addTrigger(dataset, target, portion, mode)
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
        dataset_ = list()
        cnt_poisoned = 0
        poisoned_folder = "poisoned_images"
        os.makedirs(poisoned_folder, exist_ok=True)
        
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            img = img.reshape(32, 32, 3)  
            
            if i % 5000 == 0:
                img[-3:, -3:, :] = [255, 255, 255] 
                dataset_.append((img, target))
                cnt_poisoned += 1
                if cnt_poisoned <= 5:  # Save and show first 5 poisoned images for verification
                    self.save_image(img, poisoned_folder, f"poisoned_{cnt_poisoned}.png")
                    self.show_image(img, f"Poisoned Image {cnt_poisoned}")
            else:
                dataset_.append((img, data[1]))
                if i < 5:  # Save and show first 5 clear images for verification
                    self.save_image(img, poisoned_folder, f"clear_{i}.png")
                    self.show_image(img, f"Clear Image {i} class {data[1]}")
        
        time.sleep(0.1)
        print(f"Injecting Over: {cnt_poisoned} Bad Imgs, {len(dataset) - cnt_poisoned} Clean Imgs")
        return dataset_

    def save_image(self, img, folder, filename):
        img = img.astype(np.uint8)
        img = Image.fromarray(img, mode='RGB')
        img.save(os.path.join(folder, filename))

    def show_image(self, img, title="Image"):
        plt.imshow(img)
        plt.title(title)
        plt.show()