import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import time


class MyDataset(Dataset):

    def __init__(self, dataset, target, portion=0.1, mode="train", device=torch.device("cuda")):
        self.dataset = self.addTrigger(dataset, target, portion, mode)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img = np.array(self.dataset[item][0])
        img = img[..., np.newaxis]
        img = torch.Tensor(img).permute(2, 0, 1)
        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def addTrigger(self, dataset, target, portion, mode):
        dataset_list = np.random.permutation(len(dataset))
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        
        dataset_ = list()
        
        poisoned_folder = "poisoned_images"
        os.makedirs(poisoned_folder, exist_ok=True)
        cnt1 = 0
        cnt2 = 0
        
        for j in tqdm(range(len(dataset))):
            i = dataset_list[j]
            data = dataset[i]
            img = np.array(data[0])
            img = img.squeeze(0)
            width = img.shape[0]
            height = img.shape[1]
            if i in perm:
                img[width - 3, height - 3] = 1
                img[width - 3, height - 2] = 1
                img[width - 2, height - 3] = 1
                img[width - 2, height - 2] = 1
                
                if cnt1 < 5:
                    self.save_image(img, poisoned_folder, cnt1)
                    self.show_image(img, f"Poisoned Image original class {data[1]} target class {target}")
                    cnt1+=1
                
                dataset_.append((img, target))
                
            else:
                if cnt2 < 5:
                    self.save_image(img, poisoned_folder, cnt2)
                    self.show_image(img, f"Clear Image class {data[1]}")
                    cnt2+=1

                dataset_.append((img, data[1]))
        time.sleep(0.1)
        print("Injecting Over: " + str(len(perm)) + " Bad Imgs, " + str(len(dataset_) - len(perm)) + " Clean Imgs")
        return dataset_

    def save_image(self, img, folder, index):
        img = img.astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        img.save(os.path.join(folder, f"poisoned_{index}.png"))

    def show_image(self, img, title="Image"):
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()
