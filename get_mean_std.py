import os
import torch 
from torchvision import datasets, transforms 
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset 
from tqdm.notebook import tqdm 
from time import time 
import pandas as pd


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

N_CHANNELS = 1 
dataset = datasets.MNIST("data", download=True, train=True, transform=transforms.ToTensor()) 
full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count()) 
before = time() 
mean = torch.zeros(1) 
std = torch.zeros(1) 

print('==> Computing mean and std..') 
for inputs, _labels in tqdm(full_loader): 
	for i in range(N_CHANNELS): 
		mean[i] += inputs[:,i,:,:].mean() 
		std[i] += inputs[:,i,:,:].std() 
mean.div_(len(dataset)) 
std.div_(len(dataset)) 
print(mean, std) 
print("time elapsed: ", time()-before)
