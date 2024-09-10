#patient_id,image_path,mask_path,mask 
import csv
import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision import transforms,utils
from PIL import Image
import pandas as pd

data_mask = pd.read_csv('Healthcare_AI_Datasets/Brain_MRI/data_mask.csv')


class BrainDataset():

    def __init__(self, csv_file, root_dir, transform=None):
        """
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = self.data_frame.iloc[idx, 1]
        brain_img_path = os.path.join(self.root_dir,
                                image_path)
        brain_img = Image.open(brain_img_path)

        if self.transform:
            brain_img = self.transform(brain_img)

        mask = self.data_frame.iloc[idx, 3]
        mask = torch.tensor(mask, dtype=torch.float32)

        return brain_img, mask

    def get_data_loaders(self, batchsize = 32, train_split = 0.7, val_split = 0.15):
        #Calculates the split size
        total_size = len(self)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size


        train_dataset, val_dataset, test_dataset = random_split(self, [train_size, val_size, test_size])

        #Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

        return train_loader, val_loader, test_loader





