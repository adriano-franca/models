import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import albumentations as A
import albumentations.pytorch as AP

from src.preprocessing import process_dicom

BASE_DIR = "/backup/lucas/datasets/vindr-mammo/images"

def get_train_transforms():
    return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.8, 1.2), rotate=[-25,25], shear=[-12, 12], p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.5),
    A.ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([
        A.ToTensorV2()
    ])

class PatchDataset(Dataset):
    def __init__(self, dataframe, transfom):
        self.df = dataframe
        self.transform = transfom
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        label = self.df.iloc[idx]['label']

        if img_path.endswith('.png'):
            image = np.load(img_path).astype(np.float32)
        else:
            pass

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)
    
class TwoViewMammogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        raw_cc = str(self.df.iloc[idx]['path_cc'])
        raw_mlo = str(self.df.iloc[idx]['path_mlo'])

        paciente_cc, arquivo_cc = raw_cc.split('/')[-2:]
        paciente_mlo, arquivo_mlo = raw_mlo.split('/')[-2:]

        path_cc = os.path.join(BASE_DIR, paciente_cc, arquivo_cc)
        path_mlo = os.path.join(BASE_DIR, paciente_mlo, arquivo_mlo)

        label = self.df.iloc[idx]['target']

        img_cc = process_dicom(path_cc)
        img_mlo = process_dicom(path_mlo)

        img_cc = np.squeeze(img_cc)
        img_mlo = np.squeeze(img_mlo)

        if self.transform:
            aug_cc = self.transform(image=img_cc)
            img_cc = aug_cc['image']

            aug_mlo = self.transform(image=img_mlo)
            img_mlo = aug_mlo['image']
        else:
            img_cc = torch.from_numpy(img_cc).unsqueeze(0)
            img_mlo = torch.from_numpy(img_mlo).unsqueeze(0)
        
        return img_cc, img_mlo, torch.tensor(label, dtype=torch.float32)