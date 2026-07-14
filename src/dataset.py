#dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import albumentations as A
import cv2 # Certifique-se de instalar: pip install opencv-python

from src.preprocessing import process_dicom

BASE_DIR = "/backup/lucas/datasets/vindr-mammo/images"

# --- FUNÇÃO AUXILIAR ---
def apply_clahe(img):
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8) / 255.0

# ================= ALTERAÇÃO (c): REGULARIZAÇÃO / AUGMENTAÇÃO MAIS FORTE =================
# Mesma lógica aplicada em train_patches.py: mais transformações e com maior probabilidade,
# para reduzir overfitting. Também adicionamos translate ao Affine (antes só rotate/shear)
# e a normalização final (antes as imagens saíam sem normalizar, só em [0, 1]).
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(1.0, 1.0),
            translate_percent=(-0.1, 0.1),
            rotate=[-15, 15],
            shear=[-10, 10],
            p=0.8
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.5),
        A.Sharpen(alpha=(0.2, 0.4), lightness=(0.8, 1.2), p=0.3),   # equivalente ao RandomAdjustSharpness
        # A.Equalize removido: cv2.equalizeHist exige imagem uint8 de 1 canal (CV_8UC1) e
        # quebra com nossas imagens float em [0,1] (cv2.error: Assertion failed em equalizeHist).
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),  # mesma normalização usada nos patches
        A.ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
        A.ToTensorV2()
    ])
# =============================================================================================

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
        laterality = self.df.iloc[idx]['laterality']
        
        # ================= INJEÇÃO DE DENSIDADE (BI-RADS) =================
        raw_density = str(self.df.iloc[idx].get('breast_density', 'C')).upper()
        
        # Mapeamento do BI-RADS para índice do vetor
        density_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
        dens_idx = density_map.get(raw_density, 2)
        density_tensor = torch.zeros(4, dtype=torch.float32)
        density_tensor[dens_idx] = 1.0
        
        img_cc = np.squeeze(process_dicom(path_cc, laterality=laterality))
        img_mlo = np.squeeze(process_dicom(path_mlo, laterality=laterality))

        # --- APLICAÇÃO CLAHE ---
        img_cc = apply_clahe(img_cc)
        img_mlo = apply_clahe(img_mlo)

        if self.transform:
            img_cc = self.transform(image=img_cc)['image']
            img_mlo = self.transform(image=img_mlo)['image']
        else:
            img_cc = torch.from_numpy(img_cc).unsqueeze(0)
            img_mlo = torch.from_numpy(img_mlo).unsqueeze(0)
        
        # O retorno agora contém 4 elementos
        return img_cc, img_mlo, density_tensor, torch.tensor(label, dtype=torch.float32)