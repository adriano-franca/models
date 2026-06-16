import os
import numpy as np
import pydicom
import cv2

# Leitura de arquivos DICOM e inversão de cor caso tenha fundo branco (MONOCHROME1)
def load_dicom_array(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    dicom = pydicom.dcmread(file_path)
    img = dicom.pixel_array.astype(np.float32)

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = np.max(img) - img

    return img

def align_laterality(img, laterality):
    if laterality == 'L':
        img = np.fliplr(img)
        img = np.ascontiguousarray(img)

    return img

def apply_otsu_and_clip(img):
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_tissue = img * (mask > 0)

    tissue_pixels = img_tissue[mask > 0]

    if len(tissue_pixels) == 0:
        return img
    
    mu = np.mean(tissue_pixels)
    sigma = np.std(tissue_pixels)

    lower_bound = mu - (4 * sigma)
    upper_bound = mu + (4 * sigma)

    img_clipped = np.copy(img_tissue)
    img_clipped = np.clip(img_clipped, lower_bound, upper_bound)

    img_clipped = img_clipped * (mask > 0)

    return img_clipped

def process_dicom(file_path, laterality, target_width=896, target_height=1152):
    
    # 1. Carrega o array em float32
    img = load_dicom_array(file_path)

    # 2. Padroniza a lateralidade usando a informação exata do CSV
    img = align_laterality(img, laterality)

    # 3. Aplica a máscara e o clipping de anomalias de brilho
    img = apply_otsu_and_clip(img)

    # 4. Redimensiona para o tamanho de entrada da rede
    img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # 5. Normalização Z-Score
    mu_img = np.mean(img_resized)
    std_img = np.std(img_resized)

    if std_img > 0:
        img_normalized = (img_resized - mu_img) / std_img
    else:
        img_normalized = img_resized - mu_img

    # 6. Adiciona a dimensão do canal (1, H, W) para o PyTorch
    img_tensor_ready = np.expand_dims(img_normalized, axis=0)

    return img_tensor_ready