import os
import pandas as pd
import pydicom
import numpy as np
import cv2
from tqdm import tqdm
import random

# ================= CONFIGURAÇÕES =================
CSV_PATH = 'finding_annotations_split.csv'
BASE_DIR = '/backup/lucas/datasets/vindr-mammo/images'
OUTPUT_DIR = 'dataset_patches'
PATCH_SIZE = 384

# Cria as pastas para as duas classes
os.makedirs(os.path.join(OUTPUT_DIR, 'normal'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'anormal'), exist_ok=True)
# =================================================

def load_dicom_image(filepath):
    dicom = pydicom.dcmread(filepath)
    img = dicom.pixel_array.astype(np.float32)
    
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = np.max(img) - img
        
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img = (img * 65535.0).astype(np.uint16)
    return img

def crop_and_resize(img, xmin, ymin, xmax, ymax, size=384):
    h, w = img.shape
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    
    box_w = xmax - xmin
    box_h = ymax - ymin
    side = int(max(box_w, box_h) * 1.1) 
    
    new_xmin = max(0, cx - side // 2)
    new_ymin = max(0, cy - side // 2)
    new_xmax = min(w, cx + side // 2)
    new_ymax = min(h, cy + side // 2)
    
    patch = img[new_ymin:new_ymax, new_xmin:new_xmax]
    if patch.size > 0:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch

def extract_normal_patch(img, size=384):
    h, w = img.shape
    
    crop_size = random.randint(150, 450) 
    
    for _ in range(30): # Tentamos 30 vezes achar tecido com o tamanho gerado
        rx = random.randint(0, w - crop_size)
        ry = random.randint(0, h - crop_size)
        patch = img[ry:ry+crop_size, rx:rx+crop_size]
        
        # O limiar de 15000 garante que não apanhamos o fundo preto (escala 16-bits)
        if np.mean(patch) > 15000:
            return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
            
    # Fallback se falhar as 30 tentativas (tenta tirar do centro exato da imagem)
    return None

# ================= EXECUÇÃO =================
print("A carregar o ficheiro de anotações...")
df = pd.read_csv(CSV_PATH)

# FILTRO OTIMIZADO: Apenas as linhas que pertencem ao conjunto de Treino
df_treino = df[df['split'] == 'training'].copy()

count_anormal = 0
count_normal = 0
erros_leitura = 0

print(f"A processar {len(df_treino)} anotações do conjunto de treino...")
print("A extrair recortes... Isto pode demorar alguns minutos.")

for idx, row in tqdm(df_treino.iterrows(), total=len(df_treino)):
    study_id = str(row['study_id'])
    image_id = str(row['image_id'])
    
    dicom_path = os.path.join(BASE_DIR, study_id, f"{image_id}.dicom")
    if not os.path.exists(dicom_path):
        continue
        
    has_bbox = not pd.isna(row['xmin'])
    
    # Se for NORMAL, aplica a sua lógica de descarte para equilibrar os dados (~20%)
    if not has_bbox:
        chance_de_guardar = 1.0 if count_normal < 50 else 0.2
        if random.random() > chance_de_guardar:
            continue
            
    try:
        # CORREÇÃO 2 (Textura): Lemos a imagem original para ambos os casos.
        # Ambas as classes vão usar exatamente a mesma fonte de píxeis para evitar enviesamento matemático.
        img = load_dicom_image(dicom_path)
        
        if has_bbox:
            # É ANORMAL (Tem Bounding Box)
            xmin, ymin = int(float(row['xmin'])), int(float(row['ymin']))
            xmax, ymax = int(float(row['xmax'])), int(float(row['ymax']))
            
            patch = crop_and_resize(img, xmin, ymin, xmax, ymax, PATCH_SIZE)
            if patch is not None and patch.size > 0:
                save_path = os.path.join(OUTPUT_DIR, 'anormal', f"{image_id}_{idx}.png")
                cv2.imwrite(save_path, patch)
                count_anormal += 1
                
        else:
            # É NORMAL (Sem coordenadas)
            patch = extract_normal_patch(img, PATCH_SIZE)
            if patch is not None and patch.size > 0:
                save_path = os.path.join(OUTPUT_DIR, 'normal', f"{image_id}_{idx}.png")
                cv2.imwrite(save_path, patch)
                count_normal += 1
                
    except Exception as e:
        erros_leitura += 1

print("\n=== EXTRAÇÃO CONCLUÍDA ===")
print(f"Patches Anormais de Treino: {count_anormal}")
print(f"Patches Normais de Treino: {count_normal}")
if erros_leitura > 0:
    print(f"Aviso: {erros_leitura} imagens não puderam ser lidas (DICOM corrompido).")