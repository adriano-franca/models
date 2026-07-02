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
TARGET_W = 896
TARGET_H = 1152

os.makedirs(os.path.join(OUTPUT_DIR, 'normal'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'anormal'), exist_ok=True)
# =================================================

def process_and_track_bbox(dicom_path, row, target_w=896, target_h=1152):
    """
    Aplica o pipeline exato do Dual-View e arrasta as coordenadas da bbox junto.
    """
    dicom = pydicom.dcmread(dicom_path)
    img = dicom.pixel_array.astype(np.float32)
    
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = np.max(img) - img

    h_orig, w_orig = img.shape
    has_bbox = not pd.isna(row.get('xmin'))
    
    if has_bbox:
        xmin, ymin = float(row['xmin']), float(row['ymin'])
        xmax, ymax = float(row['xmax']), float(row['ymax'])
    else:
        xmin, ymin, xmax, ymax = 0, 0, 0, 0

    # 1. LATERALIDADE
    laterality = getattr(dicom, 'ImageLaterality', 'R')
    if laterality == 'L':
        img = np.fliplr(img)
        img = np.ascontiguousarray(img)
        if has_bbox:
            xmin_new = w_orig - xmax
            xmax_new = w_orig - xmin
            xmin, xmax = xmin_new, xmax_new

    # 2. OTSU CROP
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        coords = cv2.findNonZero(clean_mask)
        
        if coords is not None:
            crop_x, crop_y, crop_w, crop_h = cv2.boundingRect(coords)
            img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            clean_mask = clean_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # Atualiza BBox (Offset)
            if has_bbox:
                xmin -= crop_x
                xmax -= crop_x
                ymin -= crop_y
                ymax -= crop_y
                
            # Clipping de Anomalias (igual ao seu preprocessing)
            tissue_pixels = img[clean_mask > 0]
            if len(tissue_pixels) > 0:
                mu, sigma = np.mean(tissue_pixels), np.std(tissue_pixels)
                img = np.clip(img, mu - 4*sigma, mu + 4*sigma)

    # 3. RESIZE E PAD
    h_crop, w_crop = img.shape
    scale = min(target_w / w_crop, target_h / h_crop)
    new_w, new_h = int(w_crop * scale), int(h_crop * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    img_final = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    # Atualiza BBox (Scale + Shift)
    if has_bbox:
        xmin = (xmin * scale) + left
        xmax = (xmax * scale) + left
        ymin = (ymin * scale) + top
        ymax = (ymax * scale) + top

    # 4. NORMALIZAÇÃO (Mantendo o seu formato 16-bits para os Patches)
    img_final = (img_final - np.min(img_final)) / (np.max(img_final) - np.min(img_final) + 1e-8)
    img_final = (img_final * 65535.0).astype(np.uint16)

    bbox_final = (xmin, ymin, xmax, ymax) if has_bbox else None
    return img_final, bbox_final

def extract_exact_patch(img, bbox, size=384):
    """ Corta os 384x384 centrados exatamente na nova BBox. """
    xmin, ymin, xmax, ymax = bbox
    cx = int((xmin + xmax) // 2)
    cy = int((ymin + ymax) // 2)
    
    h, w = img.shape
    half = size // 2
    
    # Garante que o recorte não sai para fora da imagem (ajusta para as bordas)
    px = max(0, min(cx - half, w - size))
    py = max(0, min(cy - half, h - size))
    
    patch = img[py:py+size, px:px+size]
    return patch

def extract_random_normal_patch(img, size=384):
    """ Corta 384x384 do tecido saudável MAIS DENSO possível (Hard Negative). """
    h, w = img.shape
    
    melhor_patch = None
    maior_media = -1
    
    # 30 tentativas: equilíbrio entre procurar tecido denso e velocidade de extração
    for _ in range(30):
        rx = random.randint(0, w - size)
        ry = random.randint(0, h - size)
        patch = img[ry:ry+size, rx:rx+size]
        
        media_atual = np.mean(patch)
        
        # 1. Se encontrou o novo limiar (> 18000), devolve imediatamente
        if media_atual > 18000:
            return patch
            
        # 2. Vai guardando o patch mais branco (denso) que encontrar no caminho
        if media_atual > maior_media:
            maior_media = media_atual
            melhor_patch = patch
            
    # 3. Fallback: Se após 30 tentativas a mama for muito gordurosa e não tiver chegado a 18000,
    # devolvemos o patch mais denso que conseguimos arranjar (desde que não seja fundo preto).
    if maior_media > 10000:
        return melhor_patch
        
    return None

# ================= EXECUÇÃO =================
if __name__ == "__main__":
    print("A carregar o ficheiro de anotações...")
    df = pd.read_csv(CSV_PATH)

    count_anormal = 0
    count_normal = 0
    erros_leitura = 0

    print(f"A processar {len(df)} anotações... A criar as fundações corretas!")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        study_id = str(row['study_id'])
        image_id = str(row['image_id'])
        
        dicom_path = os.path.join(BASE_DIR, study_id, f"{image_id}.dicom")
        if not os.path.exists(dicom_path):
            continue
            
        has_bbox = not pd.isna(row.get('xmin'))
        
        # Sub-amostragem de normais
        if not has_bbox:
            chance_de_guardar = 1.0 if count_normal < 50 else 0.2
            if random.random() > chance_de_guardar:
                continue
                
        try:
            # MAGIA ACONTECE AQUI: Imagem pre-processada E coordenadas ajustadas
            img_processed, new_bbox = process_and_track_bbox(dicom_path, row, TARGET_W, TARGET_H)
            
            if has_bbox and new_bbox is not None:
                patch = extract_exact_patch(img_processed, new_bbox, PATCH_SIZE)
                if patch.size > 0:
                    save_path = os.path.join(OUTPUT_DIR, 'anormal', f"{image_id}_{idx}.png")
                    cv2.imwrite(save_path, patch)
                    count_anormal += 1
            else:
                patch = extract_random_normal_patch(img_processed, PATCH_SIZE)
                if patch is not None and patch.size > 0:
                    save_path = os.path.join(OUTPUT_DIR, 'normal', f"{image_id}_{idx}.png")
                    cv2.imwrite(save_path, patch)
                    count_normal += 1
                    
        except Exception as e:
            erros_leitura += 1

    print("\n=== EXTRAÇÃO GEOMETRICAMENTE CORRETA CONCLUÍDA ===")
    print(f"Patches Anormais: {count_anormal}")
    print(f"Patches Normais: {count_normal}")
    if erros_leitura > 0:
        print(f"Aviso: {erros_leitura} falhas.")