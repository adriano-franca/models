import os
import random
import pandas as pd
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= CONFIGURAÇÕES =================
CSV_PATH = 'finding_annotations.csv'
BASE_DIR = '/backup/lucas/datasets/vindr-mammo/images'
PATCH_SIZE = 384

def load_dicom_image(filepath):
    dicom = pydicom.dcmread(filepath)
    img = dicom.pixel_array.astype(np.float32)
    
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = np.max(img) - img
        
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img = (img * 65535.0).astype(np.uint16)
    return img

def get_random_samples(csv_path):
    df = pd.read_csv(csv_path)
    
    # Separar anormais (com bounding box) e normais (sem bounding box)
    df_anormal = df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax'])
    df_normal = df[df['xmin'].isna()]
    
    sample_anormal = df_anormal.sample(1).iloc[0]
    sample_normal = df_normal.sample(1).iloc[0]
    
    return sample_anormal, sample_normal

def get_abnormal_patch_steps(img, xmin, ymin, xmax, ymax, size=384):
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
    
    patch_raw = img[new_ymin:new_ymax, new_xmin:new_xmax]
    patch_resized = cv2.resize(patch_raw, (size, size), interpolation=cv2.INTER_AREA) if patch_raw.size > 0 else patch_raw
    
    return {
        "box_lesion": (xmin, ymin, box_w, box_h), # Caixa original da lesão
        "box_crop": (new_xmin, new_ymin, new_xmax - new_xmin, new_ymax - new_ymin), # Área efetivamente recortada
        "patch_raw": patch_raw,
        "patch_resized": patch_resized
    }

def get_normal_patch_steps(img, size=384):
    h, w = img.shape
    crop_size = min(500, h, w) 
    
    rx, ry = 0, 0
    patch_raw = None
    
    for _ in range(10): 
        rx = random.randint(0, w - crop_size)
        ry = random.randint(0, h - crop_size)
        patch_temp = img[ry:ry+crop_size, rx:rx+crop_size]
        
        if np.mean(patch_temp) > 20:
            patch_raw = patch_temp
            break
            
    # Fallback caso não encontre área com tecido
    if patch_raw is None:
        cy, cx = h//2, w//2
        ry, rx = cy - crop_size//2, cx - crop_size//2
        patch_raw = img[ry:ry+crop_size, rx:rx+crop_size]
        
    patch_resized = cv2.resize(patch_raw, (size, size), interpolation=cv2.INTER_AREA)
    
    return {
        "box_crop": (rx, ry, crop_size, crop_size),
        "patch_raw": patch_raw,
        "patch_resized": patch_resized
    }

def plot_patch_extraction(img_anormal, steps_anormal, info_anormal, img_normal, steps_normal, info_normal):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Visualização da Extração de Recortes (Patches)', fontsize=16, fontweight='bold')
    
    # ================= LINHA 1: ANORMAL =================
    ax_full_anormal = axes[0, 0]
    ax_full_anormal.imshow(img_anormal, cmap='gray')
    ax_full_anormal.set_title(f"Completa ANORMAL\nEstudo: {info_anormal['study_id'][:8]}...")
    ax_full_anormal.axis('off')
    
    # Desenhar Bounding Box da Lesão (Vermelho)
    lx, ly, lw, lh = steps_anormal["box_lesion"]
    rect_lesion = patches.Rectangle((lx, ly), lw, lh, linewidth=2, edgecolor='red', facecolor='none', label='Lesão')
    ax_full_anormal.add_patch(rect_lesion)
    
    # Desenhar Bounding Box do Recorte (Verde)
    cx, cy, cw, ch = steps_anormal["box_crop"]
    rect_crop = patches.Rectangle((cx, cy), cw, ch, linewidth=2, edgecolor='green', linestyle='--', facecolor='none', label='Área do Recorte')
    ax_full_anormal.add_patch(rect_crop)
    ax_full_anormal.legend(loc='upper right')
    
    ax_raw_anormal = axes[0, 1]
    ax_raw_anormal.imshow(steps_anormal["patch_raw"], cmap='gray')
    ax_raw_anormal.set_title(f"Recorte Bruto ({cw}x{ch})")
    ax_raw_anormal.axis('off')
    
    ax_res_anormal = axes[0, 2]
    ax_res_anormal.imshow(steps_anormal["patch_resized"], cmap='gray')
    ax_res_anormal.set_title(f"Redimensionado ({PATCH_SIZE}x{PATCH_SIZE})")
    ax_res_anormal.axis('off')
    
    # ================= LINHA 2: NORMAL =================
    ax_full_normal = axes[1, 0]
    ax_full_normal.imshow(img_normal, cmap='gray')
    ax_full_normal.set_title(f"Completa NORMAL\nEstudo: {info_normal['study_id'][:8]}...")
    ax_full_normal.axis('off')
    
    # Desenhar Bounding Box do Recorte Aleatório (Verde)
    nx, ny, nw, nh = steps_normal["box_crop"]
    rect_norm = patches.Rectangle((nx, ny), nw, nh, linewidth=2, edgecolor='green', linestyle='--', facecolor='none', label='Recorte Aleatório')
    ax_full_normal.add_patch(rect_norm)
    ax_full_normal.legend(loc='upper right')
    
    ax_raw_normal = axes[1, 1]
    ax_raw_normal.imshow(steps_normal["patch_raw"], cmap='gray')
    ax_raw_normal.set_title(f"Recorte Bruto ({nw}x{nh})")
    ax_raw_normal.axis('off')
    
    ax_res_normal = axes[1, 2]
    ax_res_normal.imshow(steps_normal["patch_resized"], cmap='gray')
    ax_res_normal.set_title(f"Redimensionado ({PATCH_SIZE}x{PATCH_SIZE})")
    ax_res_normal.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/comparacao_extracao_patches.png')
    plt.close()
    print("Imagem guardada com sucesso em plots/comparacao_extracao_patches.png")

if __name__ == "__main__":
    print("A selecionar imagens aleatórias...")
    sample_anormal, sample_normal = get_random_samples(CSV_PATH)
    
    path_anormal = os.path.join(BASE_DIR, str(sample_anormal['study_id']), f"{sample_anormal['image_id']}.dicom")
    path_normal = os.path.join(BASE_DIR, str(sample_normal['study_id']), f"{sample_normal['image_id']}.dicom")
    
    print("A carregar as imagens DICOM...")
    img_anormal = load_dicom_image(path_anormal)
    img_normal = load_dicom_image(path_normal)
    
    print("A processar recortes...")
    # Processar anormal
    xmin, ymin = int(float(sample_anormal['xmin'])), int(float(sample_anormal['ymin']))
    xmax, ymax = int(float(sample_anormal['xmax'])), int(float(sample_anormal['ymax']))
    steps_anormal = get_abnormal_patch_steps(img_anormal, xmin, ymin, xmax, ymax, PATCH_SIZE)
    
    # Processar normal
    steps_normal = get_normal_patch_steps(img_normal, PATCH_SIZE)
    
    print("A gerar o gráfico...")
    plot_patch_extraction(img_anormal, steps_anormal, sample_anormal, img_normal, steps_normal, sample_normal)