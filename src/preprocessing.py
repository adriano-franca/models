import os
import numpy as np
import pydicom
import cv2

# ====================================================================
# 1. FUNÇÕES ORIGINAIS INTACTAS
# ====================================================================
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

# ====================================================================
# 2. NOVA LÓGICA: OTSU + CLIPPING + CROP (Recorte do Fundo Preto)
# ====================================================================
def apply_otsu_clip_and_crop(img):
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---> NOVO: Isolar APENAS a mama (Maior Componente Conectado) <---
    # Encontra todos os contornos/manchas na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontra o contorno com a maior área (que será 100% de certeza o tecido mamário)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Cria uma máscara totalmente limpa (preta)
        clean_mask = np.zeros_like(mask)
        
        # Desenha apenas a mama (maior contorno) nesta nova máscara, pintando de branco
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Agora a máscara limpa só tem a mama, ignorando as letras "CC-L" ou "MLO"
        mask = clean_mask
    # ----------------------------------------------------------------

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

    # Descobre a "Caixa" usando a máscara limpa (sem o texto)
    coords = cv2.findNonZero((mask > 0).astype(np.uint8))
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Corta a imagem exata onde a mama começa e termina
        img_cropped = img_clipped[y:y+h, x:x+w]
        return img_cropped
    else:
        return img_clipped

# ====================================================================
# 3. NOVA LÓGICA: RESIZE PROPORCIONAL + PADDING (Letterbox)
# ====================================================================
def resize_and_pad(img, target_width, target_height):
    h, w = img.shape
    
    # Encontra o fator de escala que não deforma a imagem
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Redimensiona mantendo a proporção exata (microcalcificações continuam redondas)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calcula quanto espaço preto precisamos adicionar para chegar ao target_width/height
    delta_w = target_width - new_w
    delta_h = target_height - new_h
    
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Adiciona as barras pretas em volta (padding)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    return img_padded

# ====================================================================
# 4. O PIPELINE PRINCIPAL ATUALIZADO
# ====================================================================
def process_dicom(file_path, laterality, target_width=896, target_height=1152):
    
    # 1. Carrega o array em float32
    img = load_dicom_array(file_path)

    # 2. Padroniza a lateralidade (Todos virados para a direita)
    img = align_laterality(img, laterality)

    # 3. Aplica a máscara, clipping de anomalias E recorta o fundo inútil
    img = apply_otsu_clip_and_crop(img)

    # 4. Redimensiona preservando a anatomia real + preenchimento
    img_resized = resize_and_pad(img, target_width, target_height)

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