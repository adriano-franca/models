import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# 1. CONFIGURAÇÕES
# ---------------------------------------------------------
CSV_PATH = "finding_annotations.csv"
DICOM_DIR = "/backup/lucas/datasets/vindr-mammo/images"                 # Onde estão as suas pastas de pacientes (study_id)
OUTPUT_DIR = "./patches_dataset" # Nova pasta onde os recortes PNG serão guardados
OUTPUT_CSV = "patches_annotations.csv"

PATCH_SIZE = 512
STRIDE = 256
BG_THRESHOLD = 15

# ---------------------------------------------------------
# 2. FUNÇÕES AUXILIARES
# ---------------------------------------------------------
def check_overlap(patch, lesion):
    """Verifica se o recorte interseta a caixa delimitadora (bounding box) da anomalia."""
    ix_min, iy_min = max(patch[0], lesion[0]), max(patch[1], lesion[1])
    ix_max, iy_max = min(patch[2], lesion[2]), min(patch[3], lesion[3])
    return (ix_min < ix_max) and (iy_min < iy_max)

def create_dirs():
    """Garante que a pasta de saída existe."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# ---------------------------------------------------------
# 3. LÓGICA PRINCIPAL DE EXTRAÇÃO
# ---------------------------------------------------------
def main():
    create_dirs()
    print("A carregar anotações...")
    df = pd.read_csv(CSV_PATH)
    
    # Agrupar as anotações por imagem para não abrir o mesmo DICOM várias vezes
    grouped = df.groupby('image_id')
    
    new_csv_data = []
    
    print(f"Iniciando a extração para {len(grouped)} imagens DICOM...")
    
    # tqdm cria a barra de progresso para acompanharmos o tempo restante
    for image_id, group in tqdm(grouped, desc="A extrair recortes"):
        study_id = group.iloc[0]['study_id']
        img_label = 0.0 if group.iloc[0]['finding_categories'] == "['No Finding']" else 1.0
        
        # Mapear as caixas delimitadoras (lesões) desta imagem
        boxes = []
        for _, row in group.iterrows():
            if pd.notnull(row['xmin']) and row['finding_categories'] != "['No Finding']":
                boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        
        # Caminho para o ficheiro DICOM
        dicom_path = os.path.join(DICOM_DIR, study_id, f"{image_id}.dicom")
        
        if not os.path.exists(dicom_path):
            continue # Salta se o ficheiro não existir
            
        try:
            dcm = pydicom.dcmread(dicom_path)
            img = dcm.pixel_array
        except Exception as e:
            continue
            
        h, w = img.shape
        
        # Varrer a imagem em formato de grelha
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                
                # Extrair o recorte
                y_max = min(y + PATCH_SIZE, h)
                x_max = min(x + PATCH_SIZE, w)
                patch_img = img[y:y_max, x:x_max]
                
                # Preenchimento (padding) caso o recorte bata na borda e seja menor que 512
                if patch_img.shape[0] < PATCH_SIZE or patch_img.shape[1] < PATCH_SIZE:
                    pad_y = PATCH_SIZE - patch_img.shape[0]
                    pad_x = PATCH_SIZE - patch_img.shape[1]
                    patch_img = np.pad(patch_img, ((0, pad_y), (0, pad_x)), mode='constant')
                
                # Descartar recortes que são quase totalmente fundo preto
                if patch_img.mean() <= BG_THRESHOLD:
                    continue
                
                # Determinar o rótulo deste recorte
                patch_box = [x, y, x + PATCH_SIZE, y + PATCH_SIZE]
                label = 0.0
                for box in boxes:
                    if check_overlap(patch_box, box):
                        label = 1.0
                        break
                
                # Guardar a imagem PNG
                patch_filename = f"{image_id}_x{x}_y{y}.png"
                patch_filepath = os.path.join(OUTPUT_DIR, patch_filename)
                
                # Converter para PIL e guardar (otimizado para não perder textura)
                pil_img = Image.fromarray(patch_img).convert('L') # 'L' para tons de cinza
                pil_img.save(patch_filepath)
                
                # Registar no novo CSV
                new_csv_data.append({
                    'patch_filename': patch_filename,
                    'image_id': image_id,
                    'study_id': study_id,
                    'x': x,
                    'y': y,
                    'label': label,
                    'img_label': img_label
                })

    # Guardar o novo dataset tabular
    print("\nA extração foi concluída! A guardar o novo CSV...")
    new_df = pd.DataFrame(new_csv_data)
    new_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Sucesso! {len(new_df)} recortes válidos foram gerados.")
    print(f"As imagens estão em: {OUTPUT_DIR}")
    print(f"O ficheiro de controlo é: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()