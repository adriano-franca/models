import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import (
    load_dicom_array,
    align_laterality,
    apply_otsu_clip_and_crop,
    resize_and_pad
)

def get_random_patient_views(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    df_train = df[df['split'] == 'training']
    
    grouped = df_train.groupby(['study_id', 'laterality'])
    
    valid_groups = []
    for name, group in grouped:
        views = group['view_position'].unique()
        if 'CC' in views and 'MLO' in views:
            valid_groups.append(name)
            
    selected_study, selected_lat = random.choice(valid_groups)
    patient_data = df_train[(df_train['study_id'] == selected_study) & (df_train['laterality'] == selected_lat)]
    
    row_cc = patient_data[patient_data['view_position'] == 'CC'].iloc[0]
    row_mlo = patient_data[patient_data['view_position'] == 'MLO'].iloc[0]
    
    path_cc = os.path.join(base_dir, selected_study, f"{row_cc['image_id']}.dicom")
    path_mlo = os.path.join(base_dir, selected_study, f"{row_mlo['image_id']}.dicom")
    
    return path_cc, path_mlo, selected_lat

def get_pipeline_steps(file_path, laterality, target_width=896, target_height=1152):
    steps = {}
    
    img_raw = load_dicom_array(file_path)
    steps['Original'] = img_raw
    
    img_aligned = align_laterality(img_raw, laterality)
    steps['Alinhada'] = img_aligned
    
    img_cropped = apply_otsu_clip_and_crop(img_aligned)
    steps['Recortada'] = img_cropped
    
    img_resized = resize_and_pad(img_cropped, target_width, target_height)
    steps['Redimensionada'] = img_resized
    
    mu_img = np.mean(img_resized)
    std_img = np.std(img_resized)
    img_norm = (img_resized - mu_img) / std_img if std_img > 0 else img_resized - mu_img
    steps['Normalizada (Z-Score)'] = img_norm
    
    return steps

def plot_step_by_step_comparison(steps_cc, steps_mlo, study_id):
    step_names = list(steps_cc.keys())
    n_steps = len(step_names)
    
    fig, axes = plt.subplots(n_steps, 2, figsize=(12, 5 * n_steps))
    fig.suptitle(f'Evolução do Pré-processamento: CC vs MLO\nPaciente: {study_id}', fontsize=16, fontweight='bold')
    
    for i, step in enumerate(step_names):
        ax_cc = axes[i, 0]
        im_cc = ax_cc.imshow(steps_cc[step], cmap='gray')
        ax_cc.set_title(f'Vista CC - {step}')
        ax_cc.axis('off')
        fig.colorbar(im_cc, ax=ax_cc, fraction=0.046, pad=0.04)
        
        ax_mlo = axes[i, 1]
        im_mlo = ax_mlo.imshow(steps_mlo[step], cmap='gray')
        ax_mlo.set_title(f'Vista MLO - {step}')
        ax_mlo.axis('off')
        fig.colorbar(im_mlo, ax=ax_mlo, fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join('plots', 'comparacao_pre_processamento.png'))
    plt.close()
    print("Imagem guardada com sucesso em plots/comparacao_pre_processamento.png")

if __name__ == "__main__":
    CSV_PATH = "finding_annotations.csv"
    BASE_DIR = "/backup/lucas/datasets/vindr-mammo/images" 
    
    os.makedirs('plots', exist_ok=True)
    
    cc_path, mlo_path, lat = get_random_patient_views(CSV_PATH, BASE_DIR)
    
    patient_id = cc_path.split('/')[-2]
    
    steps_cc = get_pipeline_steps(cc_path, lat)
    steps_mlo = get_pipeline_steps(mlo_path, lat)
    
    plot_step_by_step_comparison(steps_cc, steps_mlo, patient_id)