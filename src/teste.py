import os
import matplotlib.pyplot as plt

# Importa as funções do seu ficheiro preprocessing.py (que está na mesma pasta)
from preprocessing import (
    load_dicom_array, 
    align_laterality, 
    apply_otsu_clip_and_crop, 
    resize_and_pad
)

def visualize_pipeline(dicom_path, laterality, target_w=896, target_h=1152):
    print(f"A processar: {dicom_path}")
    
    # Passo 1: Carregar e alinhar (Imagem Bruta)
    img_raw = load_dicom_array(dicom_path)
    img_aligned = align_laterality(img_raw, laterality)
    
    # Passo 2: Otsu Mask e Crop (Remove o fundo)
    img_cropped = apply_otsu_clip_and_crop(img_aligned)
    
    # Passo 3: Resize Proporcional e Padding (Pronto para a rede)
    img_final = resize_and_pad(img_cropped, target_w, target_h)
    
    # ==========================================
    # PLOT DOS RESULTADOS LADO A LADO
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Teste do Pipeline de Pré-processamento: "Crop & Pad"', fontsize=16, fontweight='bold')
    
    # Imagem 1: Original Alinhada
    axes[0].imshow(img_aligned, cmap='gray')
    axes[0].set_title(f"1. Original Alinhada\nShape: {img_aligned.shape}")
    axes[0].axis('off')
    
    # Imagem 2: Apenas o Tecido (Cropped)
    axes[1].imshow(img_cropped, cmap='gray')
    axes[1].set_title(f"2. Bounding Box (Otsu + Crop)\nShape: {img_cropped.shape}")
    axes[1].axis('off')
    
    # Imagem 3: Padded & Resized (Entrada da ConvNeXt)
    axes[2].imshow(img_final, cmap='gray')
    axes[2].set_title(f"3. Final: Resize Proporcional + Pad\nShape: {img_final.shape}")
    # Desenhar um contorno vermelho para evidenciar onde as barras pretas (padding) foram adicionadas
    axes[2].plot([0, target_w, target_w, 0, 0], [0, 0, target_h, target_h, 0], color='red', lw=2, linestyle='--')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('teste_pipeline.png', dpi=300)
    print("✅ Sucesso! O gráfico foi guardado como 'teste_pipeline.png' na pasta atual.")
    plt.close()

if __name__ == "__main__":
    # Substitua pelo caminho exato de UM ficheiro DICOM de teste no seu servidor
    # Usei o seu BASE_DIR como exemplo
    SAMPLE_DICOM_PATH = "/backup/lucas/datasets/vindr-mammo/images/00a369b4ec1e5e0ff34e6bd838e5f2d6/f3cbed97f4bb7897467e1e8bab45966e.dicom"
    
    # Defina se é L (Esquerda) ou R (Direita) de acordo com o CSV
    LATERALITY = 'L' 
    
    if os.path.exists(SAMPLE_DICOM_PATH):
        visualize_pipeline(SAMPLE_DICOM_PATH, LATERALITY)
    else:
        print(f"ERRO: Não foi possível encontrar o ficheiro em:\n{SAMPLE_DICOM_PATH}\nPor favor, atualize o caminho no script.")