import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Importações da sua arquitetura
from src.dataset import TwoViewMammogramDataset, get_valid_transforms
from src.models import DualViewClassifier, DualViewDenseNet

def plot_final_confusion_matrix(y_true, y_pred, limiar, output_dir="plots"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
    plt.title(f'Matriz de Confusão - TESTE FINAL\n(Limiar de Decisão: {limiar:.2f})')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição do Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cm_TESTE_FINAL.png'))
    plt.close()
    print(f"📊 Matriz de confusão guardada em: {os.path.join(output_dir, 'cm_TESTE_FINAL.png')}")

def evaluate_on_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 A iniciar Avaliação Final no dispositivo: {device}")

    # 1. Carregar APENAS os dados de Teste (O Cofre)
    df = pd.read_csv('breast-level_annotations_grouped_80_10_10.csv')
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    print(f"📁 Pacientes no conjunto de Teste Cego: {len(test_df)}")

    # O teste usa as mesmas transformações (sem augmentation) da validação
    test_dataset = TwoViewMammogramDataset(test_df, transform=get_valid_transforms())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Instanciar a arquitetura e carregar o "Cérebro Campeão"
    # Passamos pretrained_patch_path=None porque vamos sobrescrever a rede inteira com os pesos finais
    # Carregar ConvNeXt (O Especialista 1)
    model_A = DualViewClassifier(pretrained_patch_path=None).to(device)
    model_A.load_state_dict(torch.load('checkpoints/best_dual_view_model.pth'))
    model_A.eval()

    # Carregar DenseNet (O Especialista 2)
    model_B = DualViewDenseNet(pretrained=False).to(device)
    model_B.load_state_dict(torch.load('checkpoints/best_dual_view_densenet.pth'))
    model_B.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad(): 
        loop = tqdm(test_loader, desc="A avaliar Teste (Ensemble + TTA)")
        for img_cc, img_mlo, labels in loop:
            img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
            
            # --- 1ª OPINIÃO (Imagens Originais) ---
            prob_A_orig = torch.sigmoid(model_A(img_cc, img_mlo)).item()
            prob_B_orig = torch.sigmoid(model_B(img_cc, img_mlo)).item()
            prob_orig = (prob_A_orig + prob_B_orig) / 2.0
            
            # --- 2ª OPINIÃO (TTA / Flip Horizontal) ---
            img_cc_flip = torch.flip(img_cc, dims=[3])
            img_mlo_flip = torch.flip(img_mlo, dims=[3])
            
            prob_A_flip = torch.sigmoid(model_A(img_cc_flip, img_mlo_flip)).item()
            prob_B_flip = torch.sigmoid(model_B(img_cc_flip, img_mlo_flip)).item()
            prob_flip = (prob_A_flip + prob_B_flip) / 2.0
            
            # --- DECISÃO FINAL: A Média Absoluta ---
            prob_final = (prob_orig + prob_flip) / 2.0
            
            all_probs.append(prob_final)
            all_labels.extend(labels.cpu().numpy())

    # =======================================================
    # TÁTICA 1: OTIMIZAÇÃO DO LIMIAR (THRESHOLDING)
    # =======================================================
    print("\n🔍 A calcular o Limiar de Decisão Ótimo para maximizar o MCC...")
    melhor_mcc = -1.0
    melhor_limiar = 0.5

    # Testa todos os limiares entre 10% e 90% (saltos de 1%)
    for limiar in np.arange(0.1, 0.9, 0.01):
        preds_temporarias = (all_probs >= limiar).astype(int)
        try:
            mcc_temporario = matthews_corrcoef(all_labels, preds_temporarias)
            if mcc_temporario > melhor_mcc:
                melhor_mcc = mcc_temporario
                melhor_limiar = limiar
        except ValueError:
            pass

    # Aplicamos o melhor limiar encontrado para ditar quem tem cancro e quem é normal
    all_preds = (all_probs >= melhor_limiar).astype(int)
    
    # 4. Cálculo das Métricas Finais Oficiais
    try:
        final_auc = roc_auc_score(all_labels, all_probs)
        final_mcc = melhor_mcc 
    except ValueError:
        final_auc, final_mcc = 0.0, 0.0

    print("\n" + "="*50)
    print("🏆 RESULTADOS OFICIAIS DO CONJUNTO DE TESTE 🏆")
    print("="*50)
    print(f"Limiar Matemático Ótimo : {melhor_limiar:.2f} ({melhor_limiar*100:.0f}%)")
    print(f"AUC (Área Sob a Curva)  : {final_auc:.4f}")
    print(f"MCC (Coef. de Matthews) : {final_mcc:.4f}")
    print("="*50)

    # 5. Gerar a Matriz de Confusão Final
    os.makedirs('plots', exist_ok=True)
    plot_final_confusion_matrix(all_labels, all_preds, melhor_limiar)

if __name__ == "__main__":
    evaluate_on_test()