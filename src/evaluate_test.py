#evaluate_test.py

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    precision_recall_curve, recall_score, precision_score, accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# CORREÇÃO: EnsembleDualViewClassifier é a arquitetura atual (dois backbones + late fusion),
# não DualViewClassifier (versão mais antiga, backbone único compartilhado).
from src.dataset import TwoViewMammogramDataset, get_valid_transforms
from src.models import EnsembleDualViewClassifier

# ================= CONFIGURAÇÕES =================
# ATENÇÃO: confirme que este é o MESMO csv usado no train.py. Um csv diferente pode
# significar splits diferentes dos que o modelo foi treinado/validado, invalidando a comparação.
CSV_PATH = 'breast-level_annotations_final_limpo(2).csv'
CHECKPOINT_PATH = 'checkpoints/best_dual_view_model_modified.pth'

META_SENSIBILIDADE = 0.90  # meta clínica: sensibilidade mínima aceitável no rastreamento
USAR_TTA_NO_TESTE = True   # test-time augmentation (flip horizontal) só na avaliação final


def plot_final_confusion_matrix(y_true, y_pred, limiar, output_dir="plots"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
    plt.title(f'Matriz de Confusão - TESTE FINAL\n(Limiar de Decisão: {limiar:.2f})')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição do Modelo')
    plt.tight_layout()
    caminho = os.path.join(output_dir, 'cm_TESTE_FINAL.png')
    plt.savefig(caminho)
    plt.close()
    print(f"📊 Matriz de confusão guardada em: {caminho}")  # CORREÇÃO: o print original citava um nome de arquivo diferente do salvo


def rodar_inferencia(model, loader, device, usar_tta=False, desc="Inferência"):
    """
    Roda o modelo sobre um DataLoader e retorna (labels, probs) como arrays numpy.
    Reutilizada tanto para a validação (busca de limiar) quanto para o teste (avaliação final).
    """
    all_labels, all_probs = [], []
    with torch.no_grad():
        for img_cc, img_mlo, density, labels in tqdm(loader, desc=desc):
            img_cc, img_mlo, density = img_cc.to(device), img_mlo.to(device), density.to(device)

            # ================= CORREÇÃO: modelo retorna (out_cc, out_mlo); ensemble = média das probs =================
            out_cc, out_mlo = model(img_cc, img_mlo, density)
            probs = (torch.sigmoid(out_cc) + torch.sigmoid(out_mlo)) / 2.0
            # ============================================================================================================

            if usar_tta:
                # Flip horizontal como test-time augmentation. A densidade não é afetada pelo flip.
                # OBS: para mamografia, inverter a lateralidade pode não ser neutro clinicamente —
                # vale confirmar que essa invariância é desejável para o seu caso antes de manter o TTA.
                img_cc_flip = torch.flip(img_cc, dims=[3])
                img_mlo_flip = torch.flip(img_mlo, dims=[3])
                out_cc_f, out_mlo_f = model(img_cc_flip, img_mlo_flip, density)
                probs_flip = (torch.sigmoid(out_cc_f) + torch.sigmoid(out_mlo_f)) / 2.0
                probs = (probs + probs_flip) / 2.0

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def encontrar_limiar_clinico(labels, probs, meta_sensibilidade=META_SENSIBILIDADE):
    """
    Dentre os limiares que atingem a sensibilidade mínima desejada, escolhe o de maior
    precisão (menos falsos positivos). Deve ser chamado SOBRE A VALIDAÇÃO, nunca sobre o teste.
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    indices_validos = np.where(recalls[:-1] >= meta_sensibilidade)[0]

    if len(indices_validos) > 0:
        indice_escolhido = indices_validos[np.argmax(precisions[indices_validos])]
        return float(thresholds[indice_escolhido])
    else:
        print(f"⚠️ Nenhum limiar atingiu {meta_sensibilidade*100:.0f}% de sensibilidade na validação. "
              f"Usando limiar padrão de 0.5.")
        return 0.5


def evaluate_on_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 A iniciar Avaliação Final no dispositivo: {device}")

    df = pd.read_csv(CSV_PATH)
    valid_df = df[df['split'] == 'validation'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    print(f"📁 Exames em Validação: {len(valid_df)} | Exames no Teste Cego: {len(test_df)}")

    valid_dataset = TwoViewMammogramDataset(valid_df, transform=get_valid_transforms())
    test_dataset = TwoViewMammogramDataset(test_df, transform=get_valid_transforms())

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # pretrained_patch_path=None: os backbones serão sobrescritos pelo checkpoint completo abaixo,
    # não faz sentido gastar tempo/memória carregando os pesos do patch classifier antes disso.
    model = EnsembleDualViewClassifier(pretrained_patch_path=None).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ ERRO: Ficheiro {CHECKPOINT_PATH} não encontrado! Certifique-se de que o treino já salvou um checkpoint.")
        return

    print("🧠 A carregar os pesos do modelo campeão...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()  # desliga Dropout etc.

    # ================= CORREÇÃO METODOLÓGICA: limiar escolhido na VALIDAÇÃO, não no teste =================
    # O limiar de decisão precisa ser fixado usando dados que já influenciaram a seleção do
    # checkpoint (validação) — nunca o conjunto de teste cego, sob risco de vazamento de dado
    # (a métrica de teste deixaria de ser uma estimativa honesta de generalização).
    print(f"\n🔍 A calcular o Limiar de Decisão Clínico na VALIDAÇÃO "
          f"(meta: Sensibilidade >= {META_SENSIBILIDADE*100:.0f}%)...")
    labels_valid, probs_valid = rodar_inferencia(model, valid_loader, device, usar_tta=False, desc="Validação (limiar)")
    melhor_limiar = encontrar_limiar_clinico(labels_valid, probs_valid)
    print(f"Limiar fixado a partir da validação: {melhor_limiar:.4f}")
    # ================================================================================================================

    print(f"\n🧪 A avaliar o conjunto de Teste Cego{' (com TTA — flip horizontal)' if USAR_TTA_NO_TESTE else ''}...")
    labels_test, probs_test = rodar_inferencia(model, test_loader, device, usar_tta=USAR_TTA_NO_TESTE, desc="Teste (avaliação final)")

    all_preds = (probs_test >= melhor_limiar).astype(int)

    try:
        final_auc = roc_auc_score(labels_test, probs_test)
        final_mcc = matthews_corrcoef(labels_test, all_preds)
        final_sens = recall_score(labels_test, all_preds, zero_division=0)
        final_prec = precision_score(labels_test, all_preds, zero_division=0)
        final_acc = accuracy_score(labels_test, all_preds)
        final_f1 = f1_score(labels_test, all_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels_test, all_preds).ravel()
        final_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        final_auc, final_mcc, final_sens, final_spec, final_prec, final_acc, final_f1 = (0.0,) * 7

    print("\n" + "=" * 50)
    print("🏆 RESULTADOS OFICIAIS DO CONJUNTO DE TESTE (CENÁRIO CLÍNICO) 🏆")
    print("=" * 50)
    print(f"Limiar Clínico (fixado na validação): {melhor_limiar:.4f}")
    print(f"AUC (Área Sob a Curva)  : {final_auc:.4f}")
    print(f"Sensibilidade (Recall)  : {final_sens:.4f}")
    print(f"Especificidade          : {final_spec:.4f}")
    print(f"Precisão                : {final_prec:.4f}")
    print(f"Acurácia                : {final_acc:.4f}")
    print(f"F1-Score                : {final_f1:.4f}")
    print(f"MCC (Coef. de Matthews) : {final_mcc:.4f}")
    print("=" * 50)

    os.makedirs('plots', exist_ok=True)
    plot_final_confusion_matrix(labels_test, all_preds, melhor_limiar)


if __name__ == "__main__":
    evaluate_on_test()