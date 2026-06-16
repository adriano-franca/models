import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import wandb

from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import LayerGradCam, LayerAttribution, GuidedGradCam

from src.dataset import TwoViewMammogramDataset, get_train_transforms, get_valid_transforms
from src.models import DualViewClassifier

def plot_confusion_matrix(y_true, y_pred, epoch, output_dir="plots"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
    plt.title('Matriz de Confusão - Época {epoch+1}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição do Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_epoch_{epoch+1}.png'))
    plt.close()

def plot_gradcam(model, valid_dataset, device, predicted_label, epoch, idx, output_dir="plots"):
    model.eval()

    img_cc, img_mlo, label = valid_dataset[idx]

    img_cc = img_cc.unsqueeze(0).to(device)
    img_mlo = img_mlo.unsqueeze(0).to(device)

    target_label = label.item() if isinstance(label, torch.Tensor) else label

    layer_alvo = model.backbone.stages[-1]

    guided_gc = GuidedGradCam(model, layer_alvo)

    attr_cc, attr_mlo = guided_gc.attribute((img_cc, img_mlo), target=0)

    heatmap_cc = attr_cc.squeeze().cpu().detach().numpy()
    heatmap_cc = np.abs(heatmap_cc)
    if heatmap_cc.max() > 0:
        heatmap_cc /= heatmap_cc.max()

    heatmap_mlo = attr_mlo.squeeze().cpu().detach().numpy()
    heatmap_mlo = np.abs(heatmap_mlo)
    if heatmap_mlo.max() > 0:
        heatmap_mlo /= heatmap_mlo.max()

    viz_cc = img_cc.squeeze().cpu().numpy()
    viz_mlo = img_mlo.squeeze().cpu().numpy()
    viz_cc = (viz_cc - viz_cc.min()) / (viz_cc.max() - viz_cc.min() + 1e-8)
    viz_mlo = (viz_mlo - viz_mlo.min()) / (viz_mlo.max() - viz_mlo.min() + 1e-8)

    str_real = "Anormal" if target_label == 1 else "Normal"
    str_previsto = "Anormal" if predicted_label == 1 else "Normal"

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Guided Grad-CAM - Época {epoch+1} | Paciente: #{idx}\nRótulo Real: {str_real} | Previsto pela Rede: {str_previsto}', fontsize=16, fontweight='bold')

    axes[0].imshow(viz_cc, cmap='gray')
    axes[0].imshow(heatmap_cc, cmap='magma', alpha=0.5) 
    axes[0].set_title('Vista CC')
    axes[0].axis('off')

    axes[1].imshow(viz_mlo, cmap='gray')
    axes[1].imshow(heatmap_mlo, cmap='magma', alpha=0.5)
    axes[1].set_title('Vista Mediolateral Oblíqua (MLO)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gradcam_epoch_{epoch+1}.png'))
    plt.close()

def train_dual_view_model(csv_path, epochs=15, batch_size=1, accumulation_steps=8, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"A usar o dispositivo: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    wandb.init(
        project="mestrado-visao-mamografia", 
        name=f"DualView-PetriniModified-bs{batch_size}-acc{accumulation_steps}", 
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "accumulation_steps": accumulation_steps,
            "arquitetura": "DualViewClassifier",
            "pos_weight": 8.0
        }
    )

    df = pd.read_csv('breast-level_annotations_grouped_80_10_10(2).csv')
    train_df = df[df['split'] == 'training'].reset_index(drop=True)
    valid_df = df[df['split'] == 'validation'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    train_dataset = TwoViewMammogramDataset(train_df, transform=get_train_transforms())
    valid_dataset = TwoViewMammogramDataset(valid_df, transform=get_valid_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DualViewClassifier(pretrained_patch_path='checkpoints/best_patch_classifier_modified.pth').to(device)

    peso_anormal = torch.tensor([8.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=peso_anormal)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_mcc = -1.0

    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")

        if epoch == 0:
            print("\nCongelando o backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = False
                
        elif epoch == 4:
            print("\nDescongelando o backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
                
            for g in optimizer.param_groups:
                g['lr'] = 1e-5 

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc="Treino")
        for batch_idx, (img_cc, img_mlo, labels) in enumerate(loop):
            img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(img_cc, img_mlo)
            loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            loop.set_postfix(loss=loss.item() * accumulation_steps)

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            loop_val = tqdm(valid_loader, desc="Validação")
            for img_cc, img_mlo, labels in loop_val:
                img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(img_cc, img_mlo)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        avg_valid_loss = valid_loss / len(valid_loader)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Limiar de 0.5 para decidir se é 0 (Normal) ou 1 (Anormal)
        all_preds = (all_probs >= 0.5).astype(int)

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
            val_mcc = matthews_corrcoef(all_labels, all_preds)

            val_acc = accuracy_score(all_labels, all_preds)
            val_prec = precision_score(all_labels, all_preds, zero_division=0)
            val_sens = recall_score(all_labels, all_preds, zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        except ValueError:
            val_auc, val_mcc = 0.0, 0.0 
            val_acc, val_prec, val_sens, val_f1, val_spec = 0.0, 0.0, 0.0, 0.0, 0.0

        print(f"Perda: Treino {avg_train_loss:.4f} | Valid {avg_valid_loss:.4f}")
        print(f"Métricas: AUC={val_auc:.4f} | MCC={val_mcc:.4f} | Acc={val_acc:.4f} | Sens={val_sens:.4f} | Espec={val_spec:.4f} | Prec={val_prec:.4f} | F1={val_f1:.4f}")

        paciente_idx = random.randint(0, len(valid_dataset) - 1)
        rotulo_previsto = int(all_preds[paciente_idx].item())

        rotulo_real = int(all_labels[paciente_idx].item())
        probabilidade = float(all_probs[paciente_idx].item()) * 100

        str_real = "Anormal" if rotulo_real == 1 else "Normal"
        str_previsto = "Anormal" if rotulo_previsto == 1 else "Normal"
        
        # Imprime apenas uma linha limpa no terminal com a informação do paciente escolhido
        print(f"🔍 Grad-CAM (Paciente #{paciente_idx}) -> Real: {str_real} | Previsto: {str_previsto} (Certeza: {probabilidade:.1f}%)")

        cm_path = os.path.join('plots', f'cm_epoch_{epoch+1}.png')
        gc_path = os.path.join('plots', f'gradcam_epoch_{epoch+1}.png')

        plot_confusion_matrix(all_labels, all_preds, epoch)

        # O Captum requer cálculo de gradientes para o Grad-CAM, por isso ligamos temporariamente
        with torch.set_grad_enabled(True):
            plot_gradcam(model, valid_dataset, device, rotulo_previsto, epoch, paciente_idx)

        wandb.log({
            "epoch": epoch + 1,
            "loss/train": avg_train_loss,
            "loss/validation": avg_valid_loss,
            "metrics/auc": val_auc,
            "metrics/mcc": val_mcc,
            "metrics/acuracia": val_acc,
            "metrics/precisao": val_prec,
            "metrics/sensibilidade": val_sens,
            "metrics/especificidade": val_spec,
            "metrics/f1_score": val_f1,
            "metrics/limiar": 0.5,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "graficos/matriz_confusao": wandb.Image(cm_path),
            "graficos/grad_cam": wandb.Image(gc_path)
        })

        # Salvando o melhor modelo com base na MCC
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            model_path = 'checkpoints/best_dual_view_model_modified.pth'
            torch.save(model.state_dict(), model_path)
            print(">>> Novo melhor modelo guardado no disco! <<<")

            wandb.save(model_path)

    # Encerra a sessão do Wandb
    wandb.finish()

if __name__ == "__main__":
    train_dual_view_model('breast-level_annotations_final_limpo(2).csv')