#train.py

import os

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.checkpointing import EarlyStoppingCheckpoint
from src.config import (
    BACKBONE_LR_AFTER_UNFREEZE,
    BACKBONE_UNFREEZE_WARMUP_STEPS,
    CLASSIFIER_LR,
    EARLY_STOPPING_PATIENCE,
    FREEZE_BACKBONE_EPOCHS,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_MIN_LR,
    LR_PLATEAU_PATIENCE,
    MODEL_CHECKPOINT_PATH,
    PATCH_CHECKPOINT_PATH,
    VALID_LOSS_SMOOTHING_WINDOW,
    WEIGHT_DECAY,
)
from src.dataset import TwoViewMammogramDataset, get_train_transforms, get_valid_transforms
from src.engine import train_one_epoch, validate_one_epoch
from src.metrics import compute_validation_metrics
from src.models import EnsembleDualViewClassifier
from src.seeding import seed_everything
from src.visualization import plot_confusion_matrix, plot_gradcam
from src.warmup import BackboneUnfreezeWarmup

import random

seed_everything(42)


def _build_datasets(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'training'].reset_index(drop=True)
    valid_df = df[df['split'] == 'validation'].reset_index(drop=True)

    train_dataset = TwoViewMammogramDataset(train_df, transform=get_train_transforms())
    valid_dataset = TwoViewMammogramDataset(valid_df, transform=get_valid_transforms())
    return train_df, train_dataset, valid_dataset


def _build_model(device):
    if not os.path.exists(PATCH_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint do patch classifier não encontrado em '{PATCH_CHECKPOINT_PATH}'."
        )
    print(f"✅ Checkpoint do patch classifier confirmado em: {PATCH_CHECKPOINT_PATH}")

    model = EnsembleDualViewClassifier(pretrained_patch_path=PATCH_CHECKPOINT_PATH).to(device)

    if FREEZE_BACKBONE_EPOCHS > 0:
        for p in model.backbone_cc.parameters():
            p.requires_grad = False
        for p in model.backbone_mlo.parameters():
            p.requires_grad = False
        print(f"🧊 Backbones congelados por {FREEZE_BACKBONE_EPOCHS} época(s) de warm-up.")

    return model


def _build_optimizer(model):
    return AdamW([
        {'params': model.backbone_cc.parameters(), 'lr': BACKBONE_LR_AFTER_UNFREEZE},
        {'params': model.backbone_mlo.parameters(), 'lr': BACKBONE_LR_AFTER_UNFREEZE},
        {'params': model.classifier_cc.parameters(), 'lr': CLASSIFIER_LR},
        {'params': model.classifier_mlo.parameters(), 'lr': CLASSIFIER_LR}
    ], weight_decay=WEIGHT_DECAY)


def _unfreeze_backbone(model, optimizer, epoch, warmup):
    for p in model.backbone_cc.parameters():
        p.requires_grad = True
    for p in model.backbone_mlo.parameters():
        p.requires_grad = True
    print(f"🔥 Backbones descongelados a partir da época {epoch+1}.")
    warmup.activate(optimizer)


def _generate_epoch_visualizations(model, valid_dataset, device, epoch,
                                    all_labels, all_probs, all_preds,
                                    batches_nao_finitos_valid):
    """
    # ================= NOVA ALTERAÇÃO: PROTEÇÃO CONTRA DESALINHAMENTO DE ÍNDICES =================
    # all_preds/all_labels/all_probs só contêm os batches que NÃO foram pulados. Se algum
    # batch foi ignorado por loss não-finita, os índices desses arrays não correspondem mais
    # 1:1 à ordem original de valid_dataset — indexar por um paciente aleatório baseado em
    # len(valid_dataset) poderia pegar a predição ERRADA (ou, no limite, um IndexError se o
    # array ficou mais curto/vazio). Só geramos esses artefatos visuais em épocas "limpas"
    # (nenhum batch pulado); nas demais, a época segue normalmente — só sem esse preview.
    # ============================================================================================================
    """
    gerar = (len(all_preds) > 0) and (batches_nao_finitos_valid == 0)
    if not gerar:
        print("⚠️ Pulando matriz de confusão/Grad-CAM nesta época "
              "(validação incompleta ou vazia — índices não alinhados com segurança).")
        return None, None

    paciente_idx = random.randint(0, len(valid_dataset) - 1)
    rotulo_previsto = int(all_preds[paciente_idx].item())
    rotulo_real = int(all_labels[paciente_idx].item())
    probabilidade = float(all_probs[paciente_idx].item()) * 100

    str_real = "Anormal" if rotulo_real == 1 else "Normal"
    str_previsto = "Anormal" if rotulo_previsto == 1 else "Normal"
    print(f"🔍 Grad-CAM (Paciente #{paciente_idx}) -> Real: {str_real} | Previsto: {str_previsto} "
          f"(Certeza: {probabilidade:.1f}%)")

    cm_path = os.path.join('plots', f'cm_epoch_{epoch+1}.png')
    gc_path = os.path.join('plots', f'gradcam_epoch_{epoch+1}.png')

    plot_confusion_matrix(all_labels, all_preds, epoch)
    with torch.set_grad_enabled(True):
        plot_gradcam(model, valid_dataset, device, rotulo_previsto, epoch, paciente_idx)

    return cm_path, gc_path


def train_dual_view_model(csv_path, epochs=15, batch_size=2, accumulation_steps=2, lr=1e-4):
    # AJUSTE (OOM ao destravar os backbones): batch_size caiu de 4 -> 2 e
    # accumulation_steps subiu de 4 -> 8, mantendo o MESMO batch efetivo (16).
    # Quando os backbones estão congelados, o PyTorch não precisa guardar as
    # ativações internas deles para o backward (só a saída final, usada pela
    # cabeça de classificação) — por isso o treino cabia tranquilo na VRAM até
    # a época 3. Ao destravar (época 4), passa a ser necessário reter TODAS as
    # ativações intermediárias das duas ConvNeXt-Small inteiras para o backward,
    # o que aumenta bruscamente o consumo de memória. Reduzir o batch físico
    # (mantendo o efetivo via mais acumulação) é a forma mais direta e segura de
    # dar folga a essa memória sem mudar nada do comportamento estatístico do
    # treino. Se ainda estourar memória na época em que destrava, tente
    # batch_size=1, accumulation_steps=16.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"A usar o dispositivo: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    train_df, train_dataset, valid_dataset = _build_datasets(csv_path)

    coluna_rotulo = 'target'
    contagem_classes = train_df[coluna_rotulo].value_counts().to_dict()
    pos_weight_value = 2.0
    print(f"Peso aplicado à classe Anormal (pos_weight calculado): {pos_weight_value:.2f}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    wandb.init(
        project="mestrado-visao-mamografia-dualview",
        name=f"DualView-PetriniModified-bs{batch_size}-acc{accumulation_steps}",
        config={
            "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
            "accumulation_steps": accumulation_steps, "arquitetura": "DualViewClassifier",
            "pos_weight": pos_weight_value, "class_imbalance_strategy": "pos_weight_only_no_sampler",
            "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS, "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "model_selection_criterion": "valid_loss_smoothed", "valid_loss_smoothing_window": VALID_LOSS_SMOOTHING_WINDOW,
            "lr_scheduler": "ReduceLROnPlateau", "lr_plateau_factor": LR_PLATEAU_FACTOR,
            "lr_plateau_patience": LR_PLATEAU_PATIENCE, "lr_plateau_min_lr": LR_PLATEAU_MIN_LR,
            "weight_decay": WEIGHT_DECAY, "backbone_lr_after_unfreeze": BACKBONE_LR_AFTER_UNFREEZE,
            "patch_checkpoint_path": PATCH_CHECKPOINT_PATH
        }
    )

    model = _build_model(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value]).to(device))
    optimizer = _build_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_PLATEAU_FACTOR,
                                   patience=LR_PLATEAU_PATIENCE, min_lr=LR_PLATEAU_MIN_LR)
    scaler = torch.amp.GradScaler('cuda')
    warmup = BackboneUnfreezeWarmup(
        total_steps=BACKBONE_UNFREEZE_WARMUP_STEPS,
        backbone_lr_target=BACKBONE_LR_AFTER_UNFREEZE,
        classifier_lr_target=CLASSIFIER_LR,
    )
    ckpt = EarlyStoppingCheckpoint(model_path=MODEL_CHECKPOINT_PATH, patience=EARLY_STOPPING_PATIENCE)

    valid_loss_history = []

    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")

        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            _unfreeze_backbone(model, optimizer, epoch, warmup)
            ckpt.reset_patience()
            valid_loss_history = []
            print("   ↳ Paciência do early stopping e histórico de suavização reiniciados "
                  "(dando fôlego justo ao fine-tuning completo).")

        avg_train_loss, batches_nao_finitos_treino = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            epoch, FREEZE_BACKBONE_EPOCHS, accumulation_steps, warmup
        )

        avg_valid_loss, all_labels, all_probs, batches_nao_finitos_valid = validate_one_epoch(
            model, valid_loader, criterion, device, epoch
        )

        m = compute_validation_metrics(all_labels, all_probs)
        all_preds = m["all_preds"]

        valid_loss_history.append(avg_valid_loss)
        janela = valid_loss_history[-VALID_LOSS_SMOOTHING_WINDOW:]
        valid_loss_suavizada = sum(janela) / len(janela)

        scheduler.step(valid_loss_suavizada)
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[2]['lr']

        print(f"Perda: Treino {avg_train_loss:.4f} | Valid {avg_valid_loss:.4f} | Valid (suavizada) {valid_loss_suavizada:.4f}")
        print(f"LR Atual -> Backbones: {current_lr_backbone:.2e} | Classificadores: {current_lr_classifier:.2e}")
        print(f"Métricas: AUC={m['auc']:.4f} | MCC={m['mcc']:.4f} | Acc={m['acc']:.4f} | "
              f"Sens={m['sens']:.4f} | Espec={m['spec']:.4f} | Prec={m['prec']:.4f} | F1={m['f1']:.4f}")

        cm_path, gc_path = _generate_epoch_visualizations(
            model, valid_dataset, device, epoch,
            all_labels, all_probs, all_preds, batches_nao_finitos_valid
        )

        log_dict = {
            "epoch": epoch + 1,
            "loss/train": avg_train_loss,
            "loss/validation": avg_valid_loss,
            "loss/validation_smoothed": valid_loss_suavizada,
            "metrics/auc": m['auc'],
            "metrics/mcc": m['mcc'],
            "metrics/acuracia": m['acc'],
            "metrics/precisao": m['prec'],
            "metrics/sensibilidade": m['sens'],
            "metrics/especificidade": m['spec'],
            "metrics/f1_score": m['f1'],
            "metrics/limiar": m['limiar'],
            "learning_rate/backbone": current_lr_backbone,
            "learning_rate/classifier": current_lr_classifier,
            "backbone_frozen": epoch < FREEZE_BACKBONE_EPOCHS,
            "batches_nao_finitos/treino": batches_nao_finitos_treino,
            "batches_nao_finitos/validacao": batches_nao_finitos_valid,
        }
        if cm_path is not None:
            log_dict["graficos/matriz_confusao"] = wandb.Image(cm_path)
            log_dict["graficos/grad_cam"] = wandb.Image(gc_path)

        wandb.log(log_dict)

        deve_parar = ckpt.step(valid_loss_suavizada, m['mcc'], model)
        if deve_parar:
            print(f"(interrompido na época {epoch+1})")
            break

    wandb.summary["best_valid_loss_smoothed"] = ckpt.melhor_valid_loss
    wandb.summary["best_valid_mcc_at_checkpoint"] = ckpt.melhor_mcc_no_ponto_salvo
    wandb.finish()


if __name__ == "__main__":
    train_dual_view_model('breast-level_annotations_final_limpo(2).csv')