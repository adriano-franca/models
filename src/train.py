#train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import wandb

from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import GuidedGradCam

from src.dataset import TwoViewMammogramDataset, get_train_transforms, get_valid_transforms
from src.models import EnsembleDualViewClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

FREEZE_BACKBONE_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 4
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 2
LR_PLATEAU_MIN_LR = 1e-7
VALID_LOSS_SMOOTHING_WINDOW = 3
WEIGHT_DECAY = 5e-2
BACKBONE_LR_AFTER_UNFREEZE = 5e-6
PATCH_CHECKPOINT_PATH = 'checkpoints/patch_classifier_convnext_density_clahe.pth'

# ================= NOVA ALTERAÇÃO: WARM-UP GRADUAL DE LR NO DESTRAVAMENTO DO BACKBONE =================
# Enquanto o backbone fica congelado, ele nunca recebe gradiente, então os buffers internos
# do AdamW para esses parâmetros (exp_avg, exp_avg_sq — as médias móveis que adaptam o LR por
# parâmetro) nunca são inicializados. No instante em que o backbone destrava, esses parâmetros
# começam do zero absoluto nos primeiros passos do otimizador, o que pode gerar atualizações
# mal calibradas antes das estatísticas internas se estabilizarem. Em vez de saltar direto para
# BACKBONE_LR_AFTER_UNFREEZE, o LR do backbone sobe LINEARMENTE de ~0 até esse valor ao longo
# dos primeiros passos de otimizador após o destravamento — atenuando qualquer "choque" inicial,
# independente do mecanismo exato por trás dele.
BACKBONE_UNFREEZE_WARMUP_STEPS = 100
# ============================================================================================================

def plot_confusion_matrix(y_true, y_pred, epoch, output_dir="plots"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
    plt.title(f'Matriz de Confusão - Época {epoch+1}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição do Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_epoch_{epoch+1}.png'))
    plt.close()

def find_best_threshold(labels, probs):
    best_mcc = -1
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.91, 0.01):
        preds = (np.array(probs) > thresh).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh
    return best_thresh, best_mcc

def plot_gradcam(model, valid_dataset, device, predicted_label, epoch, idx, output_dir="plots"):
    model.eval()
    img_cc, img_mlo, density, label = valid_dataset[idx]

    img_cc = img_cc.unsqueeze(0).to(device).requires_grad_(True)
    img_mlo = img_mlo.unsqueeze(0).to(device).requires_grad_(True)
    density = density.unsqueeze(0).to(device)

    target_label = label.item() if isinstance(label, torch.Tensor) else label

    class UnifiedViewWrapper(nn.Module):
        def __init__(self, model, view_type):
            super().__init__()
            self.model = model
            self.view_type = view_type

        def forward(self, x):
            if self.view_type == 'cc':
                return self.model(x, img_mlo, density)
            else:
                return self.model(img_cc, x, density)

    model_cc = UnifiedViewWrapper(model, 'cc')
    model_mlo = UnifiedViewWrapper(model, 'mlo')

    layer_alvo_cc = model.backbone_cc.stages[-1]
    layer_alvo_mlo = model.backbone_mlo.stages[-1]

    guided_gc_cc = GuidedGradCam(model_cc, layer_alvo_cc)
    guided_gc_mlo = GuidedGradCam(model_mlo, layer_alvo_mlo)

    attr_cc = guided_gc_cc.attribute(img_cc, target=0)
    attr_mlo = guided_gc_mlo.attribute(img_mlo, target=0)

    heatmap_cc = attr_cc.squeeze().cpu().detach().numpy()
    heatmap_cc = np.abs(heatmap_cc)
    if heatmap_cc.max() > 0: heatmap_cc /= heatmap_cc.max()

    heatmap_mlo = attr_mlo.squeeze().cpu().detach().numpy()
    heatmap_mlo = np.abs(heatmap_mlo)
    if heatmap_mlo.max() > 0: heatmap_mlo /= heatmap_mlo.max()

    viz_cc = img_cc.squeeze().cpu().detach().numpy()
    viz_mlo = img_mlo.squeeze().cpu().detach().numpy()
    viz_cc = (viz_cc - viz_cc.min()) / (viz_cc.max() - viz_cc.min() + 1e-8)
    viz_mlo = (viz_mlo - viz_mlo.min()) / (viz_mlo.max() - viz_mlo.min() + 1e-8)

    str_real = "Anormal" if target_label == 1 else "Normal"
    str_previsto = "Anormal" if predicted_label == 1 else "Normal"

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Guided Grad-CAM (Ensemble) - Época {epoch+1} | Paciente: #{idx}\nRótulo Real: {str_real} | Previsto: {str_previsto}', fontsize=16, fontweight='bold')

    axes[0].imshow(viz_cc, cmap='gray')
    axes[0].imshow(heatmap_cc, cmap='magma', alpha=0.5)
    axes[0].set_title('Vista CC (Rede CC)')
    axes[0].axis('off')

    axes[1].imshow(viz_mlo, cmap='gray')
    axes[1].imshow(heatmap_mlo, cmap='magma', alpha=0.5)
    axes[1].set_title('Vista MLO (Rede MLO)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gradcam_epoch_{epoch+1}.png'))
    plt.close()

def train_dual_view_model(csv_path, epochs=15, batch_size=2, accumulation_steps=8, lr=1e-4):
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

    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'training'].reset_index(drop=True)
    valid_df = df[df['split'] == 'validation'].reset_index(drop=True)

    train_dataset = TwoViewMammogramDataset(train_df, transform=get_train_transforms())
    valid_dataset = TwoViewMammogramDataset(valid_df, transform=get_valid_transforms())

    coluna_rotulo = 'target'
    contagem_classes = train_df[coluna_rotulo].value_counts().to_dict()
    pos_weight_value = contagem_classes[0] / (contagem_classes[1] + 1e-8)
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

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value]).to(device))

    optimizer = AdamW([
        {'params': model.backbone_cc.parameters(), 'lr': BACKBONE_LR_AFTER_UNFREEZE},
        {'params': model.backbone_mlo.parameters(), 'lr': BACKBONE_LR_AFTER_UNFREEZE},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE, min_lr=LR_PLATEAU_MIN_LR)
    scaler = torch.amp.GradScaler('cuda')

    melhor_valid_loss = float('inf')
    melhor_mcc_no_ponto_salvo = -1.0
    best_mcc = -1.0
    epocas_sem_melhora = 0
    valid_loss_history = []

    # NOVA ALTERAÇÃO: contador de passos do warm-up gradual de LR do backbone.
    # None = warm-up inativo (ainda não começou ou já terminou).
    passos_desde_destravamento = None

    model_path = 'checkpoints/best_dual_view_model_modified.pth'
    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")

        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            for p in model.backbone_cc.parameters():
                p.requires_grad = True
            for p in model.backbone_mlo.parameters():
                p.requires_grad = True
            print(f"🔥 Backbones descongelados a partir da época {epoch+1}.")

            # ================= NOVA ALTERAÇÃO: FOLGA PARA O EARLY STOPPING PÓS-DESTRAVAMENTO =================
            # Destravar os backbones é uma perturbação grande no treino (o regime de perda muda
            # de patamar). Sem isso, o "solavanco" natural dessa transição ficava preso na janela
            # de suavização por várias épocas, consumindo a paciência do early stopping bem no
            # momento em que o fine-tuning de verdade estava começando — como visto no log em que
            # o treino parou na época 5 com o checkpoint da época 1 (antes do backbone sequer
            # se mover). Zeramos o contador de paciência (dando um fôlego justo pro modelo provar
            # que o fine-tuning completo ajuda) e limpamos o histórico da média móvel (para a
            # suavização não misturar perdas de regimes diferentes: congelado vs. destravado).
            # IMPORTANTE: melhor_valid_loss NÃO é resetado — o modelo só é salvo se realmente
            # superar o melhor resultado histórico, então essa mudança não afrouxa o critério de
            # qualidade do checkpoint, só dá tempo justo para tentar alcançá-lo/superá-lo.
            epocas_sem_melhora = 0
            valid_loss_history = []
            print("   ↳ Paciência do early stopping e histórico de suavização reiniciados "
                  "(dando fôlego justo ao fine-tuning completo).")
            # ============================================================================================================

            # ================= NOVA ALTERAÇÃO: ATIVA O WARM-UP GRADUAL DE LR DO BACKBONE =================
            passos_desde_destravamento = 0
            # Começa o LR do backbone bem baixo (será rampeado linearmente nos próximos
            # BACKBONE_UNFREEZE_WARMUP_STEPS passos de otimizador, dentro do loop de treino abaixo).
            optimizer.param_groups[0]['lr'] = BACKBONE_LR_AFTER_UNFREEZE / BACKBONE_UNFREEZE_WARMUP_STEPS
            optimizer.param_groups[1]['lr'] = BACKBONE_LR_AFTER_UNFREEZE / BACKBONE_UNFREEZE_WARMUP_STEPS
            print(f"   ↳ Warm-up gradual de LR do backbone ativado "
                  f"({BACKBONE_UNFREEZE_WARMUP_STEPS} passos até {BACKBONE_LR_AFTER_UNFREEZE:.2e}).")
            # ============================================================================================================

        model.train()
        train_loss = 0.0
        train_batches_validos = 0  # NOVA ALTERAÇÃO: conta só os batches que contribuíram de fato
        batches_nao_finitos_treino = 0
        optimizer.zero_grad(set_to_none=True)

        loop = tqdm(train_loader, desc="Treino")
        for batch_idx, (img_cc, img_mlo, density, labels) in enumerate(loop):
            img_cc, img_mlo, density = img_cc.to(device), img_mlo.to(device), density.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.amp.autocast('cuda'):
                out = model(img_cc, img_mlo, density)
                loss = criterion(out, labels)

            # ================= NOVA ALTERAÇÃO: PROTEÇÃO CONTRA LOSS NÃO-FINITA (NaN/Inf) =================
            # A perda de validação também virava NaN mesmo sem nenhum update de peso acontecer ali
            # (roda inteiramente sob torch.no_grad()) — sinal de que um único batch com forward
            # não-finito (NaN/Inf) já é suficiente para contaminar a soma acumulada da época inteira
            # (NaN + qualquer_coisa = NaN). Em vez de deixar isso se propagar silenciosamente, pulamos
            # a contribuição desse batch específico (sem backward, sem contar na média) e registramos
            # um aviso — o que também nos dá visibilidade de QUANTAS vezes isso ocorre.
            if not torch.isfinite(loss):
                batches_nao_finitos_treino += 1
                print(f"⚠️ Loss não-finita no batch de treino {batch_idx} (época {epoch+1}). "
                      f"Batch ignorado (sem backward/update).")
                optimizer.zero_grad(set_to_none=True)
                continue
            # ====================================================================================================

            loss_para_backward = loss / accumulation_steps
            scaler.scale(loss_para_backward).backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # ================= NOVA ALTERAÇÃO: WARM-UP GRADUAL DE LR + DIAGNÓSTICO DE NORMA =================
                if passos_desde_destravamento is not None:
                    if passos_desde_destravamento < BACKBONE_UNFREEZE_WARMUP_STEPS:
                        # Diagnóstico: norma do gradiente ANTES do clip, para sabermos se estamos
                        # de fato vendo explosão de gradiente nos primeiros passos pós-destravamento.
                        print(f"   [warm-up backbone] passo {passos_desde_destravamento+1}/"
                              f"{BACKBONE_UNFREEZE_WARMUP_STEPS} | norma do gradiente (pré-clip) = "
                              f"{grad_norm.item():.4f}")

                        progresso = (passos_desde_destravamento + 1) / BACKBONE_UNFREEZE_WARMUP_STEPS
                        novo_lr_backbone = BACKBONE_LR_AFTER_UNFREEZE * progresso
                        optimizer.param_groups[0]['lr'] = novo_lr_backbone
                        optimizer.param_groups[1]['lr'] = novo_lr_backbone
                        passos_desde_destravamento += 1
                    else:
                        # Warm-up concluído: LR já está no valor-alvo; entrega o controle de volta
                        # para o ReduceLROnPlateau normalmente a partir daqui.
                        passos_desde_destravamento = None
                        print(f"   [warm-up backbone] concluído — LR agora em {BACKBONE_LR_AFTER_UNFREEZE:.2e}.")
                # ============================================================================================================

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            train_batches_validos += 1
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / max(train_batches_validos, 1)
        if batches_nao_finitos_treino > 0:
            print(f"⚠️ Total de batches de treino ignorados por loss não-finita nesta época: {batches_nao_finitos_treino}")

        model.eval()
        valid_loss = 0.0
        valid_batches_validos = 0  # NOVA ALTERAÇÃO: idem, para a validação
        batches_nao_finitos_valid = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            loop_val = tqdm(valid_loader, desc="Validação")
            for img_cc, img_mlo, density, labels in loop_val:
                img_cc, img_mlo, density = img_cc.to(device), img_mlo.to(device), density.to(device)
                labels = labels.to(device).unsqueeze(1)

                with torch.amp.autocast('cuda'):
                    out = model(img_cc, img_mlo, density)
                    loss = criterion(out, labels)
                    prob_ensemble = torch.sigmoid(out)

                # ================= NOVA ALTERAÇÃO: mesma proteção na validação =================
                if not torch.isfinite(loss):
                    batches_nao_finitos_valid += 1
                    print(f"⚠️ Loss não-finita num batch de validação (época {epoch+1}). Batch ignorado.")
                    continue
                # ========================================================================================

                valid_loss += loss.item()
                valid_batches_validos += 1
                probs = prob_ensemble.cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        # ================= NOVA ALTERAÇÃO: PROTEÇÃO CONTRA "MELHOR ÉPOCA FALSA" =================
        # Se TODOS os batches de validação forem não-finitos, valid_loss e valid_batches_validos
        # ficam ambos em 0, e 0.0/max(0,1) = 0.0 -- o que pareceria a MELHOR perda possível e faria
        # o checkpoint (quebrado) ser salvo como "novo melhor modelo". Em vez disso, tratamos uma
        # validação inteiramente não-finita como o PIOR resultado possível (infinito), garantindo
        # que essa época nunca seja escolhida como checkpoint nem reinicie o contador de paciência.
        if valid_batches_validos == 0:
            avg_valid_loss = float('inf')
            print(f"🛑 TODA a validação desta época foi não-finita ({batches_nao_finitos_valid} "
                  f"batches). Tratando como pior resultado possível (não será salva como melhor).")
        else:
            avg_valid_loss = valid_loss / valid_batches_validos
        # ============================================================================================================
        if batches_nao_finitos_valid > 0:
            print(f"⚠️ Total de batches de validação ignorados por loss não-finita nesta época: {batches_nao_finitos_valid}")

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # ================= NOVA ALTERAÇÃO: PROTEÇÃO CONTRA ARRAYS VAZIOS =================
        # Se toda a validação foi não-finita (all_labels/all_probs vazios), find_best_threshold()
        # e as métricas do sklearn quebrariam com ValueError ("Found empty array"). Em vez de
        # deixar o treino inteiro morrer por causa de uma única época ruim, registramos métricas
        # degeneradas (zeradas) e deixamos o early stopping/checkpoint (já protegidos acima)
        # lidarem normalmente com essa época — o treino continua para a próxima.
        if len(all_labels) == 0:
            melhor_limiar_epoca = 0.5
            val_auc, val_mcc = 0.0, 0.0
            val_acc, val_prec, val_sens, val_f1, val_spec = 0.0, 0.0, 0.0, 0.0, 0.0
            all_preds = np.array([])
        else:
            melhor_limiar_epoca, _ = find_best_threshold(all_labels, all_probs)
            all_preds = (all_probs >= melhor_limiar_epoca).astype(int)

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
        # ============================================================================================================

        valid_loss_history.append(avg_valid_loss)
        janela = valid_loss_history[-VALID_LOSS_SMOOTHING_WINDOW:]
        valid_loss_suavizada = sum(janela) / len(janela)

        scheduler.step(valid_loss_suavizada)
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[2]['lr']

        print(f"Perda: Treino {avg_train_loss:.4f} | Valid {avg_valid_loss:.4f} | Valid (suavizada) {valid_loss_suavizada:.4f}")
        print(f"LR Atual -> Backbones: {current_lr_backbone:.2e} | Classificadores: {current_lr_classifier:.2e}")
        print(f"Métricas: AUC={val_auc:.4f} | MCC={val_mcc:.4f} | Acc={val_acc:.4f} | Sens={val_sens:.4f} | Espec={val_spec:.4f} | Prec={val_prec:.4f} | F1={val_f1:.4f}")

        # ================= NOVA ALTERAÇÃO: PROTEÇÃO CONTRA DESALINHAMENTO DE ÍNDICES =================
        # all_preds/all_labels/all_probs só contêm os batches que NÃO foram pulados. Se algum
        # batch foi ignorado por loss não-finita, os índices desses arrays não correspondem mais
        # 1:1 à ordem original de valid_dataset — indexar por um paciente aleatório baseado em
        # len(valid_dataset) poderia pegar a predição ERRADA (ou, no limite, um IndexError se o
        # array ficou mais curto/vazio). Só geramos esses artefatos visuais em épocas "limpas"
        # (nenhum batch pulado); nas demais, a época segue normalmente — só sem esse preview.
        gerar_visualizacoes = (len(all_preds) > 0) and (batches_nao_finitos_valid == 0)
        cm_path, gc_path = None, None

        if gerar_visualizacoes:
            paciente_idx = random.randint(0, len(valid_dataset) - 1)
            rotulo_previsto = int(all_preds[paciente_idx].item())
            rotulo_real = int(all_labels[paciente_idx].item())
            probabilidade = float(all_probs[paciente_idx].item()) * 100

            str_real = "Anormal" if rotulo_real == 1 else "Normal"
            str_previsto = "Anormal" if rotulo_previsto == 1 else "Normal"

            print(f"🔍 Grad-CAM (Paciente #{paciente_idx}) -> Real: {str_real} | Previsto: {str_previsto} (Certeza: {probabilidade:.1f}%)")

            cm_path = os.path.join('plots', f'cm_epoch_{epoch+1}.png')
            gc_path = os.path.join('plots', f'gradcam_epoch_{epoch+1}.png')

            plot_confusion_matrix(all_labels, all_preds, epoch)

            with torch.set_grad_enabled(True):
                plot_gradcam(model, valid_dataset, device, rotulo_previsto, epoch, paciente_idx)
        else:
            print("⚠️ Pulando matriz de confusão/Grad-CAM nesta época "
                  "(validação incompleta ou vazia — índices não alinhados com segurança).")
        # ============================================================================================================

        log_dict = {
            "epoch": epoch + 1,
            "loss/train": avg_train_loss,
            "loss/validation": avg_valid_loss,
            "loss/validation_smoothed": valid_loss_suavizada,
            "metrics/auc": val_auc,
            "metrics/mcc": val_mcc,
            "metrics/acuracia": val_acc,
            "metrics/precisao": val_prec,
            "metrics/sensibilidade": val_sens,
            "metrics/especificidade": val_spec,
            "metrics/f1_score": val_f1,
            "metrics/limiar": melhor_limiar_epoca,
            "learning_rate/backbone": current_lr_backbone,
            "learning_rate/classifier": current_lr_classifier,
            "backbone_frozen": epoch < FREEZE_BACKBONE_EPOCHS,
            "batches_nao_finitos/treino": batches_nao_finitos_treino,
            "batches_nao_finitos/validacao": batches_nao_finitos_valid,
        }
        if gerar_visualizacoes:
            log_dict["graficos/matriz_confusao"] = wandb.Image(cm_path)
            log_dict["graficos/grad_cam"] = wandb.Image(gc_path)

        wandb.log(log_dict)

        if valid_loss_suavizada < melhor_valid_loss:
            melhor_valid_loss = valid_loss_suavizada
            melhor_mcc_no_ponto_salvo = val_mcc
            best_mcc = val_mcc
            torch.save(model.state_dict(), model_path)
            epocas_sem_melhora = 0
            print(f"✅ Novo melhor modelo guardado! (Valid Loss suavizada: {valid_loss_suavizada:.4f} | MCC: {val_mcc:.4f})")
            wandb.save(model_path)
        else:
            epocas_sem_melhora += 1
            print(f"⏳ Sem melhora na perda de validação há {epocas_sem_melhora} época(s). (Melhor: {melhor_valid_loss:.4f})")

        if epocas_sem_melhora >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping ativado na época {epoch+1} (sem melhora por {EARLY_STOPPING_PATIENCE} épocas seguidas).")
            break

    wandb.summary["best_valid_loss_smoothed"] = melhor_valid_loss
    wandb.summary["best_valid_mcc_at_checkpoint"] = melhor_mcc_no_ponto_salvo
    wandb.finish()

if __name__ == "__main__":
    train_dual_view_model('breast-level_annotations_final_limpo(2).csv')