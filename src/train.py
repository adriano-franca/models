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

# ================= REPRODUTIBILIDADE =================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ================= ALTERAÇÃO (d): WARM-UP COM BACKBONE CONGELADO =================
# Mesma lógica do train_patches.py: nas primeiras épocas só os classificadores
# (cabeças CC e MLO) treinam, com os dois backbones congelados.
# AJUSTE: subiu de 2 -> 3 épocas. No log anterior, o overfitting explodia logo na
# época em que o backbone era destravado (perda de treino caindo para ~1e-6/1e-8
# poucas épocas depois) — um warm-up um pouco mais longo dá tempo das cabeças
# convergirem antes de liberar o fine-tuning completo.
FREEZE_BACKBONE_EPOCHS = 3

# ================= ALTERAÇÃO (a): EARLY STOPPING =================
# Nº de épocas sem melhora na perda de validação (suavizada) antes de parar o treino.
EARLY_STOPPING_PATIENCE = 4

# ================= NOVA ALTERAÇÃO: ReduceLROnPlateau =================
# Reduz o LR quando a perda de validação (suavizada) estagna, em vez de um
# cronograma fixo que ignora o comportamento real do treino.
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 2
LR_PLATEAU_MIN_LR = 1e-7

# ================= NOVA ALTERAÇÃO: suavização da perda de validação =================
# Média móvel das últimas N épocas, para evitar que o early stopping/checkpoint
# reajam a um pico isolado de ruído.
VALID_LOSS_SMOOTHING_WINDOW = 3

# ================= CORREÇÃO: WEIGHT_DECAY MAIS FORTE (redução de overfitting) =================
# No log anterior a perda de treino colapsava para ~1e-6/1e-8 (memorização quase
# completa de amostras individuais) enquanto a perda de validação ficava em 0.3-0.6.
# Subir o weight_decay é uma das alavancas mais diretas para conter isso.
WEIGHT_DECAY = 5e-2  # antes: 1e-2

# ================= CORREÇÃO: LR do backbone reduzido após destravar (overfitting) =================
# O salto de perda de treino para quase zero começa exatamente quando os backbones
# são destravados. Baixar o LR do backbone retarda esse processo e dá mais chance
# do ReduceLROnPlateau intervir antes da memorização severa.
BACKBONE_LR_AFTER_UNFREEZE = 5e-6  # antes: 1e-5

# ================= CORREÇÃO: CAMINHO DO CHECKPOINT DO PATCH CLASSIFIER =================
# O train_patches.py atual salva em 'patch_classifier_convnext_density_clahe.pth'
# (com CLAHE). O caminho antigo aqui ('patch_classifier_convnext_density.pth', sem
# '_clahe') provavelmente aponta para um checkpoint desatualizado de uma run anterior,
# sem as melhorias de CLAHE/regularização feitas depois. Ajustado para apontar para
# o checkpoint mais recente. Se o arquivo abaixo não existir no seu disco, o script
# para com um erro claro em vez de falhar silenciosamente ou carregar o modelo errado.
PATCH_CHECKPOINT_PATH = 'checkpoints/patch_classifier_convnext_density_clahe.pth'

# ================= ALTERAÇÃO (c): NOTA SOBRE REGULARIZAÇÃO =================
# O dropout do classificador (0.2 -> 0.35) foi aumentado em train_patches.py dentro
# do próprio modelo (PatchClassifierWithDensity). O EnsembleDualViewClassifier está
# definido em src/models.py, que não foi fornecido aqui — se ele tiver um parâmetro
# de dropout equivalente nas cabeças classifier_cc/classifier_mlo, ajuste-o também
# para 0.35 para manter a mesma regularização entre os dois scripts.
# =================================================================================

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

    # CORREÇÃO: Desempacotar a densidade (4 variáveis)
    img_cc, img_mlo, density, label = valid_dataset[idx]

    img_cc = img_cc.unsqueeze(0).to(device)
    img_cc.requires_grad = True

    img_mlo = img_mlo.unsqueeze(0).to(device)
    img_mlo.requires_grad = True
    
    # Prepara a densidade para o wrapper
    density = density.unsqueeze(0).to(device)

    target_label = label.item() if isinstance(label, torch.Tensor) else label

    # --- WRAPPERS PARA O GRAD-CAM (CORRIGIDO) ---
    class SingleViewWrapper(nn.Module):
        def __init__(self, backbone, avg_pool, max_pool, flatten, classifier, density_tensor):
            super().__init__()
            self.backbone = backbone
            self.avg_pool = avg_pool
            self.max_pool = max_pool
            self.flatten = flatten
            self.classifier = classifier
            self.density = density_tensor
            
        def forward(self, x):
            feat = self.backbone.forward_features(x)
            avg = self.flatten(self.avg_pool(feat))
            max_x = self.flatten(self.max_pool(feat))
            # Concatena exatamente como no EnsembleDualViewClassifier
            pool = torch.cat([avg, max_x, self.density], dim=1) 
            return self.classifier(pool)

    # Inicializa os wrappers com a densidade do paciente atual
    model_cc = SingleViewWrapper(model.backbone_cc, model.global_avg_pool, model.global_max_pool, model.flatten, model.classifier_cc, density)
    model_mlo = SingleViewWrapper(model.backbone_mlo, model.global_avg_pool, model.global_max_pool, model.flatten, model.classifier_mlo, density)

    layer_alvo_cc = model_cc.backbone.stages[-1]
    layer_alvo_mlo = model_mlo.backbone.stages[-1]

    guided_gc_cc = GuidedGradCam(model_cc, layer_alvo_cc)
    guided_gc_mlo = GuidedGradCam(model_mlo, layer_alvo_mlo)

    attr_cc = guided_gc_cc.attribute(img_cc, target=0)
    attr_mlo = guided_gc_mlo.attribute(img_mlo, target=0)

    # O resto do código do heatmap mantém-se igual
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

# ================= NOVA ALTERAÇÃO: LOSS SOBRE A PROBABILIDADE DO ENSEMBLE =================
# Antes, o treino otimizava só loss_cc e loss_mlo — cada cabeça sendo cobrada como se
# fosse sozinha responsável pelo diagnóstico. Mas a métrica que de fato importa é a
# probabilidade JÁ COMBINADA (média de sigmoid(out_cc) e sigmoid(out_mlo)), calculada
# antes apenas na avaliação. Isso é um "surrogate loss mismatch": o gradiente nunca
# "sentia" o efeito da fusão. Esta função calcula uma BCE ponderada (mesma semântica de
# pos_weight do BCEWithLogitsLoss) diretamente sobre a probabilidade já combinada.
#
# nn.BCELoss não aceita o argumento pos_weight (só o BCEWithLogitsLoss aceita, e esse
# exige logits, não probabilidades) — por isso a fórmula é implementada manualmente aqui.
def weighted_bce_from_probs(probs, labels, pos_weight, eps=1e-7):
    probs = probs.clamp(eps, 1.0 - eps)
    loss = -(pos_weight * labels * torch.log(probs) + (1.0 - labels) * torch.log(1.0 - probs))
    return loss.mean()

# Peso do termo de loss do ensemble em relação à média das losses por vista.
# loss_total = (loss_cc + loss_mlo) / 2  +  ENSEMBLE_LOSS_WEIGHT * loss_ensemble
ENSEMBLE_LOSS_WEIGHT = 1.0
# ================================================================================================


def train_dual_view_model(csv_path, epochs=15, batch_size=4, accumulation_steps=4, lr=1e-4):
    # AJUSTE (GPU: RTX 4060 Ti, 16GB VRAM): batch_size subiu de 1 -> 4 e
    # accumulation_steps caiu de 16 -> 4, mantendo o MESMO batch efetivo (16),
    # mas com gradientes calculados sobre 4 amostras por vez em vez de 1.
    # Isso reduz o ruído por passo e deve ajudar a conter o overfitting severo
    # observado no log (perda de treino caindo a ~1e-8 com batch_size=1).
    # Ponto de partida conservador para 16GB com duas ConvNeXt-small (384x384,
    # 1 canal) + AMP. Se dentro dos primeiros passos a GPU estourar memória
    # (CUDA out of memory), reduza para batch_size=2, accumulation_steps=8.
    # Se sobrar VRAM (acompanhe com `nvidia-smi` ou `watch -n1 nvidia-smi`
    # durante o treino), pode tentar batch_size=8, accumulation_steps=2.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"A usar o dispositivo: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Corrigido o uso da variável csv_path
    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'training'].reset_index(drop=True)
    valid_df = df[df['split'] == 'validation'].reset_index(drop=True)

    train_dataset = TwoViewMammogramDataset(train_df, transform=get_train_transforms())
    valid_dataset = TwoViewMammogramDataset(valid_df, transform=get_valid_transforms())

    coluna_rotulo = 'target'

    contagem_classes = train_df[coluna_rotulo].value_counts().to_dict()

    # ================= CORREÇÃO: pos_weight REALMENTE CALCULADO E APLICADO =================
    # Antes, o wandb.config dizia "pos_weight": 20.0 mas o BCEWithLogitsLoss era
    # instanciado sem nenhum pos_weight — o valor nunca chegava na loss de fato.
    # Calculado aqui do mesmo jeito que no train_patches.py: proporção neg/pos.
    pos_weight_value = contagem_classes[0] / (contagem_classes[1] + 1e-8)
    print(f"Peso aplicado à classe Anormal (pos_weight calculado): {pos_weight_value:.2f}")
    # ==========================================================================================

    # ================= CORREÇÃO: REMOVIDO O WeightedRandomSampler (dupla compensação) =================
    # Antes, o WeightedRandomSampler (com replacement=True) já rebalanceava a frequência das
    # classes nos batches, sobrepondo-se ao pos_weight agora aplicado corretamente na loss —
    # os dois mecanismos juntos super-corrigiam o desbalanço (visível no log: Sens=0,90/Spec=0,40
    # logo na 1ª época). Além disso, como exames "Anormal" são raros, o sampler com reposição
    # praticamente garantia que os MESMOS poucos exames positivos aparecessem repetidas vezes
    # dentro de uma única época — o que facilita memorização (overfitting) especificamente
    # nesses exemplos. Mantido apenas o pos_weight (mecanismo único de rebalanceamento): cada
    # exame é visto no máximo uma vez por época (shuffle normal), mas o erro na classe minoritária
    # continua sendo penalizado proporcionalmente mais na função de perda.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # ======================================================================================================

    wandb.init(
        project="mestrado-visao-mamografia-dualview", # Alterado para distinguir do modelo de patches
        name=f"DualView-PetriniModified-bs{batch_size}-acc{accumulation_steps}",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "accumulation_steps": accumulation_steps,
            "arquitetura": "DualViewClassifier",
            "pos_weight": pos_weight_value,
            "class_imbalance_strategy": "pos_weight_only_no_sampler",
            "ensemble_loss_weight": ENSEMBLE_LOSS_WEIGHT,
            "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "model_selection_criterion": "valid_loss_smoothed",
            "valid_loss_smoothing_window": VALID_LOSS_SMOOTHING_WINDOW,
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_plateau_factor": LR_PLATEAU_FACTOR,
            "lr_plateau_patience": LR_PLATEAU_PATIENCE,
            "lr_plateau_min_lr": LR_PLATEAU_MIN_LR,
            "weight_decay": WEIGHT_DECAY,
            "backbone_lr_after_unfreeze": BACKBONE_LR_AFTER_UNFREEZE,
            "patch_checkpoint_path": PATCH_CHECKPOINT_PATH
        }
    )

    # ================= CORREÇÃO: VERIFICA SE O CHECKPOINT DO PATCH CLASSIFIER EXISTE =================
    # Falha alto e claro em vez de deixar o EnsembleDualViewClassifier silenciosamente
    # carregar pesos errados/desatualizados (ou os padrões do timm) se o caminho estiver errado.
    if not os.path.exists(PATCH_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint do patch classifier não encontrado em '{PATCH_CHECKPOINT_PATH}'. "
            f"Confirme se esse é o arquivo certo (ex.: o mais recente gerado pelo "
            f"train_patches.py) antes de continuar."
        )
    print(f"✅ Checkpoint do patch classifier confirmado em: {PATCH_CHECKPOINT_PATH}")
    # ======================================================================================================

    # Integração do seu modelo de patches campeão
    model = EnsembleDualViewClassifier(
        pretrained_patch_path=PATCH_CHECKPOINT_PATH).to(device)

    # ================= ALTERAÇÃO (d): CONGELA OS BACKBONES NO INÍCIO =================
    # Durante as primeiras FREEZE_BACKBONE_EPOCHS épocas, só os classificadores (CC e MLO)
    # treinam. Isso estabiliza o início do fine-tuning do ensemble antes de liberar os backbones.
    if FREEZE_BACKBONE_EPOCHS > 0:
        for p in model.backbone_cc.parameters():
            p.requires_grad = False
        for p in model.backbone_mlo.parameters():
            p.requires_grad = False
        print(f"🧊 Backbones (CC e MLO) congelados por {FREEZE_BACKBONE_EPOCHS} época(s) de warm-up.")
    # ===================================================================================

    # ================= CORREÇÃO: pos_weight passado de fato para a loss =================
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value]).to(device))
    # ==========================================================================================

    optimizer = AdamW([
        {'params': model.backbone_cc.parameters(), 'lr': BACKBONE_LR_AFTER_UNFREEZE},
        {'params': model.backbone_mlo.parameters(), 'lr': BACKBONE_LR_AFTER_UNFREEZE},
        {'params': model.classifier_cc.parameters(), 'lr': 1e-4},
        {'params': model.classifier_mlo.parameters(), 'lr': 1e-4}
    ], weight_decay=WEIGHT_DECAY)

    # ================= NOVA ALTERAÇÃO: ReduceLROnPlateau =================
    # Reduz o LR quando a perda de validação (suavizada) para de melhorar, em vez de
    # decair num cronograma fixo.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        min_lr=LR_PLATEAU_MIN_LR
    )
    # =======================================================================

    # Inicializa o Scaler para Mixed Precision (Aceleração e Poupança de Memória)
    scaler = torch.amp.GradScaler('cuda')

    # ================= ALTERAÇÃO (b): SELEÇÃO PELO MELHOR MODELO POR PERDA DE VALIDAÇÃO =================
    # Assim como em train_patches.py, o checkpoint passa a ser salvo pela perda de validação
    # suavizada (métrica threshold-free e mais estável), não mais pelo MCC recalculado a cada época.
    melhor_valid_loss = float('inf')
    melhor_mcc_no_ponto_salvo = -1.0
    best_mcc = -1.0  # mantido apenas para referência/print, não é mais o critério de checkpoint

    # ================= ALTERAÇÃO (a): CONTROLE DE EARLY STOPPING =================
    epocas_sem_melhora = 0

    # ================= NOVA ALTERAÇÃO: histórico para a média móvel da perda de validação =====
    valid_loss_history = []

    model_path = 'checkpoints/best_dual_view_model_modified.pth'

    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")

        # ================= ALTERAÇÃO (d): DESCONGELA OS BACKBONES APÓS O WARM-UP =================
        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            for p in model.backbone_cc.parameters():
                p.requires_grad = True
            for p in model.backbone_mlo.parameters():
                p.requires_grad = True
            print(f"🔥 Backbones descongelados a partir da época {epoch+1}. Fine-tuning completo iniciado.")
        # =============================================================================================

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc="Treino")
        for batch_idx, (img_cc, img_mlo, density, labels) in enumerate(loop):
            img_cc, img_mlo, density = img_cc.to(device), img_mlo.to(device), density.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.amp.autocast('cuda'):
                # Passamos a densidade no forward
                out_cc, out_mlo = model(img_cc, img_mlo, density)
                loss_cc = criterion(out_cc, labels)
                loss_mlo = criterion(out_mlo, labels)

                # ================= NOVA ALTERAÇÃO: LOSS SOBRE A PROBABILIDADE DO ENSEMBLE =================
                # Calcula a probabilidade combinada (mesma fórmula usada na avaliação) DENTRO do
                # grafo de autograd, para que o gradiente também penalize erros na decisão final
                # já fundida — não só em cada vista isoladamente.
                prob_cc_train = torch.sigmoid(out_cc)
                prob_mlo_train = torch.sigmoid(out_mlo)
                prob_ensemble_train = (prob_cc_train + prob_mlo_train) / 2.0
                loss_ensemble = weighted_bce_from_probs(prob_ensemble_train, labels, pos_weight_value)

                loss = (loss_cc + loss_mlo) / 2.0 + ENSEMBLE_LOSS_WEIGHT * loss_ensemble
                # ================================================================================================
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                # ================= NOVA ALTERAÇÃO: GRADIENT CLIPPING =================
                # Mesma proteção usada em train_patches.py contra gradientes explosivos,
                # aplicada apenas no momento em que o passo de otimização de fato ocorre.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # =======================================================================

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            loop.set_postfix(loss=loss.item() * accumulation_steps)

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        valid_loss_ensemble_component = 0.0  # NOVA ALTERAÇÃO: acompanhar o termo do ensemble isoladamente
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            loop_val = tqdm(valid_loader, desc="Validação")
            for img_cc, img_mlo, density, labels in loop_val:
                img_cc, img_mlo, density = img_cc.to(device), img_mlo.to(device), density.to(device)
                labels = labels.to(device).unsqueeze(1)

                with torch.amp.autocast('cuda'):
                    # Passamos a densidade no forward
                    out_cc, out_mlo = model(img_cc, img_mlo, density)
                    loss_cc = criterion(out_cc, labels)
                    loss_mlo = criterion(out_mlo, labels)

                    # ENSEMBLE: Sigmoide de cada vista, seguido da média
                    prob_cc = torch.sigmoid(out_cc)
                    prob_mlo = torch.sigmoid(out_mlo)
                    prob_ensemble = (prob_cc + prob_mlo) / 2.0

                    # ================= NOVA ALTERAÇÃO: mesma loss do ensemble usada no treino =================
                    # Mantém avg_valid_loss/EMA/early-stopping coerentes com o que de fato está
                    # sendo otimizado agora (média das losses por vista + loss da fusão).
                    loss_ensemble = weighted_bce_from_probs(prob_ensemble, labels, pos_weight_value)
                    loss = (loss_cc + loss_mlo) / 2.0 + ENSEMBLE_LOSS_WEIGHT * loss_ensemble
                    # ================================================================================================

                valid_loss += loss.item()
                valid_loss_ensemble_component += loss_ensemble.item()

                probs = prob_ensemble.cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_loss_ensemble_component = valid_loss_ensemble_component / len(valid_loader)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Procura o melhor limiar dinamicamente para esta época
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

        # ================= NOVA ALTERAÇÃO: SUAVIZAÇÃO DA PERDA DE VALIDAÇÃO =================
        # Média móvel das últimas VALID_LOSS_SMOOTHING_WINDOW épocas, usada tanto para o
        # checkpoint/early stopping quanto para alimentar o ReduceLROnPlateau.
        valid_loss_history.append(avg_valid_loss)
        janela = valid_loss_history[-VALID_LOSS_SMOOTHING_WINDOW:]
        valid_loss_suavizada = sum(janela) / len(janela)
        # =======================================================================================

        # ================= NOVA ALTERAÇÃO: ReduceLROnPlateau usa a métrica suavizada =========
        scheduler.step(valid_loss_suavizada)
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[2]['lr']
        # =======================================================================================

        print(f"Perda: Treino {avg_train_loss:.4f} | Valid {avg_valid_loss:.4f} | Valid (suavizada) {valid_loss_suavizada:.4f}")
        print(f"LR Atual -> Backbones: {current_lr_backbone:.2e} | Classificadores: {current_lr_classifier:.2e}")
        print(f"Métricas: AUC={val_auc:.4f} | MCC={val_mcc:.4f} | Acc={val_acc:.4f} | Sens={val_sens:.4f} | Espec={val_spec:.4f} | Prec={val_prec:.4f} | F1={val_f1:.4f}")

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

        wandb.log({
            "epoch": epoch + 1,
            "loss/train": avg_train_loss,
            "loss/validation": avg_valid_loss,
            "loss/validation_smoothed": valid_loss_suavizada,
            "loss/validation_ensemble_component": avg_valid_loss_ensemble_component,
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
            "graficos/matriz_confusao": wandb.Image(cm_path),
            "graficos/grad_cam": wandb.Image(gc_path),
            "backbone_frozen": epoch < FREEZE_BACKBONE_EPOCHS
        })

        # ================= ALTERAÇÃO (b): SELEÇÃO POR PERDA DE VALIDAÇÃO (SUAVIZADA) =================
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
            print(f"⏳ Sem melhora na perda de validação há {epocas_sem_melhora} época(s). "
                  f"(Melhor: {melhor_valid_loss:.4f})")
        # ====================================================================================

        # ================= ALTERAÇÃO (a): EARLY STOPPING =================
        if epocas_sem_melhora >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping ativado na época {epoch+1} "
                  f"(sem melhora por {EARLY_STOPPING_PATIENCE} épocas seguidas).")
            break
        # ====================================================================

    wandb.summary["best_valid_loss_smoothed"] = melhor_valid_loss
    wandb.summary["best_valid_mcc_at_checkpoint"] = melhor_mcc_no_ponto_salvo
    wandb.finish()

if __name__ == "__main__":
    # Agora sim, lendo corretamente o ficheiro passado como argumento
    train_dual_view_model('breast-level_annotations_final_limpo(2).csv')