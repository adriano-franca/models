#train_patches.py

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
import wandb
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import torchvision.transforms as T
import matplotlib.pyplot as plt

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

# ================= CONFIGURAÇÕES =================
PATCHES_DIR = 'dataset_patches'
BATCH_SIZE = 16
EPOCHS = 20              # teto de épocas; o early stopping normalmente para antes
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= ALTERAÇÃO (d): WARM-UP COM BACKBONE CONGELADO =================
# Número de épocas iniciais em que só o classificador treina (backbone congelado).
# Ajuda a estabilizar o início do treino antes de liberar o fine-tuning do backbone.
FREEZE_BACKBONE_EPOCHS = 2

# ================= ALTERAÇÃO (a): EARLY STOPPING =================
# Nº de épocas sem melhora na perda de validação antes de parar o treino.
EARLY_STOPPING_PATIENCE = 4

# ================= ALTERAÇÃO (c): REGULARIZAÇÃO =================
DROPOUT_RATE = 0.35  # antes: 0.2

# ================= NOVA ALTERAÇÃO: ReduceLROnPlateau =================
# Em vez de um cosine fixo que ignora se o modelo está platôando ou piorando,
# o LR só cai quando a perda de validação (suavizada) para de melhorar.
LR_PLATEAU_FACTOR = 0.5       # multiplica o LR por isso quando estagna
LR_PLATEAU_PATIENCE = 2       # nº de épocas sem melhora (na métrica suavizada) antes de reduzir o LR
LR_PLATEAU_MIN_LR = 1e-7

# ================= NOVA ALTERAÇÃO: suavização da perda de validação =================
# Com só 585 patches de validação, a perda de época a época é ruidosa. Usar uma
# média móvel evita que o early stopping/checkpoint reajam a um pico isolado.
VALID_LOSS_SMOOTHING_WINDOW = 3

# Pasta específica para os plots de explicabilidade (Grad-CAM)
GRADCAM_DIR = os.path.join('plots', 'patches')

# ================= MODELO CUSTOMIZADO =================
class PatchClassifierWithDensity(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=DROPOUT_RATE):
        super(PatchClassifierWithDensity, self).__init__()
        # Inicializa a ConvNeXt sem o classificador final (num_classes=0)
        self.backbone = timm.create_model(
            'timm/convnext_small.in12k_ft_in1k_384',
            pretrained=pretrained,
            in_chans=1,
            num_classes=0
        )

        in_features = self.backbone.num_features

        # Novo classificador que recebe as características da imagem + 4 canais da densidade
        # ALTERAÇÃO (c): dropout aumentado (0.2 -> 0.35 por padrão) para reduzir overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features + 4, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x, density):
        # 1. Extrai características visuais do patch
        features = self.backbone(x)

        # 2. Concatena com o vetor BI-RADS
        fused = torch.cat((features, density), dim=1)

        # 3. Classificação final
        out = self.classifier(fused)
        return out
# =========================================================================

# ================= GRAD-CAM MANUAL (SUBSTITUI O CAPTUM SALIENCY) =========
class GradCAM:
    """
    Implementação manual de Grad-CAM via forward/backward hooks.
    Não depende de bibliotecas externas (captum, etc.) e funciona com
    qualquer camada convolucional, incluindo os stages do ConvNeXt (timm).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def __call__(self, img_tensor, density_tensor):
        self.model.zero_grad()
        output = self.model(img_tensor, density_tensor)
        output.backward(torch.ones_like(output))

        # Peso de importância de cada canal = média espacial do gradiente
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # mantém só ativações que contribuem positivamente

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, output


def plot_gradcam_patch(gradcam, img_tensor, density_tensor, epoch, output_dir=GRADCAM_DIR):
    os.makedirs(output_dir, exist_ok=True)
    gradcam.model.eval()

    img = img_tensor.unsqueeze(0).to(DEVICE)
    density = density_tensor.unsqueeze(0).to(DEVICE)

    # CORREÇÃO: garante que exista gradiente até a ativação alvo mesmo quando o
    # backbone está congelado (warm-up). Sem isso, com todos os parâmetros do
    # backbone com requires_grad=False e a imagem também sem requires_grad,
    # o grafo inteiro fica sem gradiente e o hook nunca dispara (gradients=None).
    img.requires_grad_(True)

    cam, _ = gradcam(img, density)

    # Redimensiona o mapa de ativação para o tamanho da imagem original
    cam_resized = cv2.resize(cam, (img.shape[-1], img.shape[-2]))

    viz_img = img.squeeze().cpu().detach().numpy()
    viz_img = (viz_img - viz_img.min()) / (viz_img.max() - viz_img.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(viz_img, cmap='gray')
    plt.imshow(cam_resized, cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.title(f"Grad-CAM - Época {epoch+1}")
    plt.savefig(os.path.join(output_dir, f'gradcam_ep{epoch+1}_rand.png'))
    plt.close()
# =========================================================================

class MammogramPatchDataset(Dataset):
    def __init__(self, file_paths, densities, labels, is_train=False):
        self.file_paths = file_paths
        self.densities = densities  # Lista com as densidades raw
        self.labels = labels
        self.is_train = is_train

        # ALTERAÇÃO (c): augmentação um pouco mais forte para reduzir overfitting
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.RandomAutocontrast(p=0.2),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        self.val_transforms = T.Compose([
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # === 1. IMAGEM ===
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            img = np.zeros((384, 384), dtype=np.uint16)

        # === 1.1. APLICAÇÃO DE CLAHE PARA MELHORAR O CONTRASTE ===
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        # =========================================================

        img = img.astype(np.float32) / 65535.0
        img_tensor = torch.tensor(img).unsqueeze(0)

        if self.is_train:
            img_tensor = self.train_transforms(img_tensor)
        else:
            img_tensor = self.val_transforms(img_tensor)

        # === 2. DENSIDADE (ONE-HOT) ===
        raw_density = str(self.densities[idx]).upper()
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
        dens_idx = mapping.get(raw_density, 2)  # Padrão para 'C'

        density_tensor = torch.zeros(4, dtype=torch.float32)
        density_tensor[dens_idx] = 1.0

        # === 3. RÓTULO ===
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return img_tensor, density_tensor, label_tensor


def get_data(csv_path='finding_annotations_split.csv'):
    print("A carregar as divisões e densidades do CSV...")
    df = pd.read_csv(csv_path)

    split_map = dict(zip(df['image_id'], df['split']))
    density_map = dict(zip(df['image_id'], df.get('breast_density', 'C')))  # 'C' como default

    train_paths, train_densities, train_labels = [], [], []
    valid_paths, valid_densities, valid_labels = [], [], []
    test_paths, test_densities, test_labels = [], [], []

    def process_folder(folder_name, label_val):
        dir_path = os.path.join(PATCHES_DIR, folder_name)
        if not os.path.exists(dir_path):
            return

        for f in os.listdir(dir_path):
            if f.endswith('.png'):
                image_id = f.split('_')[0]
                img_split = split_map.get(image_id)
                img_density = density_map.get(image_id, 'C')
                full_path = os.path.join(dir_path, f)

                if img_split == 'training':
                    train_paths.append(full_path)
                    train_densities.append(img_density)
                    train_labels.append(label_val)
                elif img_split == 'validation':
                    valid_paths.append(full_path)
                    valid_densities.append(img_density)
                    valid_labels.append(label_val)
                elif img_split == 'test':
                    test_paths.append(full_path)
                    test_densities.append(img_density)
                    test_labels.append(label_val)

    process_folder('normal', 0)
    process_folder('anormal', 1)

    if len(train_paths) == 0:
        raise ValueError("Nenhuma imagem mapeada para Treino! Verifique se a extração e o CSV estão corretos.")

    return train_paths, valid_paths, test_paths, train_densities, valid_densities, test_densities, train_labels, valid_labels, test_labels


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


def train_patch_model():
    print(f"A usar o dispositivo: {DEVICE}")

    os.makedirs(GRADCAM_DIR, exist_ok=True)

    wandb.init(
        project="mestrado-visao-mamografia-patches",
        name="Patch-Classifier-ConvNeXt-DensityFusion-CLAHE-GradCAM",
        config={
            "learning_rate": LR,
            "architecture": "ConvNeXt_With_MetadataFusion",
            "epochs_max": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": 384,
            "explainability": "GradCAM",
            "dropout_rate": DROPOUT_RATE,
            "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "model_selection_criterion": "valid_loss_smoothed",
            "valid_loss_smoothing_window": VALID_LOSS_SMOOTHING_WINDOW,
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_plateau_factor": LR_PLATEAU_FACTOR,
            "lr_plateau_patience": LR_PLATEAU_PATIENCE,
            "lr_plateau_min_lr": LR_PLATEAU_MIN_LR
        }
    )

    train_paths, valid_paths, test_paths, train_dens, valid_dens, test_dens, train_labels, valid_labels, test_labels = get_data()
    print(f"Total de Patches - Treino: {len(train_paths)} | Validação: {len(valid_paths)} | Teste: {len(test_paths)}")

    train_dataset = MammogramPatchDataset(train_paths, train_dens, train_labels, is_train=True)
    valid_dataset = MammogramPatchDataset(valid_paths, valid_dens, valid_labels, is_train=False)
    test_dataset = MammogramPatchDataset(test_paths, test_dens, test_labels, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Inicializa o modelo (backbone pré-treinado + fusão com densidade)
    model = PatchClassifierWithDensity(pretrained=True)
    model = model.to(DEVICE)

    # ================= ALTERAÇÃO (d): CONGELA O BACKBONE NO INÍCIO =================
    # Durante as primeiras FREEZE_BACKBONE_EPOCHS épocas, só o classificador treina.
    # Isso estabiliza o início do treino antes de liberar o fine-tuning do backbone.
    if FREEZE_BACKBONE_EPOCHS > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"🧊 Backbone congelado por {FREEZE_BACKBONE_EPOCHS} época(s) de warm-up.")
    # =================================================================================

    # ================= GRAD-CAM: instanciado uma única vez =================
    # Target layer: último stage do ConvNeXt (mapa espacial, antes do pooling global)
    gradcam = GradCAM(model, target_layer=model.backbone.stages[-1])
    # =========================================================================

    num_normais = train_labels.count(0)
    num_anormais = train_labels.count(1)
    peso_positivo = num_normais / (num_anormais + 1e-8)
    print(f"Peso aplicado à classe anormal (pos_weight): {peso_positivo:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_positivo]).to(DEVICE))
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-1)

    # ReduceLROnPlateau: reduz o LR quando a perda de validação (suavizada) estagna,
    # em vez de decair num cronograma fixo que ignora o comportamento real do treino.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        min_lr=LR_PLATEAU_MIN_LR
    )
    scaler = torch.amp.GradScaler('cuda')

    # ALTERAÇÃO (b): seleção do melhor modelo passa a ser pela perda de validação
    # (métrica threshold-free e mais estável), não mais pelo MCC recalculado a cada época.
    melhor_valid_loss = float('inf')       # menor perda SUAVIZADA já vista
    melhor_mcc_no_ponto_salvo = -1.0
    melhor_limiar_global = 0.5

    # ALTERAÇÃO (a): controle de early stopping
    epocas_sem_melhora = 0

    # NOVA ALTERAÇÃO: histórico para a média móvel da perda de validação
    valid_loss_history = []

    os.makedirs('checkpoints', exist_ok=True)
    caminho_save = 'checkpoints/patch_classifier_convnext_density_clahe.pth'

    for epoch in range(EPOCHS):

        # ================= ALTERAÇÃO (d): DESCONGELA O BACKBONE APÓS O WARM-UP =================
        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print(f"🔥 Backbone descongelado a partir da época {epoch+1}. Fine-tuning completo iniciado.")
        # ==========================================================================================

        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Treino]")

        for imgs, densities, labels in loop:
            imgs, densities, labels = imgs.to(DEVICE), densities.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(imgs, densities)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()

        # ================= GRAD-CAM: gerado fora do autocast, para evitar problemas de precisão =================
        idx_random = random.randint(0, len(valid_dataset) - 1)
        img_s, dens_s, _ = valid_dataset[idx_random]

        plot_gradcam_patch(gradcam, img_s, dens_s, epoch, GRADCAM_DIR)

        path_img = os.path.join(GRADCAM_DIR, f"gradcam_ep{epoch+1}_rand.png")
        wandb.log({"GradCAM_Patch": wandb.Image(path_img)})
        # ===========================================================================================================

        valid_loss = 0
        todas_preds = []
        todos_labels = []

        with torch.no_grad():
            for imgs, densities, labels in tqdm(valid_loader, desc=f"Época {epoch+1}/{EPOCHS} [Validação]"):
                imgs, densities, labels = imgs.to(DEVICE), densities.to(DEVICE), labels.to(DEVICE)

                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(imgs, densities)
                    loss = criterion(outputs, labels)

                valid_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                todas_preds.extend(probs)
                todos_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(todos_labels, todas_preds)

        melhor_limiar, mcc = find_best_threshold(todos_labels, todas_preds)
        print(f"Limiar ideal desta época: {melhor_limiar:.2f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        # ================= NOVA ALTERAÇÃO: SUAVIZAÇÃO DA PERDA DE VALIDAÇÃO =================
        # Média móvel das últimas VALID_LOSS_SMOOTHING_WINDOW épocas. Usada tanto para
        # decidir o checkpoint/early stopping quanto para alimentar o ReduceLROnPlateau,
        # evitando reações a um pico isolado de ruído (comum com só 585 patches de validação).
        valid_loss_history.append(avg_valid_loss)
        janela = valid_loss_history[-VALID_LOSS_SMOOTHING_WINDOW:]
        valid_loss_suavizada = sum(janela) / len(janela)
        # =======================================================================================

        # ================= NOVA ALTERAÇÃO: ReduceLROnPlateau usa a métrica suavizada =========
        scheduler.step(valid_loss_suavizada)
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[1]['lr']
        # =======================================================================================

        print(f"--- Fim da Época {epoch+1} ---")
        print(f"LR Atual -> Backbone: {current_lr_backbone:.2e} | Classificador: {current_lr_classifier:.2e}")
        print(f"Perda Média: Treino {avg_train_loss:.4f} | Valid {avg_valid_loss:.4f} | Valid (suavizada) {valid_loss_suavizada:.4f}")
        print(f"Métricas: AUC = {auc:.4f} | MCC = {mcc:.4f}\n")

        wandb.log({
            "epoch": epoch + 1,
            "learning_rate_backbone": current_lr_backbone,
            "learning_rate_classifier": current_lr_classifier,
            "train_loss_epoch": avg_train_loss,
            "valid_loss_epoch": avg_valid_loss,
            "valid_loss_smoothed": valid_loss_suavizada,
            "valid_auc": auc,
            "valid_mcc": mcc,
            "backbone_frozen": epoch < FREEZE_BACKBONE_EPOCHS
        })

        # ================= ALTERAÇÃO (b): SELEÇÃO POR PERDA DE VALIDAÇÃO (SUAVIZADA) =================
        if valid_loss_suavizada < melhor_valid_loss:
            melhor_valid_loss = valid_loss_suavizada
            melhor_mcc_no_ponto_salvo = mcc
            melhor_limiar_global = melhor_limiar
            torch.save(model.state_dict(), caminho_save)
            epocas_sem_melhora = 0
            print(f"✅ Novo melhor modelo guardado! (Valid Loss suavizada: {valid_loss_suavizada:.4f} | MCC: {mcc:.4f})")
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

    print("\n" + "="*50)
    print("🚀 A iniciar avaliação no Conjunto de Teste Cego...")

    model.load_state_dict(torch.load(caminho_save))
    model.eval()

    todas_preds_test = []
    todos_labels_test = []

    with torch.no_grad():
        for imgs, densities, labels in tqdm(test_loader, desc="A avaliar Teste"):
            imgs, densities = imgs.to(DEVICE), densities.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(imgs, densities)
            probs = torch.sigmoid(outputs).cpu().numpy()

            todas_preds_test.extend(probs)
            todos_labels_test.extend(labels.numpy())

    auc_test = roc_auc_score(todos_labels_test, todas_preds_test)
    preds_binarias_test = (np.array(todas_preds_test) > melhor_limiar_global).astype(int)
    mcc_test = matthews_corrcoef(todos_labels_test, preds_binarias_test)

    print("\n🏆 RESULTADOS DEFINITIVOS (TESTE CEGO) 🏆")
    print(f"AUC: {auc_test:.4f}")
    print(f"MCC: {mcc_test:.4f}")
    print("="*50)

    wandb.summary["test_auc_final"] = auc_test
    wandb.summary["test_mcc_final"] = mcc_test
    wandb.summary["best_valid_loss_smoothed"] = melhor_valid_loss
    wandb.summary["best_valid_mcc_at_checkpoint"] = melhor_mcc_no_ponto_salvo
    wandb.summary["threshold_used"] = melhor_limiar_global
    wandb.finish()

    gradcam.remove_hooks()

if __name__ == "__main__":
    train_patch_model()