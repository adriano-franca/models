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
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import torchvision.transforms as T

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
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= ALTERAÇÃO 1: NOVO MODELO CUSTOMIZADO =================
class PatchClassifierWithDensity(nn.Module):
    def __init__(self, pretrained=True):
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
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features + 4, 128),
            nn.GELU(),
            nn.Dropout(0.2),
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

class MammogramPatchDataset(Dataset):
    def __init__(self, file_paths, densities, labels, is_train=False):
        self.file_paths = file_paths
        self.densities = densities # Nova lista com as densidades raw
        self.labels = labels
        self.is_train = is_train

        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
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
        img = img.astype(np.float32) / 65535.0
        img_tensor = torch.tensor(img).unsqueeze(0)

        if self.is_train:
            img_tensor = self.train_transforms(img_tensor)
        else:
            img_tensor = self.val_transforms(img_tensor)
            
        # === 2. DENSIDADE (ONE-HOT) ===
        raw_density = str(self.densities[idx]).upper()
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
        dens_idx = mapping.get(raw_density, 2) # Padrão para 'C'
        
        density_tensor = torch.zeros(4, dtype=torch.float32)
        density_tensor[dens_idx] = 1.0

        # === 3. RÓTULO ===
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        # Retorna agora as 3 variáveis
        return img_tensor, density_tensor, label_tensor

def get_data(csv_path='finding_annotations_split.csv'):
    print("A carregar as divisões e densidades do CSV...")
    df = pd.read_csv(csv_path)
    
    # ================= ALTERAÇÃO 2: MAPEAR DENSIDADE DO CSV =================
    split_map = dict(zip(df['image_id'], df['split']))
    density_map = dict(zip(df['image_id'], df.get('breast_density', 'C'))) # 'C' como default caso a coluna não exista

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
                img_density = density_map.get(image_id, 'C') # Recupera a densidade deste ID
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
        raise ValueError(f"Nenhuma imagem mapeada para Treino! Verifique se a extração e o CSV estão corretos.")

    # Retorna agora as densidades também
    return train_paths, valid_paths, test_paths, train_densities, valid_densities, test_densities, train_labels, valid_labels, test_labels

def train_patch_model():
    print(f"A usar o dispositivo: {DEVICE}")
    
    wandb.init(
        project="mestrado-visao-mamografia-patches", 
        name="Patch-Classifier-ConvNeXt-DensityFusion",    
        config={
            "learning_rate": LR,
            "architecture": "ConvNeXt_With_MetadataFusion",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": 384
        }
    )

    # Desempacota as densidades
    train_paths, valid_paths, test_paths, train_dens, valid_dens, test_dens, train_labels, valid_labels, test_labels = get_data()
    print(f"Total de Patches - Treino: {len(train_paths)} | Validação: {len(valid_paths)} | Teste: {len(test_paths)}")
    
    # Passa as densidades para os Datasets
    train_dataset = MammogramPatchDataset(train_paths, train_dens, train_labels, is_train=True)
    valid_dataset = MammogramPatchDataset(valid_paths, valid_dens, valid_labels, is_train=False)
    test_dataset = MammogramPatchDataset(test_paths, test_dens, test_labels, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Inicializa o novo modelo (com o backbone pré-treinado na ImageNet e pronto para fundir densidade)
    model = PatchClassifierWithDensity(pretrained=True)
    model = model.to(DEVICE)
    
    num_normais = train_labels.count(0)
    num_anormais = train_labels.count(1)
    peso_positivo = num_normais / (num_anormais + 1e-8)
    print(f"Peso aplicado à classe anormal (pos_weight): {peso_positivo:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_positivo]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-1)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')

    melhor_mcc = -1.0
    os.makedirs('checkpoints', exist_ok=True)
    caminho_save = 'checkpoints/patch_classifier_convnext_density.pth'
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Treino]")
        
        # ================= ALTERAÇÃO 3: DESEMPACOTAMENTO DO DATALOADER (TREINO) =================
        for imgs, densities, labels in loop:
            imgs, densities, labels = imgs.to(DEVICE), densities.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                # Forward agora recebe a imagem e a densidade
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
        valid_loss = 0
        todas_preds = []
        todos_labels = []
        
        with torch.no_grad():
            # ================= ALTERAÇÃO 4: DESEMPACOTAMENTO DO DATALOADER (VALIDAÇÃO) =================
            for imgs, densities, labels in tqdm(valid_loader, desc=f"Época {epoch+1}/{EPOCHS} [Validação]"):
                imgs, densities, labels = imgs.to(DEVICE), densities.to(DEVICE), labels.to(DEVICE)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(imgs, densities)
                    loss = criterion(outputs, labels)
                    
                valid_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                todas_preds.extend(probs)
                todos_labels.extend(labels.cpu().numpy())
                
        scheduler.step()
        
        auc = roc_auc_score(todos_labels, todas_preds)
        preds_binarias = (np.array(todas_preds) > 0.5).astype(int)
        mcc = matthews_corrcoef(todos_labels, preds_binarias)
        
        avg_train_loss = train_loss/len(train_loader)
        avg_valid_loss = valid_loss/len(valid_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"--- Fim da Época {epoch+1} ---")
        print(f"LR Atual: {current_lr:.2e}")
        print(f"Perda Média: Treino {avg_train_loss:.4f} | Valid {avg_valid_loss:.4f}")
        print(f"Métricas: AUC = {auc:.4f} | MCC = {mcc:.4f}\n")
        
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss_epoch": avg_train_loss,
            "valid_loss_epoch": avg_valid_loss,
            "valid_auc": auc,
            "valid_mcc": mcc
        })
        
        if mcc > melhor_mcc:
            melhor_mcc = mcc
            torch.save(model.state_dict(), caminho_save)
            print(f"✅ Novo melhor modelo guardado! (MCC: {mcc:.4f})")
            
    print("\n" + "="*50)
    print("🚀 A iniciar avaliação no Conjunto de Teste Cego...")
    
    model.load_state_dict(torch.load(caminho_save))
    model.eval()
    
    todas_preds_test = []
    todos_labels_test = []
    
    with torch.no_grad():
        # ================= ALTERAÇÃO 5: DESEMPACOTAMENTO DO DATALOADER (TESTE) =================
        for imgs, densities, labels in tqdm(test_loader, desc="A avaliar Teste"):
            imgs, densities = imgs.to(DEVICE), densities.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(imgs, densities)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            todas_preds_test.extend(probs)
            todos_labels_test.extend(labels.numpy())
            
    auc_test = roc_auc_score(todos_labels_test, todas_preds_test)
    preds_binarias_test = (np.array(todas_preds_test) > 0.5).astype(int)
    mcc_test = matthews_corrcoef(todos_labels_test, preds_binarias_test)
    
    print("\n🏆 RESULTADOS DEFINITIVOS (TESTE CEGO) 🏆")
    print(f"AUC: {auc_test:.4f}")
    print(f"MCC: {mcc_test:.4f}")
    print("="*50)
    
    wandb.summary["test_auc_final"] = auc_test
    wandb.summary["test_mcc_final"] = mcc_test
    wandb.finish()

if __name__ == "__main__":
    train_patch_model()