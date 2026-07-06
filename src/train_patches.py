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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import torchvision.transforms as T

# ================= ALTERAÇÃO 5: GARANTIR REPRODUTIBILIDADE (SEEDS) =================
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

class MammogramPatchDataset(Dataset):
    def __init__(self, file_paths, labels, is_train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

        # ================= ALTERAÇÃO 3: MELHORIA NO DATA AUGMENTATION E NORMALIZAÇÃO =================
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            T.Normalize(mean=[0.5], std=[0.5]) # Normalização essencial para a ConvNeXt
        ])

        # Para validação e teste, aplicamos APENAS a normalização
        self.val_transforms = T.Compose([
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # 1. Carregar a 16-bits! (Crucial para não perder o contraste da lesão)
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None: # Proteção contra ficheiros corrompidos
            img = np.zeros((384, 384), dtype=np.uint16)
            
        # 2. Normalizar para Float [0, 1]
        img = img.astype(np.float32) / 65535.0
        
        # 3. Formatar para o PyTorch (Canal, Altura, Largura) -> (1, 384, 384)
        img_tensor = torch.tensor(img).unsqueeze(0)

        # 4. Aplicar transformações de acordo com o modo (Treino vs Validação/Teste)
        if self.is_train:
            img_tensor = self.train_transforms(img_tensor)
        else:
            img_tensor = self.val_transforms(img_tensor)

        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return img_tensor, label_tensor

def get_data(csv_path='finding_annotations_split.csv'):
    # 1. Carregar o CSV e criar um "Dicionário de Separação"
    print("A carregar as divisões oficiais do CSV...")
    df = pd.read_csv(csv_path)
    
    split_map = dict(zip(df['image_id'], df['split']))

    train_paths, train_labels = [], []
    valid_paths, valid_labels = [], []
    test_paths, test_labels = [], []

    # 2. Função interna para processar cada pasta
    def process_folder(folder_name, label_val):
        dir_path = os.path.join(PATCHES_DIR, folder_name)
        if not os.path.exists(dir_path):
            return

        for f in os.listdir(dir_path):
            if f.endswith('.png'):
                image_id = f.split('_')[0]
                img_split = split_map.get(image_id)
                full_path = os.path.join(dir_path, f)

                if img_split == 'training':
                    train_paths.append(full_path)
                    train_labels.append(label_val)
                elif img_split == 'validation':
                    valid_paths.append(full_path)
                    valid_labels.append(label_val)
                elif img_split == 'test':
                    test_paths.append(full_path)
                    test_labels.append(label_val)
                else:
                    pass

    # 3. Ler ambas as pastas e encaminhar os arquivos
    process_folder('normal', 0)
    process_folder('anormal', 1)

    if len(train_paths) == 0:
        raise ValueError(f"Nenhuma imagem mapeada para Treino! Verifique se a extração e o CSV estão corretos.")

    return train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels

def train_patch_model():
    print(f"A usar o dispositivo: {DEVICE}")
    
    # ================= WANDB INIT =================
    wandb.init(
        project="mestrado-visao-mamografia-patches", 
        name="Patch-Classifier-ConvNeXt-Final",    
        config={
            "learning_rate": LR,
            "architecture": "timm/convnext_small.in12k_ft_in1k_384",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": 384
        }
    )
    # ===============================================

    train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels = get_data()
    print(f"Total de Patches - Treino: {len(train_paths)} | Validação: {len(valid_paths)} | Teste: {len(test_paths)}")
    
    train_dataset = MammogramPatchDataset(train_paths, train_labels, is_train=True)
    valid_dataset = MammogramPatchDataset(valid_paths, valid_labels, is_train=False)
    test_dataset = MammogramPatchDataset(test_paths, test_labels, is_train=False)
    
    # ================= ALTERAÇÃO 1 (Parte 1): OTIMIZAÇÃO DATALOADER (pin_memory=True) =================
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Inicializa a ConvNeXt Base
    model = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=False, in_chans=1, num_classes=1)
    model = model.to(DEVICE)
    
    # Cálculo do pos_weight para balanceamento do BCE Loss
    num_normais = train_labels.count(0)
    num_anormais = train_labels.count(1)
    peso_positivo = num_normais / (num_anormais + 1e-8)
    print(f"Peso aplicado à classe anormal (pos_weight): {peso_positivo:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_positivo]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-1)
    
    # ================= ALTERAÇÃO 2: LEARNING RATE SCHEDULER =================
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # ================= ALTERAÇÃO 1 (Parte 2): INICIALIZAR SCALER PARA AMP =================
    scaler = torch.amp.GradScaler('cuda')

    melhor_mcc = -1.0
    os.makedirs('checkpoints', exist_ok=True)
    caminho_save = 'checkpoints/patch_classifier_convnext_small.in12k_ft_in1k_384.pth'
    
    # ================= LOOP DE TREINO =================
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Treino]")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # ================= ALTERAÇÃO 1 (Parte 3): MIXED PRECISION (AMP) =================
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Desescala antes do clip_grad_norm_ para os valores de gradiente corretos
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            # =================================================================================
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # VALIDAÇÃO
        model.eval()
        valid_loss = 0
        todas_preds = []
        todos_labels = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(valid_loader, desc=f"Época {epoch+1}/{EPOCHS} [Validação]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                
                # Na validação podemos usar o autocast para acelerar a inferência também
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    
                valid_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                todas_preds.extend(probs)
                todos_labels.extend(labels.cpu().numpy())
                
        # Atualizar o scheduler no final da época
        scheduler.step()
        
        # Calcular Métricas de Validação
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
        
        # Log da época no WandB (incluíndo LR)
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss_epoch": avg_train_loss,
            "valid_loss_epoch": avg_valid_loss,
            "valid_auc": auc,
            "valid_mcc": mcc
        })
        
        # Salva o melhor modelo baseado no MCC da Validação
        if mcc > melhor_mcc:
            melhor_mcc = mcc
            torch.save(model.state_dict(), caminho_save)
            print(f"✅ Novo melhor modelo guardado! (MCC: {mcc:.4f})")
            
    # ========================================================
    # AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (PÓS-TREINO)
    # ========================================================
    print("\n" + "="*50)
    print("🚀 A iniciar avaliação no Conjunto de Teste Cego...")
    
    # Carrega os pesos do modelo campeão
    model.load_state_dict(torch.load(caminho_save))
    model.eval()
    
    todas_preds_test = []
    todos_labels_test = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="A avaliar Teste"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            todas_preds_test.extend(probs)
            todos_labels_test.extend(labels.numpy())
            
    # Calcular Métricas do Teste
    auc_test = roc_auc_score(todos_labels_test, todas_preds_test)
    preds_binarias_test = (np.array(todas_preds_test) > 0.5).astype(int)
    mcc_test = matthews_corrcoef(todos_labels_test, preds_binarias_test)
    
    print("\n🏆 RESULTADOS DEFINITIVOS (TESTE CEGO) 🏆")
    print(f"AUC: {auc_test:.4f}")
    print(f"MCC: {mcc_test:.4f}")
    print("="*50)
    
    # Regista o resultado final no resumo do WandB
    wandb.summary["test_auc_final"] = auc_test
    wandb.summary["test_mcc_final"] = mcc_test
    
    # Fecha a sessão
    wandb.finish()

if __name__ == "__main__":
    train_patch_model()