import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# ================= CONFIGURAÇÕES =================
PATCHES_DIR = 'dataset_patches'
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class MammogramPatchDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # 1. Carregar a 16-bits! (Crucial para não perder o contraste da lesão)
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None: # Proteção contra ficheiros corrompidos
            img = np.zeros((224, 224), dtype=np.uint16)
            
        # 2. Normalizar para Float [0, 1]
        img = img.astype(np.float32) / 65535.0
        
        # 3. Formatar para o PyTorch (Canal, Altura, Largura) -> (1, 224, 224)
        img_tensor = torch.tensor(img).unsqueeze(0)
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return img_tensor, label_tensor

def get_data():
    paths = []
    labels = []
    
    # Ler normais (0)
    normal_dir = os.path.join(PATCHES_DIR, 'normal')
    for f in os.listdir(normal_dir):
        if f.endswith('.png'):
            paths.append(os.path.join(normal_dir, f))
            labels.append(0)
            
    # Ler anormais (1)
    anormal_dir = os.path.join(PATCHES_DIR, 'anormal')
    for f in os.listdir(anormal_dir):
        if f.endswith('.png'):
            paths.append(os.path.join(anormal_dir, f))
            labels.append(1)
            
    # Divisão 80% Treino / 20% Validação
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return train_paths, valid_paths, train_labels, valid_labels

def train_patch_model():
    print(f"A usar o dispositivo: {DEVICE}")
    
    train_paths, valid_paths, train_labels, valid_labels = get_data()
    print(f"Total de Patches - Treino: {len(train_paths)} | Validação: {len(valid_paths)}")
    
    train_dataset = MammogramPatchDataset(train_paths, train_labels)
    valid_dataset = MammogramPatchDataset(valid_paths, valid_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Instanciar a ConvNeXt Base (Exatamente a mesma que usaremos depois)
    # in_chans=1 (porque é tons de cinza), num_classes=1 (porque é binário: normal/anormal)
    model = timm.create_model('convnext_base_in22k', pretrained=True, in_chans=1, num_classes=1)
    model = model.to(DEVICE)
    
    # Calcular o peso das classes para evitar o "Colapso" (a rede prever tudo como 0)
    num_normais = train_labels.count(0)
    num_anormais = train_labels.count(1)
    peso_positivo = num_normais / (num_anormais + 1e-8)
    print(f"Peso aplicado à classe anormal: {peso_positivo:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_positivo]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    melhor_mcc = -1.0
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Treino]")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
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
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                # Guardar predições para calcular AUC e MCC
                probs = torch.sigmoid(outputs).cpu().numpy()
                todas_preds.extend(probs)
                todos_labels.extend(labels.cpu().numpy())
                
        # Calcular Métricas
        auc = roc_auc_score(todos_labels, todas_preds)
        preds_binarias = (np.array(todas_preds) > 0.5).astype(int)
        mcc = matthews_corrcoef(todos_labels, preds_binarias)
        
        print(f"--- Fim da Época {epoch+1} ---")
        print(f"Perda Média: Treino {train_loss/len(train_loader):.4f} | Valid {valid_loss/len(valid_loader):.4f}")
        print(f"Métricas: AUC = {auc:.4f} | MCC = {mcc:.4f}\n")
        
        # Guardar o melhor modelo
        if mcc > melhor_mcc:
            melhor_mcc = mcc
            caminho_save = 'checkpoints/best_patch_classifier_modified.pth'
            torch.save(model.state_dict(), caminho_save)
            print(f"Novo melhor modelo guardado! (MCC: {mcc:.4f})")

if __name__ == "__main__":
    train_patch_model()