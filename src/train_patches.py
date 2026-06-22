import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# ================= CONFIGURAÇÕES =================
# Certifique-se de executar o comando 'python src/train_patches.py' a partir da RAIZ do projeto
# para que ele encontre esta pasta corretamente.
PATCHES_DIR = 'dataset_patches' 
BATCH_SIZE = 8
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
    if os.path.exists(normal_dir):
        for f in os.listdir(normal_dir):
            if f.endswith('.png'):
                paths.append(os.path.join(normal_dir, f))
                labels.append(0)
            
    # Ler anormais (1)
    anormal_dir = os.path.join(PATCHES_DIR, 'anormal')
    if os.path.exists(anormal_dir):
        for f in os.listdir(anormal_dir):
            if f.endswith('.png'):
                paths.append(os.path.join(anormal_dir, f))
                labels.append(1)
            
    if len(paths) == 0:
        raise ValueError(f"Nenhuma imagem encontrada na pasta {PATCHES_DIR}! Rode a extração primeiro.")

    # ================= DIVISÃO TRIPLA (80/10/10) =================
    # 1ª Divisão: 80% Treino / 20% Temporário
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    # 2ª Divisão: Divide os 20% Temporários a meio -> 10% Validação / 10% Teste
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )
    
    return train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels

def train_patch_model():
    print(f"A usar o dispositivo: {DEVICE}")
    
    # ================= WANDB INIT =================
    wandb.init(
        project="mestrado-visao-mamografia", 
        name="Patch-Classifier-ConvNeXt-Final",    
        config={
            "learning_rate": LR,
            "architecture": "convnext_base.fb_in22k",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": 224
        }
    )
    # ===============================================

    # Carrega os dados com a divisão tripla
    train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels = get_data()
    print(f"Total de Patches - Treino: {len(train_paths)} | Validação: {len(valid_paths)} | Teste: {len(test_paths)}")
    
    # Cria os 3 Datasets
    train_dataset = MammogramPatchDataset(train_paths, train_labels)
    valid_dataset = MammogramPatchDataset(valid_paths, valid_labels)
    test_dataset = MammogramPatchDataset(test_paths, test_labels)
    
    # Cria os 3 DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Inicializa a ConvNeXt Base
    model = timm.create_model('convnext_base.fb_in22k', pretrained=True, in_chans=1, num_classes=1)
    model = model.to(DEVICE)
    
    # Cálculo do pos_weight para balanceamento do BCE Loss
    num_normais = train_labels.count(0)
    num_anormais = train_labels.count(1)
    peso_positivo = num_normais / (num_anormais + 1e-8)
    print(f"Peso aplicado à classe anormal (pos_weight): {peso_positivo:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_positivo]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    melhor_mcc = -1.0
    os.makedirs('checkpoints', exist_ok=True)
    caminho_save = 'checkpoints/best_patch_classifier_modified.pth'
    
    # ================= LOOP DE TREINO =================
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
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                todas_preds.extend(probs)
                todos_labels.extend(labels.cpu().numpy())
                
        # Calcular Métricas de Validação
        auc = roc_auc_score(todos_labels, todas_preds)
        preds_binarias = (np.array(todas_preds) > 0.5).astype(int)
        mcc = matthews_corrcoef(todos_labels, preds_binarias)
        
        avg_train_loss = train_loss/len(train_loader)
        avg_valid_loss = valid_loss/len(valid_loader)
        
        print(f"--- Fim da Época {epoch+1} ---")
        print(f"Perda Média: Treino {avg_train_loss:.4f} | Valid {avg_valid_loss:.4f}")
        print(f"Métricas: AUC = {auc:.4f} | MCC = {mcc:.4f}\n")
        
        # Log da época no WandB
        wandb.log({
            "epoch": epoch + 1,
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