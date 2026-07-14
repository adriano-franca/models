import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
from tqdm import tqdm
import csv

# ---------------------------------------------------------
# 1. CONFIGURAÇÕES
# ---------------------------------------------------------
# Aponta diretamente para a sua pasta na raiz do projeto
ROOT_DIR = "dataset_patches" 
BATCH_SIZE = 16 
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. DATASET PERSONALIZADO (Lendo direto das Pastas)
# ---------------------------------------------------------
class PatchFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Lemos as pastas e forçamos a numeração correta das classes
        # 'normal' = 0 (Saudável), 'anormal' = 1 (Lesão/Alvo)
        class_mapping = {'normal': 0, 'anormal': 1}
        
        for class_name, label in class_mapping.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ AVISO: A pasta {class_dir} não foi encontrada!")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([label], dtype=torch.float32)

# ---------------------------------------------------------
# 3. TREINAMENTO E VALIDAÇÃO
# ---------------------------------------------------------
def main():
    print(f"🚀 Iniciando configuração no dispositivo: {DEVICE}")

    # Transformações
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), # Adiciona uma leve rotação para robustez
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Carrega todo o dataset
    print("A mapear as imagens nas pastas...")
    full_dataset = PatchFolderDataset(ROOT_DIR, transform=None)
    total_size = len(full_dataset)
    
    if total_size == 0:
        print("❌ ERRO: Nenhuma imagem encontrada. Verifique o caminho da pasta ROOT_DIR.")
        return
        
    print(f"📁 Total de recortes encontrados: {total_size}")

    # Divisão Simples: 80% Treino, 20% Validação
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Aplicamos as transformações corretas a cada subconjunto (um truque do PyTorch)
    train_subset.dataset.transform = transform_train
    val_subset.dataset.transform = transform_val

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Cálculo do pos_weight para desbalanceamento (se houver muito mais normais que anormais)
    labels_treino = [full_dataset.labels[i] for i in train_subset.indices]
    train_pos = sum(labels_treino)
    train_neg = len(labels_treino) - train_pos
    peso_pos = train_neg / (train_pos + 1e-8)
    
    print(f"⚖️ Balanceamento: {train_neg} Normais vs {train_pos} Anormais. pos_weight={peso_pos:.2f}")

    # Instancia a DenseNet121
    model = models.densenet121(weights='DEFAULT')
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([peso_pos]).to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_mcc = -1.0

    print("\n🔥 A iniciar o Treinamento da DenseNet...")
    for epoch in range(EPOCHS):
        # ================= TREINO =================
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Treino]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)

        # ================= VALIDAÇÃO =================
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Época {epoch+1}/{EPOCHS} [Validação]")
            for inputs, labels in loop_val:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        # Cálculo das Métricas usando o limiar padrão de 50%
        all_preds = [1 if p >= 0.5 else 0 for p in all_probs]
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
            mcc = matthews_corrcoef(all_labels, all_preds)
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        except ValueError:
            auc, mcc, tn, fp, fn, tp = 0.0, 0.0, 0, 0, 0, 0

        print(f"\n--- Fim da Época {epoch+1} ---")
        print(f"Perda Média: Treino {train_loss:.4f} | Valid {val_loss:.4f}")
        print(f"Métricas (Nível Recorte): AUC = {auc:.4f} | MCC = {mcc:.4f}")
        print(f"Confusão: TP:{tp} | FN:{fn} | TN:{tn} | FP:{fp}\n")

        # Salva o melhor modelo baseado no MCC
        if mcc > best_mcc:
            best_mcc = mcc
            torch.save(model.state_dict(), 'checkpoints/best_patch_densenet.pth')
            print(f"🌟 >>> Novo melhor modelo guardado no disco! (MCC: {best_mcc:.4f}) <<< 🌟\n")

    print(f"✅ Treinamento concluído! O modelo final otimizado está salvo na pasta checkpoints.")

if __name__ == "__main__":
    main()