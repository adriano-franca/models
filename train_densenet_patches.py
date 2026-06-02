import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURAÇÕES OTIMIZADAS PARA O NOVO DATASET
# ---------------------------------------------------------
CSV_PATH = "patches_annotations.csv"   # O novo CSV gerado
PATCHES_DIR = "./patches_dataset"      # A pasta com as imagens em PNG
BATCH_SIZE = 16                        # Podemos aumentar pois os PNGs são leves
EPOCHS = 50
LEARNING_RATE = 1e-4
LOG_FILE = "training_metrics_densenet.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('log_treinamento_densenet.txt', 'a') as f:
    f.write(f"Iniciando treinamento otimizado usando dispositivo: {DEVICE}\n")

# ---------------------------------------------------------
# 2. DATASET SIMPLIFICADO (Carregamento Ultrarrápido)
# ---------------------------------------------------------
class PreExtractedPatchDataset(Dataset):
    def __init__(self, df, patches_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.patches_dir = patches_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.patches_dir, row['patch_filename'])
        
        # Abre o PNG já recortado
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor([row['label']], dtype=torch.float32)
        img_label = torch.tensor([row['img_label']], dtype=torch.float32)
        image_id = row['image_id']
        is_valid = True # Sempre True, pois já filtramos o fundo preto na extração
        
        return image, label, image_id, is_valid, img_label

# ---------------------------------------------------------
# 3. PREPARAÇÃO E TREINAMENTO
# ---------------------------------------------------------
def main():
    print("A carregar as anotações dos recortes...")
    df = pd.read_csv(CSV_PATH)

    # Divisão 80/10/10 garantindo que imagens do mesmo paciente não vazem
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['study_id']))
    df_train, df_temp = df.iloc[train_idx], df.iloc[temp_idx]

    gss_temp = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_temp.split(df_temp, groups=df_temp['study_id']))
    df_val, df_test = df_temp.iloc[val_idx], df_temp.iloc[test_idx]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PreExtractedPatchDataset(df_train, PATCHES_DIR, transform_train)
    val_dataset = PreExtractedPatchDataset(df_val, PATCHES_DIR, transform_val)
    test_dataset = PreExtractedPatchDataset(df_test, PATCHES_DIR, transform_val)

    # num_workers=4 reativado para máxima velocidade da CPU enviando para a GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = models.densenet121(weights='DEFAULT')
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(DEVICE)

    # Cálculo dos pesos para mitigar o desbalanceamento
    train_pos = sum(df_train['label'] == 1.0)
    train_neg = len(df_train) - train_pos
    pos_weight = torch.tensor([train_neg / (train_pos + 1e-5)]).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history_train_loss, history_val_loss = [], []

    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoca', 'Loss_T', 'Loss_V', 'AUC', 'MCC', 'TP', 'FN', 'TN', 'FP'])

    print(f"Iniciando treinamento com {len(df_train)} recortes em {EPOCHS} épocas...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels, _, _, _ in tqdm(train_loader, desc=f"Treino Epoca {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validação usando agregação por imagem (Max-Pooling)
        model.eval()
        val_loss = 0.0
        img_preds = {}
        img_labels = {}
        
        with torch.no_grad():
            for inputs, labels, image_ids, _, true_img_labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                for i in range(len(image_ids)):
                    img_id = image_ids[i]
                    if img_id not in img_preds:
                        img_preds[img_id] = []
                        img_labels[img_id] = true_img_labels[i].item()
                    img_preds[img_id].append(probs[i][0])
        
        val_loss /= len(val_loader)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        
        y_true_img, y_score_img = [], []
        for img_id in img_preds.keys():
            y_true_img.append(img_labels[img_id])
            y_score_img.append(max(img_preds[img_id])) # Max-Pooling do score dos recortes
            
        y_pred_img = [1 if score >= 0.5 else 0 for score in y_score_img]

        try:
            auc = roc_auc_score(y_true_img, y_score_img)
            mcc = matthews_corrcoef(y_true_img, y_pred_img)
            tn, fp, fn, tp = confusion_matrix(y_true_img, y_pred_img, labels=[0, 1]).ravel()
        except ValueError:
            auc, mcc, tn, fp, fn, tp = 0.0, 0.0, 0, 0, 0, 0

        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{auc:.4f}", f"{mcc:.4f}", tp, fn, tn, fp])

        print(f"Epoca {epoch+1}/{EPOCHS} | Loss T: {train_loss:.4f} | Loss V: {val_loss:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")
        print(f"Matriz de Confusão -> TP:{tp} | FN:{fn} | TN:{tn} | FP:{fp}\n")

    # Gráfico e salvamento dos pesos
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), history_train_loss, color='#800000', label='Loss Treino')
    plt.plot(range(1, EPOCHS+1), history_val_loss, color='#800000', linestyle='--', label='Loss Validação')
    plt.title('Evolução do Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('densenet_loss_curve.png')
    
    torch.save(model.state_dict(), 'densenet_patch_model.pth')

    print("\n" + "="*50)
    print("INICIANDO AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (10%)")
    print("="*50)
    
    model.eval()
    img_preds_test = {}
    img_labels_test = {}
    
    with torch.no_grad():
        for inputs, _, image_ids, _, true_img_labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(len(image_ids)):
                img_id = image_ids[i]
                if img_id not in img_preds_test:
                    img_preds_test[img_id] = []
                    img_labels_test[img_id] = true_img_labels[i].item()
                img_preds_test[img_id].append(probs[i][0])
                
    y_true_test, y_score_test = [], []
    for img_id in img_preds_test.keys():
        y_true_test.append(img_labels_test[img_id])
        y_score_test.append(max(img_preds_test[img_id]))
        
    y_pred_test = [1 if score >= 0.5 else 0 for score in y_score_test]
    
    try:
        auc_test = roc_auc_score(y_true_test, y_score_test)
        mcc_test = matthews_corrcoef(y_true_test, y_pred_test)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1]).ravel()
    except ValueError:
        auc_test, mcc_test, tn_t, fp_t, fn_t, tp_t = 0.0, 0.0, 0, 0, 0, 0

    print(f"RESULTADO DO TESTE | AUC: {auc_test:.4f} | MCC: {mcc_test:.4f}")
    print(f"Matriz de Confusão (Teste) -> TP:{tp_t} | FN:{fn_t} | TN:{tn_t} | FP:{fp_t}\n")
    
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TESTE_FINAL', '-', '-', f"{auc_test:.4f}", f"{mcc_test:.4f}", tp_t, fn_t, tn_t, fp_t])

if __name__ == '__main__':
    main()