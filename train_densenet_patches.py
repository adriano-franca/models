import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURAÇÕES
# ---------------------------------------------------------
CSV_PATH = "finding_annotations.csv"
DICOM_DIR = "./"  
PATCH_SIZE = 512
STRIDE = 256
BG_THRESHOLD = 15
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
LOG_FILE = "training_metrics_densenet.csv" # Arquivo de log dedicado
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('log_treinamento_densenet.txt', 'a') as f:
    f.write(f"Usando dispositivo: {DEVICE}\n")

# ---------------------------------------------------------
# 2. DATASET PERSONALIZADO (Mantido)
# ---------------------------------------------------------
class MammographyPatchDataset(Dataset):
    def __init__(self, df, dicom_dir, patch_size=512, stride=256, is_train=True, transform=None):
        self.df = df
        self.dicom_dir = dicom_dir
        self.patch_size = patch_size
        self.stride = stride
        self.is_train = is_train
        self.transform = transform
        self.patches = []
        
        self.current_dcm_path = None
        self.current_img = None
        
        self._prepare_grid()
        
    def _prepare_grid(self):
        grouped = self.df.groupby('image_id')
        for image_id, group in grouped:
            study_id = group.iloc[0]['study_id']
            img_label = 0.0 if group.iloc[0]['finding_categories'] == "['No Finding']" else 1.0
            
            boxes = []
            for _, row in group.iterrows():
                if pd.notnull(row['xmin']) and row['finding_categories'] != "['No Finding']":
                    boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                    
            for y in range(0, 4000 - self.patch_size + 1, self.stride):
                for x in range(0, 4000 - self.patch_size + 1, self.stride):
                    patch_box = [x, y, x + self.patch_size, y + self.patch_size]
                    label = 0.0
                    for box in boxes:
                        if self._check_overlap(patch_box, box):
                            label = 1.0
                            break
                            
                    self.patches.append({
                        'study_id': study_id,
                        'image_id': image_id,
                        'x': x,
                        'y': y,
                        'label': label,
                        'img_label': img_label
                    })

    def _check_overlap(self, patch, lesion):
        ix_min, iy_min = max(patch[0], lesion[0]), max(patch[1], lesion[1])
        ix_max, iy_max = min(patch[2], lesion[2]), min(patch[3], lesion[3])
        return (ix_min < ix_max) and (iy_min < iy_max)

    def _load_img(self, study_id, image_id):
        path = os.path.join(self.dicom_dir, study_id, f"{image_id}.dicom")
        if self.current_dcm_path != path:
            dcm = pydicom.dcmread(path)
            self.current_img = dcm.pixel_array
            self.current_dcm_path = path
        return self.current_img

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        while True:
            p = self.patches[idx]
            try:
                img = self._load_img(p['study_id'], p['image_id'])
                
                y_max = min(p['y'] + self.patch_size, img.shape[0])
                x_max = min(p['x'] + self.patch_size, img.shape[1])
                patch_img = img[p['y']:y_max, p['x']:x_max]
                
                if patch_img.shape[0] < self.patch_size or patch_img.shape[1] < self.patch_size:
                    pad_y = self.patch_size - patch_img.shape[0]
                    pad_x = self.patch_size - patch_img.shape[1]
                    patch_img = np.pad(patch_img, ((0, pad_y), (0, pad_x)), mode='constant')

                is_valid = patch_img.mean() > BG_THRESHOLD
                
                if is_valid or not self.is_train:
                    patch_img = Image.fromarray(patch_img).convert('RGB')
                    if self.transform:
                        patch_img = self.transform(patch_img)
                    return patch_img, torch.tensor([p['label']], dtype=torch.float32), p['image_id'], is_valid, torch.tensor([p['img_label']], dtype=torch.float32)
                
            except Exception:
                pass
                
            idx = (idx + 1) % len(self.patches)

# ---------------------------------------------------------
# 3. PREPARAÇÃO E TREINAMENTO
# ---------------------------------------------------------
def main():
    df = pd.read_csv(CSV_PATH)

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

    train_dataset = MammographyPatchDataset(df_train, DICOM_DIR, PATCH_SIZE, STRIDE, True, transform_train)
    val_dataset = MammographyPatchDataset(df_val, DICOM_DIR, PATCH_SIZE, STRIDE, False, transform_val)
    test_dataset = MammographyPatchDataset(df_test, DICOM_DIR, PATCH_SIZE, STRIDE, False, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.densenet121(weights='DEFAULT')
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.to(DEVICE)

    train_pos = sum([1 for p in train_dataset.patches if p['label'] == 1.0])
    train_neg = len(train_dataset.patches) - train_pos
    pos_weight = torch.tensor([train_neg / (train_pos + 1e-5)]).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history_train_loss, history_val_loss = [], []

    # Criar arquivo de log e escrever o cabeçalho
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoca', 'Loss_T', 'Loss_V', 'AUC', 'MCC', 'TP', 'FN', 'TN', 'FP'])

    print("Iniciando treinamento com a arquitetura por recortes...")
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
        
        # Validação (Max-Pooling)
        model.eval()
        val_loss = 0.0
        img_preds = {}
        img_labels = {}
        
        with torch.no_grad():
            for inputs, labels, image_ids, is_valids, true_img_labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                valid_mask = is_valids.to(DEVICE)
                if valid_mask.sum() > 0:
                    loss = criterion(outputs[valid_mask], labels[valid_mask])
                    val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                for i in range(len(image_ids)):
                    if not is_valids[i]: continue
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
            y_score_img.append(max(img_preds[img_id])) # Max-Pooling
            
        # Classificação binária para Matriz de Confusão e MCC (limiar de 0.5)
        y_pred_img = [1 if score >= 0.5 else 0 for score in y_score_img]

        try:
            auc = roc_auc_score(y_true_img, y_score_img)
            mcc = matthews_corrcoef(y_true_img, y_pred_img)
            tn, fp, fn, tp = confusion_matrix(y_true_img, y_pred_img, labels=[0, 1]).ravel()
        except ValueError:
            auc, mcc, tn, fp, fn, tp = 0.0, 0.0, 0, 0, 0, 0

        # Escrever no arquivo CSV a cada época
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{auc:.4f}", f"{mcc:.4f}", tp, fn, tn, fp])

        # Imprimir no console no mesmo formato desejado
        print(f"Epoca {epoch+1}/{EPOCHS} | Loss T: {train_loss:.4f} | Loss V: {val_loss:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")
        print(f"Matriz de Confusão -> TP:{tp} | FN:{fn} | TN:{tn} | FP:{fp}\n")

    # Salva o gráfico e o modelo
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
    
    model.eval() # Garante que o modelo está em modo de avaliação
    img_preds_test = {}
    img_labels_test = {}
    
    with torch.no_grad():
        for inputs, labels, image_ids, is_valids, true_img_labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(len(image_ids)):
                if not is_valids[i]: continue
                img_id = image_ids[i]
                if img_id not in img_preds_test:
                    img_preds_test[img_id] = []
                    img_labels_test[img_id] = true_img_labels[i].item()
                img_preds_test[img_id].append(probs[i][0])
                
    y_true_test, y_score_test = [], []
    for img_id in img_preds_test.keys():
        y_true_test.append(img_labels_test[img_id])
        y_score_test.append(max(img_preds_test[img_id])) # Max-Pooling dos recortes
        
    y_pred_test = [1 if score >= 0.5 else 0 for score in y_score_test]
    
    try:
        auc_test = roc_auc_score(y_true_test, y_score_test)
        mcc_test = matthews_corrcoef(y_true_test, y_pred_test)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1]).ravel()
    except ValueError:
        auc_test, mcc_test, tn_t, fp_t, fn_t, tp_t = 0.0, 0.0, 0, 0, 0, 0

    print(f"RESULTADO DO TESTE | AUC: {auc_test:.4f} | MCC: {mcc_test:.4f}")
    print(f"Matriz de Confusão (Teste) -> TP:{tp_t} | FN:{fn_t} | TN:{tn_t} | FP:{fp_t}\n")
    
    # Opcional: Salvar o resultado do teste no arquivo de log
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TESTE_FINAL', '-', '-', f"{auc_test:.4f}", f"{mcc_test:.4f}", tp_t, fn_t, tn_t, fp_t])

if __name__ == '__main__':
    main()