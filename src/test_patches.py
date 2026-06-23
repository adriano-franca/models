import os
import cv2
import torch
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    matthews_corrcoef, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

# ================= CONFIGURAÇÕES =================
# Lembre-se de rodar na raiz do projeto: python src/test_patches.py
PATCHES_DIR = 'dataset_patches'
MODEL_PATH = 'checkpoints/best_patch_classifier_modified.pth'
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class MammogramPatchDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # 1. Carregar a 16-bits
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None: 
            img = np.zeros((224, 224), dtype=np.uint16)
            
        # 2. Normalizar
        img = img.astype(np.float32) / 65535.0
        
        # 3. Formatar para PyTorch -> (1, 224, 224)
        img_tensor = torch.tensor(img).unsqueeze(0)
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return img_tensor, label_tensor

def get_test_data():
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
        raise ValueError(f"Nenhuma imagem encontrada na pasta {PATCHES_DIR}!")

    # ================= RECRIAÇÃO DA DIVISÃO (80/10/10) =================
    # Usamos o mesmo random_state=42 para garantir que o conjunto de teste 
    # é matematicamente idêntico ao do dia do treino.
    
    _, temp_paths, _, temp_labels = train_test_split(
        paths, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    _, test_paths, _, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )
    
    return test_paths, test_labels

def evaluate_test_set():
    print(f"A usar o dispositivo: {DEVICE}")
    print(f"A procurar o modelo em: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Certifique-se de que o treino terminou e guardou o modelo.")

    # 1. Preparar os Dados
    test_paths, test_labels = get_test_data()
    print(f"Total de Patches no Conjunto de Teste: {len(test_paths)}")
    
    test_dataset = MammogramPatchDataset(test_paths, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Inicializar o Modelo e Carregar Pesos
    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, in_chans=1, num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    todas_preds_test = []
    todos_labels_test = []
    
    print("\n" + "="*50)
    print("🚀 A iniciar avaliação no Conjunto de Teste Cego...")
    
    # 3. Fazer as Inferências
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="A avaliar Teste"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            todas_preds_test.extend(probs)
            todos_labels_test.extend(labels.numpy())
            
    # 4. Calcular Métricas
    todos_labels_test = np.array(todos_labels_test)
    todas_preds_test = np.array(todas_preds_test)
    preds_binarias_test = (todas_preds_test > 0.5).astype(int)
    
    auc_test = roc_auc_score(todos_labels_test, todas_preds_test)
    mcc_test = matthews_corrcoef(todos_labels_test, preds_binarias_test)
    acc_test = accuracy_score(todos_labels_test, preds_binarias_test)
    sens_test = recall_score(todos_labels_test, preds_binarias_test, zero_division=0)
    prec_test = precision_score(todos_labels_test, preds_binarias_test, zero_division=0)
    f1_test = f1_score(todos_labels_test, preds_binarias_test, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(todos_labels_test, preds_binarias_test).ravel()
    spec_test = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 5. Apresentar os Resultados
    print("\n🏆 RESULTADOS DEFINITIVOS DO MODELO DE PATCHES (TESTE CEGO) 🏆")
    print("="*65)
    print(f"AUC (Área sob a Curva):      {auc_test:.4f}")
    print(f"MCC (Coef. de Matthews):     {mcc_test:.4f}")
    print(f"Acurácia Global:             {acc_test:.4f}")
    print(f"Sensibilidade (Recall):      {sens_test:.4f}  (Verdadeiros Positivos: {tp}/{tp+fn})")
    print(f"Especificidade:              {spec_test:.4f}  (Verdadeiros Negativos: {tn}/{tn+fp})")
    print(f"Precisão:                    {prec_test:.4f}")
    print(f"F1-Score:                    {f1_test:.4f}")
    print("="*65)
    
    print("\nMatriz de Confusão:")
    print(f"[{tn}] Normais identificados corretamente (TN)")
    print(f"[{fp}] Normais dados como Anormais (Falsos Positivos)")
    print(f"[{fn}] Anormais dados como Normais (Falsos Negativos)")
    print(f"[{tp}] Anormais identificados corretamente (TP)")
    print("="*65)

if __name__ == "__main__":
    evaluate_test_set()