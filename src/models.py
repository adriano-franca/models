import torch
import torch.nn as nn
import timm
import os

class PatchClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(PatchClassifier, self).__init__()
        self.model = timm.create_model('convnext_base_in22k', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    

class SingleViewClassifier(nn.Module):
    def __init__(self, patch_model_path=None):
        super(SingleViewClassifier, self).__init__()

        self.backbone = timm.create_model('convnext_base_in22k', pretrained=False, num_classes=0, in_chans=1)

        # TODO: Lógica para carregar os pesos de `patch_model_path` (se fornecido)
        # self.backbone.load_state_dict(torch.load(patch_model_path), strict=False)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 1)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.global_pool(features)
        flattened = self.flatten(pooled)
        out = self.classifier(flattened)
        return out
    
class DualViewClassifier(nn.Module):
    def __init__(self, pretrained_patch_path=None):
        super().__init__()

        self.backbone = timm.create_model('convnext_base_in22k', pretrained=False, num_classes=0, in_chans=1)

        '''# TODO: Lógica para carregar os pesos de `single_view_model_path` 
        # self.backbone.load_state_dict(torch.load(single_view_model_path), strict=False)'''

        if pretrained_patch_path and os.path.exists(pretrained_patch_path):
            print(f"Carregando pesos pré-treinados de patches: {pretrained_patch_path}")
            patch_state = torch.load(pretrained_patch_path)

            patch_state = {k: v for k, v in patch_state.items() if 'head' not in k}

            self.backbone.load_state_dict(patch_state, strict=False)
            print("Pesos carregados")
        else:
            print("Treinando backbone do zero")

        in_channels = self.backbone.num_features

        self.reducer = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_channels, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, img_cc, img_mlo):

        feat_cc = self.backbone.forward_features(img_cc)
        feat_mlo = self.backbone.forward_features(img_mlo)

        concat_features = torch.cat((feat_cc, feat_mlo), dim=1)

        reduced = self.reducer(concat_features)
        pooled = self.global_pool(reduced)
        flattened = self.flatten(pooled)

        out = self.classifier(flattened)

        return out
    
class DualViewDenseNet(nn.Module):
    def __init__(self, pretrained_patch_path=None):
        super(DualViewDenseNet, self).__init__()
        
        # Cria a base sem pesos do ImageNet (porque vamos carregar os seus!)
        self.backbone = timm.create_model('densenet121', pretrained=False, num_classes=0)
        
        # --- CARREGAR PESOS DOS PATCHES ---
        if pretrained_patch_path and os.path.exists(pretrained_patch_path):
            print(f"🧠 Carregando pesos pré-treinados de patches na DenseNet: {pretrained_patch_path}")
            state_dict = torch.load(pretrained_patch_path)
            
            # Filtra apenas os pesos que pertencem ao backbone (ignorando o classificador final antigo)
            backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
            
            # Se o seu modelo de patches não usava a palavra 'backbone', descomente a linha abaixo e comente a de cima
            # backbone_state_dict = {k: v for k, v in state_dict.items() if k in self.backbone.state_dict()}
            
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print("✅ Pesos da DenseNet carregados com sucesso!")
        
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, img_cc, img_mlo):
        # Se a imagem for em escala de cinzentos (1 canal), repetimos o canal 3 vezes
        if img_cc.shape[1] == 1:
            img_cc = img_cc.repeat(1, 3, 1, 1)
        if img_mlo.shape[1] == 1:
            img_mlo = img_mlo.repeat(1, 3, 1, 1)

        features_cc = self.backbone.forward_features(img_cc)
        features_cc = self.pooling(features_cc).flatten(1)
        
        features_mlo = self.backbone.forward_features(img_mlo)
        features_mlo = self.pooling(features_mlo).flatten(1)
        
        combined = torch.cat((features_cc, features_mlo), dim=1)
        out = self.classifier(combined)
        return out
    
class DualViewEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DualViewEfficientNet, self).__init__()
        # Usamos a versão 's' (small) para não estourar a memória da GPU com imagens grandes
        self.backbone = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=pretrained, num_classes=0)
        
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        
        # A EfficientNetV2-S cospe tensores de 1280. Para 2 vistas: 1280 * 2 = 2560
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, img_cc, img_mlo):
        features_cc = self.backbone.forward_features(img_cc)
        features_cc = self.pooling(features_cc).flatten(1)
        
        features_mlo = self.backbone.forward_features(img_mlo)
        features_mlo = self.pooling(features_mlo).flatten(1)
        
        combined = torch.cat((features_cc, features_mlo), dim=1)
        out = self.classifier(combined)
        return out