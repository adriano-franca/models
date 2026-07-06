import torch
import torch.nn as nn
import timm
import os

class PatchClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(PatchClassifier, self).__init__()
        self.model = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    

class SingleViewClassifier(nn.Module):
    def __init__(self, patch_model_path=None):
        super(SingleViewClassifier, self).__init__()

        self.backbone = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=True, num_classes=0, in_chans=1)

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

        self.backbone = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=True, num_classes=0, in_chans=1)

        if pretrained_patch_path and os.path.exists(pretrained_patch_path):
            print(f"Carregando pesos pré-treinados de patches: {pretrained_patch_path}")
            patch_state = torch.load(pretrained_patch_path)
            patch_state = {k: v for k, v in patch_state.items() if 'head' not in k}
            self.backbone.load_state_dict(patch_state, strict=False)
            print("Pesos carregados")
        else:
            print("Treinando backbone do zero")

        in_channels = self.backbone.num_features

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_channels * 2, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, img_cc, img_mlo):
        feat_cc = self.backbone.forward_features(img_cc)
        feat_mlo = self.backbone.forward_features(img_mlo)

        pooled_cc = self.global_pool(feat_cc)
        pooled_mlo = self.global_pool(feat_mlo)

        flat_cc = self.flatten(pooled_cc)
        flat_mlo = self.flatten(pooled_mlo)

        concat_features = torch.cat((flat_cc, flat_mlo), dim=1)

        out = self.classifier(concat_features)

        return out