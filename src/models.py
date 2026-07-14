import torch
import torch.nn as nn
import timm
import os

class PatchClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(PatchClassifier, self).__init__()
        self.model = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=pretrained, in_chans=1, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def _carregar_pesos_backbone(backbone, pretrained_patch_path, nome_debug="backbone"):
    """
    CORREÇÃO: o checkpoint salvo pelo train_patches.py vem do PatchClassifierWithDensity,
    cujo backbone é um SUBMÓDULO chamado 'backbone'. Isso faz com que TODAS as chaves do
    state_dict salvo venham prefixadas com 'backbone.' (ex: 'backbone.stem.0.weight').

    Quando esse state_dict é carregado direto num backbone timm "solto" (sem esse prefixo),
    nenhuma chave bate e o `strict=False` ignora tudo silenciosamente — ou seja, os pesos
    pré-treinados NUNCA eram de fato transferidos, mesmo sem erro nenhum aparecer no log.

    Esta função filtra apenas as chaves que começam com 'backbone.' e remove esse prefixo,
    para que baterem corretamente com as chaves internas do timm (ex: 'stem.0.weight').
    """
    if not (pretrained_patch_path and os.path.exists(pretrained_patch_path)):
        print(f"[{nome_debug}] Treinando do zero (checkpoint não encontrado ou não informado).")
        return

    checkpoint = torch.load(pretrained_patch_path, map_location='cpu')

    # Filtra só as chaves do backbone e remove o prefixo 'backbone.'
    backbone_state = {
        k[len('backbone.'):]: v
        for k, v in checkpoint.items()
        if k.startswith('backbone.')
    }

    if len(backbone_state) == 0:
        print(f"[{nome_debug}] ⚠️ Nenhuma chave 'backbone.*' encontrada em '{pretrained_patch_path}'. "
              f"Verifique se este checkpoint é mesmo do PatchClassifierWithDensity.")
        return

    resultado = backbone.load_state_dict(backbone_state, strict=False)

    n_esperadas = len(dict(backbone.named_parameters())) + len(dict(backbone.named_buffers()))
    n_carregadas = len(backbone_state) - len(resultado.unexpected_keys)

    print(f"[{nome_debug}] Pesos do patch classifier carregados de '{pretrained_patch_path}' "
          f"({n_carregadas} tensores aplicados; {len(resultado.missing_keys)} faltando, "
          f"{len(resultado.unexpected_keys)} inesperados).")

    if len(resultado.missing_keys) > 0:
        print(f"[{nome_debug}]   missing_keys (amostra): {resultado.missing_keys[:5]}")
    if len(resultado.unexpected_keys) > 0:
        print(f"[{nome_debug}]   unexpected_keys (amostra): {resultado.unexpected_keys[:5]}")


class SingleViewClassifier(nn.Module):
    def __init__(self, patch_model_path=None):
        super(SingleViewClassifier, self).__init__()

        self.backbone = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=True, num_classes=0, in_chans=1)

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

        # ================= CORREÇÃO: carregamento de pesos via helper (remove prefixo 'backbone.') =================
        _carregar_pesos_backbone(self.backbone, pretrained_patch_path, nome_debug="DualViewClassifier.backbone")
        # ==============================================================================================================

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
    
class EnsembleDualViewClassifier(nn.Module):
    def __init__(self, pretrained_patch_path=None, dropout_rate=0.4):
        super().__init__()

        self.backbone_cc = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=True, num_classes=0, in_chans=1)
        self.backbone_mlo = timm.create_model('timm/convnext_small.in12k_ft_in1k_384', pretrained=True, num_classes=0, in_chans=1)

        # ================= CORREÇÃO: carregamento de pesos via helper (remove prefixo 'backbone.') =================
        # Antes: self.backbone_cc.load_state_dict(patch_state, strict=False) com patch_state
        # ainda contendo o prefixo 'backbone.' nas chaves -> nenhuma chave batia -> os pesos
        # pré-treinados do patch classifier NUNCA eram carregados, mesmo sem erro no log.
        _carregar_pesos_backbone(self.backbone_cc, pretrained_patch_path, nome_debug="EnsembleDualView.backbone_cc")
        _carregar_pesos_backbone(self.backbone_mlo, pretrained_patch_path, nome_debug="EnsembleDualView.backbone_mlo")
        # ==============================================================================================================

        in_channels = self.backbone_cc.num_features

        # 1. Definir ambas as camadas de Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # ================= ALTERAÇÃO DENSIDADE =================
        # 2. O canal de entrada = DOBRO da imagem + 4 posições do vetor BI-RADS
        clf_in_channels = (in_channels * 2) + 4
        # =======================================================

        # ================= ALTERAÇÃO (c): dropout subiu de 0.2 -> 0.35 (padrão dropout_rate) =================
        # Mesmo valor usado no PatchClassifierWithDensity, para manter a regularização
        # consistente entre os dois modelos e ajudar a conter o overfitting observado no log.
        self.classifier_cc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(clf_in_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        self.classifier_mlo = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(clf_in_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        # ==========================================================================================================

    def forward(self, img_cc, img_mlo, density):
        # Vista CC: Extrai, faz os dois poolings e concatena COM A DENSIDADE
        feat_cc = self.backbone_cc.forward_features(img_cc)
        avg_cc = self.flatten(self.global_avg_pool(feat_cc))
        max_cc = self.flatten(self.global_max_pool(feat_cc))
        pool_cc = torch.cat([avg_cc, max_cc, density], dim=1)
        out_cc = self.classifier_cc(pool_cc)

        # Vista MLO: Extrai, faz os dois poolings e concatena COM A DENSIDADE
        feat_mlo = self.backbone_mlo.forward_features(img_mlo)
        avg_mlo = self.flatten(self.global_avg_pool(feat_mlo))
        max_mlo = self.flatten(self.global_max_pool(feat_mlo))
        pool_mlo = torch.cat([avg_mlo, max_mlo, density], dim=1)
        out_mlo = self.classifier_mlo(pool_mlo)

        return out_cc, out_mlo