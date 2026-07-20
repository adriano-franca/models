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
    if not (pretrained_patch_path and os.path.exists(pretrained_patch_path)):
        print(f"[{nome_debug}] Treinando do zero (checkpoint não encontrado ou não informado).")
        return

    checkpoint = torch.load(pretrained_patch_path, map_location='cpu')

    backbone_state = {
        k[len('backbone.'):]: v
        for k, v in checkpoint.items()
        if k.startswith('backbone.')
    }

    if len(backbone_state) == 0:
        print(f"[{nome_debug}] ⚠️ Nenhuma chave 'backbone.*' encontrada em '{pretrained_patch_path}'.")
        return

    resultado = backbone.load_state_dict(backbone_state, strict=False)
    n_carregadas = len(backbone_state) - len(resultado.unexpected_keys)

    print(f"[{nome_debug}] Pesos do patch classifier carregados de '{pretrained_patch_path}' "
          f"({n_carregadas} tensores aplicados; {len(resultado.missing_keys)} faltando, "
          f"{len(resultado.unexpected_keys)} inesperados).")


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
        _carregar_pesos_backbone(self.backbone, pretrained_patch_path, nome_debug="DualViewClassifier.backbone")
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

        self.backbone_cc.set_grad_checkpointing(enable=True)
        self.backbone_mlo.set_grad_checkpointing(enable=True)

        _carregar_pesos_backbone(self.backbone_cc, pretrained_patch_path, nome_debug="EnsembleDualView.backbone_cc")
        _carregar_pesos_backbone(self.backbone_mlo, pretrained_patch_path, nome_debug="EnsembleDualView.backbone_mlo")

        in_channels = self.backbone_cc.num_features

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.density_layer = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU()
        )

        # ================= REVERSÃO: LATE FUSION (uma cabeça por backbone) =================
        # Cada vista tem sua própria cabeça de classificação, recebendo só a sua própria
        # representação visual (avg pooling do respectivo backbone) + a densidade (compartilhada
        # entre as duas, já que é a mesma informação clínica do exame). Não há mais concatenação
        # das duas vistas antes da classificação — a fusão acontece só depois, na média das
        # probabilidades (ver forward()).
        clf_in_channels = in_channels + 16

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
        # ============================================================================================

    def forward(self, img_cc, img_mlo, density):
        dense_feat = self.density_layer(density)

        # Vista CC: features próprias + densidade -> cabeça própria -> logit próprio
        feat_cc = self.backbone_cc.forward_features(img_cc)
        avg_cc = self.flatten(self.global_avg_pool(feat_cc))
        pool_cc = torch.cat([avg_cc, dense_feat], dim=1)
        out_cc = self.classifier_cc(pool_cc)

        # Vista MLO: features próprias + densidade -> cabeça própria -> logit próprio
        feat_mlo = self.backbone_mlo.forward_features(img_mlo)
        avg_mlo = self.flatten(self.global_avg_pool(feat_mlo))
        pool_mlo = torch.cat([avg_mlo, dense_feat], dim=1)
        out_mlo = self.classifier_mlo(pool_mlo)

        return out_cc, out_mlo