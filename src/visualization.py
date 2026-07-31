# src/visualization.py
"""Geração de artefatos visuais de diagnóstico: matriz de confusão e Grad-CAM."""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, epoch, output_dir="plots"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
    plt.title(f'Matriz de Confusão - Época {epoch+1}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição do Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_epoch_{epoch+1}.png'))
    plt.close()


class _SingleViewWrapper(nn.Module):
    """Isola uma única vista (cc/mlo) do modelo dual-view para o Grad-CAM,
    mantendo a outra vista fixa como contexto."""

    def __init__(self, model, view_type, img_cc, img_mlo, density):
        super().__init__()
        self.model = model
        self.view_type = view_type
        self.img_cc = img_cc
        self.img_mlo = img_mlo
        self.density = density

    def forward(self, x):
        if self.view_type == 'cc':
            out_cc, _ = self.model(x, self.img_mlo, self.density)
            return out_cc
        else:
            _, out_mlo = self.model(self.img_cc, x, self.density)
            return out_mlo


def plot_gradcam(model, valid_dataset, device, predicted_label, epoch, idx, output_dir="plots"):
    model.eval()

    # Desativa o checkpointing apenas para a geração do mapa de calor
    model.backbone_cc.set_grad_checkpointing(enable=False)
    model.backbone_mlo.set_grad_checkpointing(enable=False)
    try:
        _plot_gradcam_interno(model, valid_dataset, device, predicted_label, epoch, idx, output_dir)
    finally:
        # Reativa para a próxima época de treino
        model.backbone_cc.set_grad_checkpointing(enable=True)
        model.backbone_mlo.set_grad_checkpointing(enable=True)


def _compute_gradcam_heatmaps(model, img_cc, img_mlo, density):
    """Roda o LayerGradCam nas duas vistas e devolve heatmaps normalizados
    já interpolados para o tamanho da imagem original."""
    model_cc = _SingleViewWrapper(model, 'cc', img_cc, img_mlo, density)
    model_mlo = _SingleViewWrapper(model, 'mlo', img_cc, img_mlo, density)

    layer_alvo_cc = model.backbone_cc.stages[-1]
    layer_alvo_mlo = model.backbone_mlo.stages[-1]

    layer_gc_cc = LayerGradCam(model_cc, layer_alvo_cc)
    layer_gc_mlo = LayerGradCam(model_mlo, layer_alvo_mlo)

    # O parâmetro relu_attributions=True aplica o ReLU da fórmula original do Grad-CAM,
    # descartando as influências negativas do gradiente
    attr_cc = layer_gc_cc.attribute(img_cc, target=0, relu_attributions=True)
    attr_mlo = layer_gc_mlo.attribute(img_mlo, target=0, relu_attributions=True)

    # O LayerGradCam retorna um tensor pequeno com as dimensões da camada alvo (ex: 12x12).
    # É obrigatório interpolar para o tamanho original da imagem (ex: 384x384).
    attr_cc = LayerAttribution.interpolate(attr_cc, img_cc.shape[2:])
    attr_mlo = LayerAttribution.interpolate(attr_mlo, img_mlo.shape[2:])

    heatmap_cc = attr_cc.squeeze().cpu().detach().numpy()
    if heatmap_cc.max() > 0:
        heatmap_cc /= heatmap_cc.max()

    heatmap_mlo = attr_mlo.squeeze().cpu().detach().numpy()
    if heatmap_mlo.max() > 0:
        heatmap_mlo /= heatmap_mlo.max()

    return heatmap_cc, heatmap_mlo


def _normalize_for_viz(img_tensor):
    viz = img_tensor.squeeze().cpu().detach().numpy()
    return (viz - viz.min()) / (viz.max() - viz.min() + 1e-8)


def _plot_gradcam_interno(model, valid_dataset, device, predicted_label, epoch, idx, output_dir="plots"):
    img_cc, img_mlo, density, label = valid_dataset[idx]

    img_cc = img_cc.unsqueeze(0).to(device).requires_grad_(True)
    img_mlo = img_mlo.unsqueeze(0).to(device).requires_grad_(True)
    density = density.unsqueeze(0).to(device)

    target_label = label.item() if isinstance(label, torch.Tensor) else label

    heatmap_cc, heatmap_mlo = _compute_gradcam_heatmaps(model, img_cc, img_mlo, density)
    viz_cc = _normalize_for_viz(img_cc)
    viz_mlo = _normalize_for_viz(img_mlo)

    str_real = "Anormal" if target_label == 1 else "Normal"
    str_previsto = "Anormal" if predicted_label == 1 else "Normal"

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(
        f'Grad-CAM (Ensemble) - Época {epoch+1} | Paciente: #{idx}\n'
        f'Rótulo Real: {str_real} | Previsto: {str_previsto}',
        fontsize=16, fontweight='bold'
    )

    axes[0].imshow(viz_cc, cmap='gray')
    axes[0].imshow(heatmap_cc, cmap='magma', alpha=0.5)
    axes[0].set_title('Vista CC (Rede CC)')
    axes[0].axis('off')

    axes[1].imshow(viz_mlo, cmap='gray')
    axes[1].imshow(heatmap_mlo, cmap='magma', alpha=0.5)
    axes[1].set_title('Vista MLO (Rede MLO)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gradcam_epoch_{epoch+1}.png'))
    plt.close()