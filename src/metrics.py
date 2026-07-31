# src/metrics.py
"""Cálculo de limiar ótimo e métricas de validação."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(labels, probs):
    """Varre limiares de decisão e devolve o que maximiza o MCC."""
    best_mcc = -1
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.91, 0.01):
        preds = (np.array(probs) > thresh).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh
    return best_thresh, best_mcc


def compute_validation_metrics(all_labels, all_probs):
    """
    Calcula limiar ótimo + métricas de validação a partir dos rótulos/probs
    acumulados na época.

    # ================= NOVA ALTERAÇÃO: PROTEÇÃO CONTRA ARRAYS VAZIOS =================
    # Se toda a validação foi não-finita (arrays vazios), find_best_threshold()
    # e as métricas do sklearn quebrariam com ValueError ("Found empty array"). Em vez de
    # deixar o treino inteiro morrer por causa de uma única época ruim, retornamos métricas
    # degeneradas (zeradas) e deixamos o early stopping/checkpoint lidarem normalmente
    # com essa época — o treino continua para a próxima.
    # ============================================================================================================

    Retorna um dict com: limiar, all_preds, auc, mcc, acc, prec, sens, f1, spec.
    """
    if len(all_labels) == 0:
        return {
            "limiar": 0.5,
            "all_preds": np.array([]),
            "auc": 0.0, "mcc": 0.0,
            "acc": 0.0, "prec": 0.0, "sens": 0.0, "f1": 0.0, "spec": 0.0,
        }

    limiar, _ = find_best_threshold(all_labels, all_probs)
    all_preds = (all_probs >= limiar).astype(int)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        sens = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        auc, mcc, acc, prec, sens, f1, spec = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "limiar": limiar, "all_preds": all_preds,
        "auc": auc, "mcc": mcc, "acc": acc,
        "prec": prec, "sens": sens, "f1": f1, "spec": spec,
    }