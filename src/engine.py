# src/engine.py
"""Loops de treino e validação de uma única época."""

import numpy as np
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                     epoch, freeze_backbone_epochs, accumulation_steps, warmup):
    """Roda uma época de treino. `warmup` é um BackboneUnfreezeWarmup (ou None-like,
    já que sua própria classe controla o estado ativo/inativo internamente).

    Retorna (avg_train_loss, batches_nao_finitos_treino).
    """
    model.train()
    train_loss = 0.0
    train_batches_validos = 0  # conta só os batches que contribuíram de fato
    batches_nao_finitos_treino = 0
    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(loader, desc="Treino")
    for batch_idx, (img_cc, img_mlo, density, labels) in enumerate(loop):
        img_cc, img_mlo, density = img_cc.to(device), img_mlo.to(device), density.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Ativa o gradient checkpointing apenas após o warm-up de congelamento
        if epoch >= freeze_backbone_epochs:
            img_cc.requires_grad_(True)
            img_mlo.requires_grad_(True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_cc, out_mlo = model(img_cc, img_mlo, density)
            loss_cc = criterion(out_cc, labels)
            loss_mlo = criterion(out_mlo, labels)
            loss = (loss_cc + loss_mlo) / 2.0

        if not torch.isfinite(loss):
            batches_nao_finitos_treino += 1
            print(f"⚠️ Loss não-finita no batch de treino {batch_idx} (época {epoch+1}). Batch ignorado.")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_para_backward = loss / accumulation_steps
        scaler.scale(loss_para_backward).backward()

        is_optimizer_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader))
        if is_optimizer_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            warmup.step(optimizer)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        train_batches_validos += 1
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / max(train_batches_validos, 1)
    if batches_nao_finitos_treino > 0:
        print(f"⚠️ Total de batches de treino ignorados por loss não-finita nesta época: {batches_nao_finitos_treino}")

    return avg_train_loss, batches_nao_finitos_treino


def validate_one_epoch(model, loader, criterion, device, epoch, log_sample_probs=False):
    """Roda uma época de validação.

    Retorna (avg_valid_loss, all_labels, all_probs, batches_nao_finitos_valid).

    # ================= NOVA ALTERAÇÃO: PROTEÇÃO CONTRA "MELHOR ÉPOCA FALSA" =================
    # Se TODOS os batches de validação forem não-finitos, valid_loss e valid_batches_validos
    # ficam ambos em 0, e 0.0/max(0,1) = 0.0 -- o que pareceria a MELHOR perda possível e faria
    # o checkpoint (quebrado) ser salvo como "novo melhor modelo". Em vez disso, tratamos uma
    # validação inteiramente não-finita como o PIOR resultado possível (infinito), garantindo
    # que essa época nunca seja escolhida como checkpoint nem reinicie o contador de paciência.
    # ============================================================================================================
    """
    model.eval()
    valid_loss = 0.0
    valid_batches_validos = 0
    batches_nao_finitos_valid = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        loop_val = tqdm(loader, desc="Validação")
        for batch_idx, (img_cc, img_mlo, density, labels) in enumerate(loop_val):
            img_cc, img_mlo, density = img_cc.to(device), img_mlo.to(device), density.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # LATE FUSION
                out_cc, out_mlo = model(img_cc, img_mlo, density)
                loss_cc = criterion(out_cc, labels)
                loss_mlo = criterion(out_mlo, labels)
                loss = (loss_cc + loss_mlo) / 2.0

                # ENSEMBLE: sigmoide de cada vista, seguido da média aritmética simples
                prob_cc = torch.sigmoid(out_cc)
                prob_mlo = torch.sigmoid(out_mlo)
                prob_ensemble = (prob_cc + prob_mlo) / 2.0

                if log_sample_probs and batch_idx == 0:
                    print(f"Probabilidades (amostra): {prob_ensemble.flatten().detach().cpu().numpy()[:5]}")

            if not torch.isfinite(loss):
                batches_nao_finitos_valid += 1
                print(f"⚠️ Loss não-finita num batch de validação (época {epoch+1}). Batch ignorado.")
                continue

            valid_loss += loss.item()
            valid_batches_validos += 1
            probs = prob_ensemble.float().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.float().cpu().numpy())

    if valid_batches_validos == 0:
        avg_valid_loss = float('inf')
        print(f"🛑 TODA a validação desta época foi não-finita ({batches_nao_finitos_valid} "
              f"batches). Tratando como pior resultado possível (não será salva como melhor).")
    else:
        avg_valid_loss = valid_loss / valid_batches_validos

    if batches_nao_finitos_valid > 0:
        print(f"⚠️ Total de batches de validação ignorados por loss não-finita nesta época: {batches_nao_finitos_valid}")

    return avg_valid_loss, np.array(all_labels), np.array(all_probs), batches_nao_finitos_valid