# src/checkpointing.py
"""Early stopping + salvamento do melhor checkpoint, com base na perda de
validação suavizada."""

import torch
import wandb


class EarlyStoppingCheckpoint:
    def __init__(self, model_path, patience):
        self.model_path = model_path
        self.patience = patience
        self.melhor_valid_loss = float('inf')
        self.melhor_mcc_no_ponto_salvo = -1.0
        self.epocas_sem_melhora = 0

    def reset_patience(self):
        """
        # ================= NOVA ALTERAÇÃO: FOLGA PARA O EARLY STOPPING PÓS-DESTRAVAMENTO =================
        # Destravar os backbones é uma perturbação grande no treino (o regime de perda muda
        # de patamar). Sem isso, o "solavanco" natural dessa transição ficava preso na janela
        # de suavização por várias épocas, consumindo a paciência do early stopping bem no
        # momento em que o fine-tuning de verdade estava começando. Zeramos o contador de
        # paciência (dando um fôlego justo pro modelo provar que o fine-tuning completo ajuda).
        # IMPORTANTE: melhor_valid_loss NÃO é resetado — o modelo só é salvo se realmente
        # superar o melhor resultado histórico, então essa mudança não afrouxa o critério de
        # qualidade do checkpoint, só dá tempo justo para tentar alcançá-lo/superá-lo.
        # ============================================================================================================
        """
        self.epocas_sem_melhora = 0

    def step(self, valid_loss_suavizada, val_mcc, model):
        """Avalia a época atual, salva o checkpoint se for o melhor resultado
        até agora, e retorna True se o early stopping deve interromper o treino."""
        melhorou = valid_loss_suavizada < self.melhor_valid_loss

        if melhorou:
            self.melhor_valid_loss = valid_loss_suavizada
            self.melhor_mcc_no_ponto_salvo = val_mcc
            torch.save(model.state_dict(), self.model_path)
            self.epocas_sem_melhora = 0
            print(f"✅ Novo melhor modelo guardado! "
                  f"(Valid Loss suavizada: {valid_loss_suavizada:.4f} | MCC: {val_mcc:.4f})")
            wandb.save(self.model_path)
        else:
            self.epocas_sem_melhora += 1
            print(f"⏳ Sem melhora na perda de validação há {self.epocas_sem_melhora} "
                  f"época(s). (Melhor: {self.melhor_valid_loss:.4f})")

        if self.epocas_sem_melhora >= self.patience:
            print(f"\n🛑 Early stopping ativado (sem melhora por {self.patience} épocas seguidas).")
            return True
        return False