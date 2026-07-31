# src/config.py
"""Constantes de configuração do treinamento do modelo dual-view."""

FREEZE_BACKBONE_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 4
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 2
LR_PLATEAU_MIN_LR = 1e-7
VALID_LOSS_SMOOTHING_WINDOW = 3
WEIGHT_DECAY = 5e-2
BACKBONE_LR_AFTER_UNFREEZE = 5e-6
CLASSIFIER_LR = 1e-4
PATCH_CHECKPOINT_PATH = 'checkpoints/patch_classifier_convnext_density_clahe.pth'
MODEL_CHECKPOINT_PATH = 'checkpoints/best_dual_view_model_modified.pth'

# ================= NOVA ALTERAÇÃO: WARM-UP GRADUAL DE LR NO DESTRAVAMENTO DO BACKBONE =================
# Enquanto o backbone fica congelado, ele nunca recebe gradiente, então os buffers internos
# do AdamW para esses parâmetros (exp_avg, exp_avg_sq — as médias móveis que adaptam o LR por
# parâmetro) nunca são inicializados. No instante em que o backbone destrava, esses parâmetros
# começam do zero absoluto nos primeiros passos do otimizador, o que pode gerar atualizações
# mal calibradas antes das estatísticas internas se estabilizarem. Em vez de saltar direto para
# BACKBONE_LR_AFTER_UNFREEZE, o LR do backbone sobe LINEARMENTE de ~0 até esse valor ao longo
# dos primeiros passos de otimizador após o destravamento — atenuando qualquer "choque" inicial,
# independente do mecanismo exato por trás dele.
BACKBONE_UNFREEZE_WARMUP_STEPS = 100
# ============================================================================================================