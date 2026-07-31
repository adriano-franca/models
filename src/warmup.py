# src/warmup.py
"""Warm-up gradual de LR no destravamento do backbone.

Enquanto o backbone fica congelado, ele nunca recebe gradiente, então os buffers
internos do AdamW para esses parâmetros (exp_avg, exp_avg_sq) nunca são
inicializados. No instante em que o backbone destrava, esses parâmetros começam
do zero absoluto nos primeiros passos do otimizador, o que pode gerar
atualizações mal calibradas antes das estatísticas internas se estabilizarem.
Em vez de saltar direto para o LR alvo, o LR sobe LINEARMENTE de ~0 até esse
valor ao longo dos primeiros `total_steps` passos de otimizador após o
destravamento.
"""


class BackboneUnfreezeWarmup:
    def __init__(self, total_steps, backbone_lr_target, classifier_lr_target,
                 backbone_group_idxs=(0, 1), classifier_group_idxs=(2, 3)):
        self.total_steps = total_steps
        self.backbone_lr_target = backbone_lr_target
        self.classifier_lr_target = classifier_lr_target
        self.backbone_group_idxs = backbone_group_idxs
        self.classifier_group_idxs = classifier_group_idxs
        self._step_count = None  # None = warm-up inativo

    @property
    def is_active(self):
        return self._step_count is not None

    def activate(self, optimizer):
        """Chamado no exato instante em que o backbone é destravado."""
        self._step_count = 0
        lr_inicial = self.backbone_lr_target / self.total_steps
        for idx in self.backbone_group_idxs:
            optimizer.param_groups[idx]['lr'] = lr_inicial
        print(f"   ↳ Warm-up gradual de LR do backbone ativado "
              f"({self.total_steps} passos até {self.backbone_lr_target:.2e}).")

    def step(self, optimizer):
        """Chamado a cada passo de otimizador (após acumulação de gradiente).
        Não faz nada se o warm-up estiver inativo."""
        if self._step_count is None:
            return

        if self._step_count < self.total_steps:
            progresso = (self._step_count + 1) / self.total_steps
            novo_lr_backbone = self.backbone_lr_target * progresso
            novo_lr_classif = self.classifier_lr_target * progresso

            for idx in self.backbone_group_idxs:
                optimizer.param_groups[idx]['lr'] = novo_lr_backbone
            for idx in self.classifier_group_idxs:
                optimizer.param_groups[idx]['lr'] = novo_lr_classif

            self._step_count += 1
        else:
            self._step_count = None
            print(f"   [warm-up backbone] concluído — LR em {self.backbone_lr_target:.2e}.")