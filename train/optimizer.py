from __future__ import annotations
'Training utilities: Noam learning-rate schedule and helpers.\n\nThis module provides a drop-in replacement for the original NoamOpt\nwrapper used in the codebase, plus a LambdaLR-compatible helper.\n'
from typing import Optional
import torch

class Noam:
    def __init__(self, d_model: int, factor: float, warmup: int, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.d_model = float(d_model)
        self.factor = float(factor)
        self.warmup = int(warmup)
        self._step = 0
        self._lr = 0.0

    @property
    def lr(self) -> float:
        return self._lr

    def rate(self, step: Optional[int]=None) -> float:
        s = self._step if step is None else int(step)
        s = max(1, s)
        return self.factor * self.d_model ** (-0.5) * min(s ** (-0.5), s * self.warmup ** (-1.5))

    def step(self) -> None:
        self._step += 1
        lr = self.rate()
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self._lr = lr
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool=False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {'d_model': self.d_model, 'factor': self.factor, 'warmup': self.warmup, '_step': self._step, '_lr': self._lr, 'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.d_model = float(state['d_model'])
        self.factor = float(state['factor'])
        self.warmup = int(state['warmup'])
        self._step = int(state['_step'])
        self._lr = float(state['_lr'])
        self.optimizer.load_state_dict(state['optimizer'])

def _infer_d_model(model: torch.nn.Module) -> int:
    if hasattr(model, 'embedding_size'):
        return int(getattr(model, 'embedding_size'))
    if hasattr(model, 'd_model'):
        return int(getattr(model, 'd_model'))
    try:
        emb = getattr(model, 'src_embed')
        if hasattr(emb, '_modules'):
            for m in emb._modules.values():
                if hasattr(m, 'd_model'):
                    return int(getattr(m, 'd_model'))
        if hasattr(emb, '__getitem__') and hasattr(emb[1], 'd_model'):
            return int(emb[1].d_model)
    except Exception:
        pass
    return 128

def build_noam_from_config(cfg: dict, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None) -> Noam:
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    opt_cfg = train_cfg.get('optimizer', {}) if isinstance(train_cfg, dict) else {}
    sch_cfg = train_cfg.get('scheduler', {}) if isinstance(train_cfg, dict) else {}
    d_model = _infer_d_model(model)
    factor = float(sch_cfg.get('factor', 1.0))
    warmup = int(sch_cfg.get('warmup', 400))
    if optimizer is None:
        lr = float(train_cfg.get('lr', 0.0))
        betas = tuple(opt_cfg.get('betas', (0.9, 0.98)))
        eps = float(opt_cfg.get('eps', 1e-09))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    return Noam(d_model, factor, warmup, optimizer)