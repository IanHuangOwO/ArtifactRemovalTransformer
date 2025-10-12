from dataclasses import dataclass
from typing import Dict, Optional
import torch
from train.metrics import mse as metric_mse, mae as metric_mae

def _to_btC(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3 and t.size(1) < t.size(2):
        return t.permute(0, 2, 1)
    return t

def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.unsqueeze(-1).expand_as(x)
    sel = x.masked_select(m)
    return sel.mean() if sel.numel() > 0 else x.new_tensor(0.0)

@dataclass
class LossResult:
    loss: torch.Tensor
    logs: Dict[str, float]
    norm: int

class LossComputer:
    def __init__(self, kind: str, zscore: bool) -> None:
        self.kind = kind
        self.zscore = zscore
        self.eps = 1e-10

    def __call__(self, out: torch.Tensor, *, target: torch.Tensor, keep_mask: Optional[torch.Tensor]) -> LossResult:
        y = _to_btC(target)
        x = _to_btC(out)
        if self.zscore:
            (xm, xs) = (x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True))
            (ym, ys) = (y.mean(dim=1, keepdim=True), y.std(dim=1, keepdim=True))
            x = (x - xm) / (xs + self.eps)
            y = (y - ym) / (ys + self.eps)
        if self.kind == 'mae':
            main = _masked_mean((x - y).abs(), keep_mask)
        else:
            main = _masked_mean((x - y).pow(2), keep_mask)
        mae_val = metric_mae(out, target, keep_mask)
        mse_val = metric_mse(out, target, keep_mask)
        logs = {'loss': float(main.detach().cpu()), 'mse': mse_val, 'mae': mae_val}
        if keep_mask is not None:
            norm = int(keep_mask.sum().item() * y.size(-1))
        else:
            norm = int(y.numel())
        return LossResult(main, logs, norm)

def build_loss_from_config(cfg: dict) -> LossComputer:
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    loss_cfg = train_cfg.get('loss', {}) if isinstance(train_cfg, dict) else {}
    
    kind = str(loss_cfg.get('type', 'mse')).lower()
    if kind not in {'mse', 'mae'}:
        kind = 'mse'
    zscore = bool(loss_cfg.get('zscore', True))
    return LossComputer(kind=kind, zscore=zscore)