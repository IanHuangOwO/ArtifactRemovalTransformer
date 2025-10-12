from __future__ import annotations
from typing import Callable, Dict, Iterable, Optional
import math
import torch

def _to_btC(t: torch.Tensor) -> torch.Tensor:
    """
    No docstring provided.
    """
    if t.dim() == 3 and t.size(1) < t.size(2):
        return t.permute(0, 2, 1)
    return t

def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    No docstring provided.
    """
    if mask is None:
        return x.mean()
    m = mask.unsqueeze(-1).expand_as(x)
    sel = x.masked_select(m)
    return sel.mean() if sel.numel() > 0 else x.new_tensor(0.0)

def mse(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None) -> float:
    """
    No docstring provided.
    """
    (x, y) = (_to_btC(x), _to_btC(y))
    return float(_masked_mean((x - y).pow(2), mask).detach().cpu())

def mae(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None) -> float:
    """
    No docstring provided.
    """
    (x, y) = (_to_btC(x), _to_btC(y))
    return float(_masked_mean((x - y).abs(), mask).detach().cpu())

def rmse(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None) -> float:
    """
    No docstring provided.
    """
    return math.sqrt(max(mse(x, y, mask), 0.0))

def corr(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None) -> float:
    """
    No docstring provided.
    """
    (x, y) = (_to_btC(x), _to_btC(y))
    if mask is not None:
        m = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_select(m)
        y = y.masked_select(m)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.std() + 1e-10) * (y.std() + 1e-10)
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).mean().detach().cpu() / denom)

def r2(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None) -> float:
    """
    No docstring provided.
    """
    (x, y) = (_to_btC(x), _to_btC(y))
    if mask is not None:
        m = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_select(m)
        y = y.masked_select(m)
    ss_res = torch.sum((y - x) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2) + 1e-10
    return float(1.0 - (ss_res / ss_tot).detach().cpu())

_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], float]] = {'mse': mse, 'mae': mae, 'rmse': rmse, 'corr': corr, 'r2': r2}

class MetricsComputer:
    """
    No docstring provided.
    """

    def __init__(self, names: list[str]) -> None:
        self.names = names

    def __call__(self, out: torch.Tensor, *, target: torch.Tensor, keep_mask: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        No docstring provided.
        """
        results: Dict[str, float] = {}
        for name in self.names:
            fn = _REGISTRY[name]
            results[name] = fn(out, target, keep_mask)
        return results

def build_metrics_from_config(cfg: dict, defaults: Iterable[str]=('mse', 'mae')) -> MetricsComputer:
    """
    Builds a MetricsComputer from a configuration dictionary.
    """
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    names = train_cfg.get('metrics') if isinstance(train_cfg, dict) else None
    if not names:
        names = list(defaults)
    
    metric_names: list[str] = [str(n).lower() for n in names if str(n).lower() in _REGISTRY]
    if not metric_names:
        metric_names = list(defaults)
    return MetricsComputer(metric_names)