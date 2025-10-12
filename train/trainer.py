from __future__ import annotations
import os
import time
from tqdm import tqdm
from typing import Optional, Dict
import torch
from torch.utils.data import DataLoader
from torch import nn

class Trainer:
    """
    No docstring provided.
    """

    def __init__(self, *, cfg: dict, save_dir: Optional[str]=None, resume: bool=False, resume_path: Optional[str]=None, model: nn.Module, opt, train_loader: Optional[DataLoader]=None, val_loader: Optional[DataLoader]=None, test_loader: Optional[DataLoader]=None, loss_comp: Optional[LossComputer]=None, metrics_comp: Optional[MetricsComputer]=None) -> None:
        self.cfg = cfg
        device_str = self.cfg.get('device') or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.model = model.to(self.device)
        self.opt = opt
        (self.train_loader, self.val_loader, self.test_loader) = (train_loader, val_loader, test_loader)
        self.save_dir = save_dir or os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'train.log')
        self.start_epoch = 0
        self.loss_comp = loss_comp
        self.metrics_comp = metrics_comp
        if resume:
            self._resume(resume_path)

    def fit(self, epochs: Optional[int]=None) -> None:
        """
        No docstring provided.
        """
        if epochs is None:
            epochs = int(self.cfg.get('train', {}).get('epochs', 60))
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)
            test_loss = self._run_epoch(self.test_loader, training=False)
            dt = time.time() - t0
            print(f'Epoch {epoch + 1}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f} | test {test_loss:.6f} | lr {self._current_lr():.4e} | {dt:.2f}s')
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f'\n{epoch}\t{train_loss:.6f}\t\t{val_loss:.6f}\t\t{test_loss:.6f}\t\t{self._current_lr():.4f}\t\t{dt:.2f}')
            state = {'epoch': epoch + 1, 'arch': str(self.model), 'state_dict': self.model.state_dict(), 'lossTr': train_loss, 'lossVal': val_loss, 'lossTs': test_loss, 'lr': self._current_lr(), 'opt_state': self.opt.state_dict() if hasattr(self.opt, 'state_dict') else None}
            self._save_checkpoint(state, filename='checkpoint.pth')
            if not hasattr(self, 'best_val'):
                self.best_val = float('inf')
            if val_loss < getattr(self, 'best_val'):
                self.best_val = float(val_loss)
                self._save_checkpoint(state, filename='best.pth')

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> float:
        """
        No docstring provided.
        """
        self.model.train(training)
        total_loss = 0.0
        total_tokens = 0
        metric_sums: Dict[str, float] = {}
        pbar = tqdm(loader, total=len(loader), desc='Train' if training else 'Eval', leave=False)
        for batch in pbar:
            if hasattr(batch, 'to'):
                batch = batch.to(self.device)
            out = self.model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            keep_mask = batch.trg[:, 0, :] != 0
            res = self.loss_comp(out, target=batch.trg, keep_mask=keep_mask)
            if training:
                if hasattr(self.opt, 'zero_grad'):
                    self.opt.zero_grad()
                res.loss.backward()
                self.opt.step()
            total_loss += float(res.loss.detach().cpu()) * max(1, res.norm)
            total_tokens += max(1, res.norm)
            if self.metrics_comp is not None:
                batch_metrics = self.metrics_comp(out, target=batch.trg, keep_mask=keep_mask)
                for (k, v) in batch_metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v * max(1, res.norm)
            running_avg_loss = float(total_loss / max(1, total_tokens))
            postfix: Dict[str, str] = {'loss': f'{running_avg_loss:.6f}'}
            if training:
                postfix['lr'] = f'{self._current_lr():.2e}'
            if metric_sums and total_tokens > 0:
                shown = 0
                for k in metric_sums.keys():
                    postfix[k] = f'{metric_sums[k] / total_tokens:.6f}'
                    shown += 1
                    if shown >= 2:
                        break
            pbar.set_postfix(postfix)
        if metric_sums and total_tokens > 0:
            avg_metrics = {k: v / total_tokens for (k, v) in metric_sums.items()}
            print('Metrics:', {k: round(val, 6) for (k, val) in avg_metrics.items()})
        return float(total_loss / max(1, total_tokens))

    def _resume(self, ckpt_path: Optional[str]) -> None:
        """
        No docstring provided.
        """
        ckpt = ckpt_path
        tr = self.cfg.get('train', {}) if isinstance(self.cfg, dict) else {}
        if ckpt is None and isinstance(tr, dict) and tr.get('resume'):
            ckpt = tr.get('resume_path') or os.path.join(self.save_dir, 'checkpoint.pth')
        if not ckpt:
            return
        if not os.path.isfile(ckpt):
            print(f'[Trainer] Resume checkpoint not found: {ckpt}')
            return
        state = torch.load(ckpt, map_location='cpu')
        if 'state_dict' in state:
            self.model.load_state_dict(state['state_dict'])
        if 'opt_state' in state and hasattr(self.opt, 'load_state_dict'):
            try:
                self.opt.load_state_dict(state['opt_state'])
            except Exception:
                pass
        self.start_epoch = int(state.get('epoch', 0))
        self.best_val = float(state.get('lossVal', float('inf')))
        print(f'[Trainer] Resumed from {ckpt} at epoch {self.start_epoch}')

    def _current_lr(self) -> float:
        """
        No docstring provided.
        """
        if hasattr(self.opt, 'rate'):
            try:
                return float(self.opt.rate())
            except Exception:
                return 0.0
        if hasattr(self.opt, 'param_groups') and getattr(self.opt, 'param_groups'):
            return float(self.opt.param_groups[0].get('lr', 0.0))
        return 0.0

    def _save_checkpoint(self, state: dict, filename: str='checkpoint.pth') -> None:
        """
        No docstring provided.
        """
        try:
            path = os.path.join(self.save_dir, filename)
            torch.save(state, path)
        except Exception as e:
            print(f'[Trainer] Failed to save checkpoint {filename}: {e}')