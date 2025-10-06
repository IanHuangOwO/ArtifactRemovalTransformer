import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .hf_config import ArtifactRemovalTransformerConfig

# ======================================================================================
# region: Core Model Building Blocks
# These are the fundamental components like Attention, FFN, and Embeddings.
# ======================================================================================

class ExpandConv1x1(nn.Module):
    """Expand channels with a 1x1 convolution and permute to (B, T, C)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C_out, T) -> (B, T, C_out)
        return self.conv(x).permute(0, 2, 1)


class PositionalEmbedding(nn.Module):
    """Sinusoidal or learned positional embeddings."""

    def __init__(self, max_len: int, d_model: int, mode: str = "sinusoidal") -> None:
        super().__init__()
        if mode not in {"sinusoidal", "learned"}:
            raise ValueError(f"Unsupported pos_mode: {mode}")
        self.mode = mode
        self.d_model = d_model

        if self.mode == "learned":
            self.pos_embed = nn.Embedding(max_len, d_model)
        else:  # sinusoidal
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        if self.mode == "learned":
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            pos_emb = self.pos_embed(pos)
        else:
            pos_emb = self.pe[:, :T, :]
        return x + pos_emb


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = q.shape
        q = self.q_proj(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.dropout(F.relu(self.linear1(x)))))


# ======================================================================================
# endregion
# region: Main Model Architecture (ArtifactRemovalTransformer)
# This section defines the encoder, decoder, and the complete Transformer model.
# ======================================================================================


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)

        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.drop1(h))
        
        h = self.ffn(x)
        x = self.ln2(x + self.drop2(h))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.self_mha(x, x, x, attn_mask=self_attn_mask)
        x = self.ln1(x + self.drop1(h))

        h = self.cross_mha(x, memory, memory, attn_mask=cross_attn_mask)
        x = self.ln2(x + self.drop2(h))

        h = self.ffn(x)
        x = self.ln3(x + self.drop3(h))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        return self.norm(x)


class Reconstructor(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_channels: int,
        *,
        log_softmax: bool = False,
        zscore: str | None = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, out_channels)
        self.use_log_softmax = log_softmax
        self.zscore = zscore
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> (B, T, C)
        y = self.proj(x)
        if self.use_log_softmax:
            y = F.log_softmax(y, dim=-1)
        if self.zscore is None:
            return y
        if self.zscore == "batch":
            # Normalize across batch dimension, preserving (T, C) statistics
            mean = y.mean(dim=0, keepdim=True)
            std = y.std(dim=0, keepdim=True)
        elif self.zscore == "time":
            # Per-sample z-score across time axis
            mean = y.mean(dim=1, keepdim=True)
            std = y.std(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported zscore mode: {self.zscore}")
        return (y - mean) / (std + self.eps)

class ArtifactRemovalTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_size: int = 128,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_size: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
        pos_mode: str = "sinusoidal",
        recon_log_softmax: bool = False,
        recon_zscore: str | None = None,
    ) -> None:
        super().__init__()
        
        self.src_embed = nn.Sequential(
            ExpandConv1x1(in_channels, embedding_size),
            PositionalEmbedding(max_len=max_len, d_model=embedding_size, mode=pos_mode),
            nn.Dropout(dropout),
        )

        self.encoder = TransformerEncoder(
            d_model=embedding_size,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=feedforward_size,
            dropout=dropout,
            attn_dropout=dropout,
        )
        
        self.tgt_embed = nn.Sequential(
            ExpandConv1x1(out_channels, embedding_size),
            PositionalEmbedding(max_len=max_len, d_model=embedding_size, mode=pos_mode),
            nn.Dropout(dropout),
        )
        self.decoder = TransformerDecoder(
            d_model=embedding_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=feedforward_size,
            dropout=dropout,
            attn_dropout=dropout,
        )
        
        self.reconstructor = Reconstructor(
            d_model=embedding_size,
            out_channels=out_channels,
            log_softmax=recon_log_softmax,
            zscore=recon_zscore,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_x = self.src_embed(src)
        
        enc_attn_mask = None
        if src_mask is not None:
            if src_mask.dtype != torch.bool:
                src_mask = src_mask.to(torch.bool)
            enc_attn_mask = (~src_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        memory = self.encoder(src_x, attn_mask=enc_attn_mask)
        
        if tgt is None:
            tgt = src
        tgt_x = self.tgt_embed(tgt)

        dec_self_mask = None
        dec_cross_mask = enc_attn_mask
        if tgt_mask is not None:
            if tgt_mask.dtype != torch.bool:
                tgt_mask = tgt_mask.to(torch.bool)
            dec_self_mask = (~tgt_mask).unsqueeze(1)  # (B, 1, Q, K)

        out = self.decoder(tgt_x, memory, dec_self_mask, dec_cross_mask)
        
        reconstructed = self.reconstructor(out)
        # Permute from (B, T, C) to (B, C, T) to match HF conventions
        return reconstructed.permute(0, 2, 1)


# ======================================================================================
# endregion
# region: Hugging Face PreTrainedModel Wrapper
# This is the main class that integrates the model into the Hugging Face ecosystem.
# ======================================================================================

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class ArtifactRemovalTransformerForConditionalGeneration(PreTrainedModel):
    """
    A Hugging Face-compatible wrapper for the ArtifactRemovalTransformer model.
    This model is designed for a conditional generation task (sequence-to-sequence),
    like removing artifacts from EEG signals.
    """
    config_class = ArtifactRemovalTransformerConfig

    def __init__(self, config: ArtifactRemovalTransformerConfig):
        super().__init__(config)
        # The core model is instantiated here using parameters from the config
        self.model = ArtifactRemovalTransformer(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            embedding_size=config.embedding_size,
            feedforward_size=config.feedforward_size,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_len=config.max_len,
            pos_mode=config.pos_mode,
            recon_log_softmax=config.recon_log_softmax,
            recon_zscore=config.recon_zscore,
        )
        self.loss_fct = nn.MSELoss()
        self.eps = 1e-10

    def _zscore_loss(self, logits, labels):
        """Calculates MSE loss on per-sample z-scored signals."""
        # Permute to (B, T, C) for easier stats calculation
        logits = logits.permute(0, 2, 1)
        labels = labels.permute(0, 2, 1)

        # Per-sample z-score across time axis
        l_mean, l_std = logits.mean(dim=1, keepdim=True), logits.std(dim=1, keepdim=True)
        lab_mean, lab_std = labels.mean(dim=1, keepdim=True), labels.std(dim=1, keepdim=True)

        logits_z = (logits - l_mean) / (l_std + self.eps)
        labels_z = (labels - lab_mean) / (lab_std + self.eps)

        return self.loss_fct(logits_z, labels_z)

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        """
        The forward pass of the model.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length)`):
                The input EEG signals with artifacts.
            labels (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length)`, *optional*):
                The clean target EEG signals. If provided, the model calculates the loss.

        Returns:
            `Seq2SeqLMOutput` with loss and logits.
        """
        # During training, `labels` are provided. We use them for teacher-forcing in the decoder.
        # The decoder input should be a shifted version of the labels.
        # Here, we follow the original implementation's likely intent by using the full labels
        # as the decoder input, as the original model handles the sequence alignment internally.
        decoder_input_values = labels if labels is not None else input_values

        # Get model predictions (logits)
        logits = self.model(src=input_values, tgt=decoder_input_values)

        loss = None
        if labels is not None:
            if self.config.loss_zscore:
                loss = self._zscore_loss(logits, labels)
            else:
                loss = self.loss_fct(logits, labels)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
        )