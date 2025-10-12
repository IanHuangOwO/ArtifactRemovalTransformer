from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
try:
    from .ART_blocks import ExpandConv1x1, PositionalEmbedding, MultiHeadAttention, FeedForward
except ImportError:
    from ART_blocks import ExpandConv1x1, PositionalEmbedding, MultiHeadAttention, FeedForward

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-05)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-05)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]=None) -> Tensor:
        h = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.drop1(h))
        h = self.ffn(x)
        x = self.ln2(x + self.drop2(h))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, attn_dropout=attn_dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-05)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]=None) -> Tensor:
        """
        Sequentially processes input tensor through multiple encoder layers with optional attention masking and final normalization.

        Summary:
        Transforms input representations by applying stacked encoder transformations and layer normalization.

        Description:
        WHY: Enables hierarchical feature extraction and contextual representation learning in transformer architectures.

        WHEN: Used during model training and inference to process sequential input data through multiple transformation stages.

        WHERE: Serves as the core forward pass mechanism in transformer encoder networks, facilitating complex representation learning.

        HOW: Iteratively applies each transformer encoder block to the input, incorporating optional attention masking, and applies final layer normalization to stabilize output representations.

        Args:
            x (Tensor): Input tensor representing sequence embeddings to be transformed.
            attn_mask (Optional[Tensor], optional): Mask tensor to prevent attending to certain positions. Defaults to None.

        Returns:
            Tensor: Transformed and normalized output tensor after passing through all encoder layers.

        Examples:
            ```python
            # Typical usage in sequence processing
            encoder = TransformerEncoder(
                d_model=512, 
                num_layers=6, 
                num_heads=8, 
                d_ff=2048
            )

            # Forward pass with optional masking
            output = encoder(
                x=input_sequence, 
                attn_mask=padding_mask
            )
            ```

        Warnings:
            - Performance depends on input quality and encoder configuration
            - Attention masking crucial for handling variable-length sequences
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)

class TransformerDecoderBlock(nn.Module):
    """
    A fundamental building block for transformer decoder architectures, responsible for contextual feature transformation and cross-attention mechanisms.

    Summary:
    Implements a single transformer decoder layer with cross-attention, residual connections, and non-linear transformations.

    Description:
    WHY: Enables sophisticated sequence-to-sequence learning by dynamically integrating contextual information through cross-attention and non-linear feature refinement.

    WHEN: Essential in neural machine translation, text generation, and sequence modeling tasks requiring complex representation learning.

    WHERE: Serves as a core component in multi-layer transformer decoder networks, facilitating hierarchical information processing between encoder and decoder representations.

    HOW: Sequentially applies multi-head cross-attention, residual connections, layer normalization, and feed-forward transformations to input embeddings.

    Parameters:
        d_model (int): Dimensionality of the model's feature space. Determines representation complexity.
        num_heads (int): Number of attention heads for multi-head attention mechanism. Enables parallel contextual feature extraction.
        d_ff (int): Dimensionality of the feed-forward network's hidden layer. Controls model's capacity for non-linear transformations.
        dropout (float, optional): Regularization rate for dropout layers. Helps prevent overfitting. Defaults to 0.0.
        attn_dropout (float, optional): Specific dropout rate for attention mechanisms. Defaults to 0.0.

    Attributes:
        ln1 (nn.LayerNorm): First layer normalization for stabilizing feature representations.
        cross_mha (MultiHeadAttention): Multi-head cross-attention mechanism for contextual feature integration.
        ffn (FeedForward): Non-linear feed-forward transformation network.
        drop1, drop2 (nn.Dropout): Dropout layers for regularization.

    Examples:
        ```python
        # Create a decoder block with specific hyperparameters
        decoder_block = TransformerDecoderBlock(
            d_model=512,      # Feature space dimensionality
            num_heads=8,      # 8 parallel attention heads
            d_ff=2048,        # Large feed-forward layer
            dropout=0.1       # 10% dropout for regularization
        )

        # Typical forward pass in a sequence generation task
        output = decoder_block(
            x=decoder_input,       # Current decoder sequence
            memory=encoder_output, # Encoder context
            self_attn_mask=mask    # Optional attention masking
        )
        ```

    Warnings:
        - Requires careful hyperparameter tuning for optimal performance
        - Performance depends on encoder-decoder context quality
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-05)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-05)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, self_attn_mask: Optional[Tensor]=None) -> Tensor:
        """
        Summary:
        Performs cross-attention and feed-forward transformation with residual connections and layer normalization.

        Description:
        WHY: Enables complex contextual representation learning by integrating cross-attention mechanisms with non-linear transformations.

        WHEN: Used in sequence-to-sequence models requiring dynamic context integration and feature refinement.

        WHERE: Serves as a fundamental building block in transformer decoder architectures, facilitating information flow between encoder and decoder representations.

        HOW: Applies multi-head cross-attention, residual connections, dropout, and layer normalization in a structured sequence to transform input representations.

        Args:
            x (Tensor): Input tensor representing decoder sequence embeddings.
            memory (Tensor): Encoder-side context tensor providing cross-attention information.
            self_attn_mask (Optional[Tensor], optional): Mask tensor to prevent attending to certain positions. Defaults to None.

        Returns:
            Tensor: Transformed output tensor after cross-attention and feed-forward processing.

        Examples:
            ```python
            # Typical usage in transformer decoder block
            decoder_block = TransformerDecoderBlock(
                d_model=512, 
                num_heads=8, 
                d_ff=2048, 
                dropout=0.1
            )

            # Forward pass with optional masking
            output = decoder_block(
                x=decoder_input, 
                memory=encoder_output, 
                self_attn_mask=subsequent_mask
            )
            ```
        """
        h = self.cross_mha(x, memory, memory, attn_mask=self_attn_mask)
        x = self.ln1(x + self.drop1(h))
        h = self.ffn(x)
        x = self.ln2(x + self.drop2(h))
        return x

class TransformerDecoder(nn.Module):
    """
    Stacked transformer decoder architecture for hierarchical sequence representation and generation.

    Summary:
    Multi-layer neural network component for advanced contextual feature transformation in sequence-to-sequence models.

    Description:
    WHY: Enables complex, hierarchical representation learning by progressively refining input sequences through multiple decoder layers with cross-attention mechanisms.

    WHEN: Critical for advanced natural language processing, machine translation, text generation, and sequence modeling tasks requiring sophisticated contextual understanding.

    WHERE: Serves as a core architectural component in transformer-based neural networks, bridging encoder representations with target sequence generation.

    HOW: Sequentially processes input through multiple transformer decoder blocks, applying cross-attention, non-linear transformations, and layer normalization to progressively extract and refine contextual features.

    Parameters:
        d_model (int): Dimensionality of the model's feature representation space. Controls representation complexity and information capacity.
        num_layers (int): Number of stacked transformer decoder layers. Determines network depth and representational power.
        num_heads (int): Number of parallel attention heads in each decoder block. Enables multi-perspective feature extraction.
        d_ff (int): Dimensionality of feed-forward network's hidden layer. Controls non-linear transformation capacity.
        dropout (float, optional): Regularization rate applied across decoder layers. Helps prevent overfitting. Defaults to 0.0.
        attn_dropout (float, optional): Specific dropout rate for attention mechanisms. Defaults to 0.0.

    Attributes:
        layers (nn.ModuleList): Sequence of transformer decoder blocks, each performing contextual feature transformation.
        norm (nn.LayerNorm): Final layer normalization to stabilize output representations.

    Examples:
        ```python
        # Initialize decoder for machine translation
        decoder = TransformerDecoder(
            d_model=512,        # Feature space dimensionality
            num_layers=6,       # 6 stacked decoder layers
            num_heads=8,        # 8 attention heads per layer
            d_ff=2048,          # Large feed-forward layers
            dropout=0.1         # 10% dropout for regularization
        )

        # Typical forward pass in sequence generation
        output = decoder(
            x=target_sequence_embeddings,  # Input sequence
            memory=encoder_context,        # Encoder representations
            self_attn_mask=generation_mask # Optional attention masking
        )
        ```

    Warnings:
        - Performance highly dependent on encoder quality and hyperparameter tuning
        - Computational complexity increases with number of layers and model dimensionality
    """

    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, attn_dropout=attn_dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor, self_attn_mask: Optional[Tensor]=None) -> Tensor:
        """
        Summary:
        Sequentially processes input through multiple transformer decoder layers with optional self-attention masking.

        Description:
        WHY: Enables hierarchical feature transformation and contextual representation learning in sequence-to-sequence models.

        WHEN: Used during inference or training of transformer-based architectures requiring cross-attention between decoder and encoder representations.

        WHERE: Serves as the core forward pass mechanism in transformer decoder networks, facilitating complex sequence generation tasks.

        HOW: Iteratively applies each transformer decoder block to the input, incorporating memory (encoder) context and optional self-attention masking, followed by layer normalization.

        Args:
            x (Tensor): Input tensor representing decoder sequence embeddings.
            memory (Tensor): Encoder-side context tensor providing cross-attention information.
            self_attn_mask (Optional[Tensor], optional): Mask tensor to prevent attending to certain positions. Defaults to None.

        Returns:
            Tensor: Transformed and normalized output tensor after passing through all decoder layers.

        Examples:
            ```python
            # Typical usage in sequence generation
            decoder = TransformerDecoder(
                d_model=512, 
                num_layers=6, 
                num_heads=8, 
                d_ff=2048
            )

            # Forward pass with optional masking
            output = decoder(
                x=decoder_input, 
                memory=encoder_output, 
                self_attn_mask=subsequent_mask
            )
            ```
        """
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask=self_attn_mask)
        return self.norm(x)

class Reconstructor(nn.Module):
    """
    Summary:
    Flexible neural network module for tensor transformation and normalization with configurable projection and scaling strategies.

    Description:
    WHY: Provides a generalized tensor projection and normalization layer that can adapt to different neural network architectures and preprocessing requirements.

    WHEN: Use in neural network architectures requiring dynamic feature transformation, dimensionality reduction, or statistical normalization of tensor representations.

    WHERE: Serves as a versatile component in machine learning pipelines, particularly in deep learning models dealing with tensor manipulations and feature engineering.

    HOW: Implements a linear projection layer with optional log-softmax activation and z-score normalization, allowing fine-grained control over tensor transformations.

    Parameters:
        d_model (int): Input tensor dimensionality to be projected.
        out_channels (int): Desired output dimensionality after projection.
        log_softmax (bool, optional): Flag to apply log-softmax transformation. Defaults to False.
        zscore (str | None, optional): Normalization mode - 'batch', 'time', or None. Defaults to None.
        eps (float, optional): Small epsilon value to prevent division by zero. Defaults to 1e-10.

    Attributes:
        proj (nn.Linear): Linear projection layer transforming input tensor.
        use_log_softmax (bool): Determines whether log-softmax is applied.
        zscore (str | None): Specifies z-score normalization strategy.
        eps (float): Numerical stability constant for normalization.

    Examples:
        ```python
        # Basic usage: simple linear projection
        reconstructor = Reconstructor(d_model=128, out_channels=64)
        output = reconstructor(input_tensor)

        # Advanced usage: with log-softmax and batch normalization
        reconstructor = Reconstructor(
            d_model=128, 
            out_channels=64, 
            log_softmax=True, 
            zscore='batch'
        )
        normalized_output = reconstructor(input_tensor)
        ```
    """

    def __init__(self, d_model: int, out_channels: int, *, log_softmax: bool=False, zscore: str | None=None, eps: float=1e-10) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, out_channels)
        self.use_log_softmax = log_softmax
        self.zscore = zscore
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Summary:
        Transforms input tensor through linear projection with optional log-softmax and z-score normalization.

        Description:
        WHY: Provides a flexible tensor transformation method with configurable normalization strategies for neural network outputs.

        WHEN: Used during neural network forward passes to project and normalize tensor representations.

        WHERE: Serves as a forward method in a neural network module, typically for dimensionality reduction or feature transformation.

        HOW: Applies a linear projection, optionally applies log-softmax, and performs z-score normalization based on specified mode.

        Args:
            x (Tensor): Input tensor to be transformed.

        Returns:
            Tensor: Transformed tensor after projection, optional log-softmax, and z-score normalization.

        Raises:
            ValueError: If an unsupported z-score normalization mode is specified.

        Examples:
            ```python
            # Default behavior: linear projection only
            reconstructor = Reconstructor(d_model=128, out_channels=64)
            output = reconstructor(input_tensor)

            # With log-softmax
            reconstructor = Reconstructor(d_model=128, out_channels=64, log_softmax=True)
            output = reconstructor(input_tensor)

            # With batch-wise z-score normalization
            reconstructor = Reconstructor(d_model=128, out_channels=64, zscore='batch')
            output = reconstructor(input_tensor)
            ```
        """
        y = self.proj(x)
        if self.use_log_softmax:
            y = F.log_softmax(y, dim=-1)
        if self.zscore is None:
            return y
        if self.zscore == 'batch':
            mean = y.mean(dim=0, keepdim=True)
            std = y.std(dim=0, keepdim=True)
        elif self.zscore == 'time':
            mean = y.mean(dim=1, keepdim=True)
            std = y.std(dim=1, keepdim=True)
        else:
            raise ValueError(f'Unsupported zscore mode: {self.zscore}')
        return (y - mean) / (std + self.eps)

class ArtifactRemovalTransformer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_size: int=128, num_encoder_layers: int=6, num_decoder_layers: int=6, num_heads: int=8, feedforward_size: int=2048, dropout: float=0.1, max_len: int=2048, pos_mode: str='sinusoidal', recon_log_softmax: bool=False, recon_zscore: str | None=None) -> None:
        super().__init__()
        self.embedding = nn.Sequential(ExpandConv1x1(in_channels, embedding_size), PositionalEmbedding(max_len=max_len, d_model=embedding_size, mode=pos_mode), nn.Dropout(dropout))
        self.encoder = TransformerEncoder(d_model=embedding_size, num_layers=num_encoder_layers, num_heads=num_heads, d_ff=feedforward_size, dropout=dropout, attn_dropout=dropout)
        self.decoder = TransformerDecoder(d_model=embedding_size, num_layers=num_decoder_layers, num_heads=num_heads, d_ff=feedforward_size, dropout=dropout, attn_dropout=dropout)
        self.reconstructor = Reconstructor(d_model=embedding_size, out_channels=out_channels, log_softmax=recon_log_softmax, zscore=recon_zscore)

    def forward(self, src: Tensor, tgt: Optional[Tensor]=None, src_mask: Optional[Tensor]=None, tgt_mask: Optional[Tensor]=None) -> Tensor:
        enc_attn_mask = None
        if src_mask is not None:
            if src_mask.dtype != torch.bool:
                src_mask = src_mask.to(torch.bool)
            enc_attn_mask = (~src_mask).unsqueeze(1).unsqueeze(2)
        src_x = self.embedding(src)
        memory_src = self.encoder(src_x, attn_mask=enc_attn_mask)
        dec_self_mask = None
        if tgt_mask is not None:
            if tgt_mask.dtype != torch.bool:
                tgt_mask = tgt_mask.to(torch.bool)
            dec_self_mask = (~tgt_mask).unsqueeze(1)
        tgt_x = self.embedding(tgt)
        memory_tgt = self.encoder(tgt_x, attn_mask=dec_self_mask)
        out = self.decoder(memory_tgt, memory_src, enc_attn_mask)
        return self.reconstructor(out)

def build_model_from_config(cfg: dict) -> ArtifactRemovalTransformer:
    m = cfg.get('model', cfg) if isinstance(cfg, dict) else {}
    in_channels = int(m.get('in_channels', 30))
    out_channels = int(m.get('out_channels', 30))
    embedding_size = int(m.get('embedding_size', 128))
    feedforward_size = int(m.get('feedforward_size', 2048))
    num_layers = int(m.get('num_layers', 6))
    num_encoder_layers = int(m.get('num_encoder_layers', num_layers))
    num_decoder_layers = int(m.get('num_decoder_layers', num_layers))
    num_heads = int(m.get('num_heads', 8))
    dropout = float(m.get('dropout', 0.1))
    max_len = int(m.get('max_len', 2048))
    pos_mode = str(m.get('pos_mode', 'sinusoidal'))
    recon_log_softmax = bool(m.get('recon_log_softmax', False))
    recon_zscore = m.get('recon_zscore', None)
    return ArtifactRemovalTransformer(in_channels=in_channels, out_channels=out_channels, embedding_size=embedding_size, feedforward_size=feedforward_size, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, num_heads=num_heads, dropout=dropout, max_len=max_len, pos_mode=pos_mode, recon_log_softmax=recon_log_softmax, recon_zscore=recon_zscore)


__all__ = [
    'ArtifactRemovalTransformer', 
    'build_model_from_config'
]


if __name__ == '__main__':

    def export_default_onnx(out_path: str='ArtifactRemovalTransformer.onnx') -> None:
        (in_ch, out_ch) = (30, 30)
        model = ArtifactRemovalTransformer(in_channels=in_ch, out_channels=out_ch, embedding_size=128, num_encoder_layers=2, num_decoder_layers=2, num_heads=8, feedforward_size=512, dropout=0.1, max_len=2048, pos_mode='sinusoidal', recon_log_softmax=False, recon_zscore=None)
        model.eval()
        T = 256
        src = torch.randn(1, in_ch, T)
        tgt = torch.randn(1, out_ch, T)
        torch.onnx.export(model, (src, tgt), out_path, input_names=['src', 'tgt'], output_names=['y'], opset_version=14, dynamic_axes={'src': {0: 'batch', 2: 'time'}, 'tgt': {0: 'batch', 2: 'time'}, 'y': {0: 'batch', 2: 'time'}})
        print(f'Exported ONNX to {out_path}')
    export_default_onnx()