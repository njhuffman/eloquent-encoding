"""
CNN MAE: encoder (8x8x19 -> 128-d) and decoder (embedding + mask -> 8x8x12 piece reconstruction).
Loss is applied only on masked positions.
"""

import torch
import torch.nn as nn

from .config import BOARD_HEIGHT, BOARD_WIDTH, EMBEDDING_DIM, ENCODER_INPUT_CHANNELS, PIECE_PLANES


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)


class ChessMAE(nn.Module):
    """
    Masked autoencoder for chess boards.
    - Encoder: (B, 19, 8, 8) -> (B, embedding_dim)
    - Decoder: (B, embedding_dim) + (B, 1, 8, 8) -> (B, 12, 8, 8)

    Default hyperparameters match the original fixed architecture (two ResBlocks at stem width,
    one at mid width, MLP head, three-stage decoder convs).
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        stem_channels: int = 128,
        num_res_blocks_low: int = 2,
        mid_channels: int = 256,
        num_res_blocks_high: int = 1,
        mlp_hidden: int = 1024,
        dropout: float = 0.2,
        decoder_channels: tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        enc_layers: list[nn.Module] = [
            nn.Conv2d(ENCODER_INPUT_CHANNELS, stem_channels, 3, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_res_blocks_low):
            enc_layers.append(ResBlock(stem_channels))
        enc_layers.extend(
            [
                nn.Conv2d(stem_channels, mid_channels, 3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ]
        )
        for _ in range(num_res_blocks_high):
            enc_layers.append(ResBlock(mid_channels))
        enc_layers.extend(
            [
                nn.Flatten(),
                nn.Linear(mid_channels * BOARD_HEIGHT * BOARD_WIDTH, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, embedding_dim),
            ]
        )
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        in_ch = embedding_dim + 1
        for out_ch in decoder_channels:
            dec_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        dec_layers.append(nn.Conv2d(in_ch, PIECE_PLANES, 3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(
        self,
        encoder_input: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        encoder_input: (B, 8, 8, 19) or (B, 19, 8, 8)
        mask: (B, 8, 8, 1) or (B, 1, 8, 8)
        Returns (embedding, decoded_piece_logits).
        """
        # Ensure NCHW
        if encoder_input.dim() == 4 and encoder_input.shape[-1] == 19:
            encoder_input = encoder_input.permute(0, 3, 1, 2)  # (B, 19, 8, 8)
        if mask.dim() == 4 and mask.shape[-1] == 1 and mask.shape[1] != 1:
            mask = mask.permute(0, 3, 1, 2)  # (B, 1, 8, 8)

        emb = self.encoder(encoder_input)  # (B, embedding_dim)
        # Broadcast embedding to spatial and concat mask
        B = emb.shape[0]
        emb_spatial = emb.view(B, -1, 1, 1).expand(B, -1, BOARD_HEIGHT, BOARD_WIDTH)
        dec_input = torch.cat([emb_spatial, mask], dim=1)  # (B, embedding_dim+1, 8, 8)
        out = self.decoder(dec_input)  # (B, 12, 8, 8)
        return emb, out


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    pred, target: (B, 12, 8, 8) or (B, 8, 8, 12)
    mask: (B, 1, 8, 8) or (B, 8, 8, 1), 1.0 = masked (where we apply loss).
    """
    # Normalize to NCHW
    if pred.dim() == 4 and pred.shape[-1] == 12:
        pred = pred.permute(0, 3, 1, 2)  # (B, 12, 8, 8)
    if target.dim() == 4 and target.shape[-1] == 12:
        target = target.permute(0, 3, 1, 2)  # (B, 12, 8, 8)
    if mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2)  # (B, 1, 8, 8)
    # mask (B, 1, 8, 8) broadcast over 12 channels
    se = (pred - target) ** 2
    masked_se = se * mask
    n_masked = mask.sum() * pred.shape[1]  # total masked elements
    if n_masked < 1:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return masked_se.sum() / n_masked.clamp(min=1)
