"""
CNN MAE: encoder (8x8x19 -> 128-d) and decoder (embedding + mask -> 8x8x12 piece reconstruction).
Loss is applied only on masked positions.
"""

import torch
import torch.nn as nn

from .config import BOARD_HEIGHT, BOARD_WIDTH, EMBEDDING_DIM, ENCODER_INPUT_CHANNELS, PIECE_PLANES


class ChessMAE(nn.Module):
    """
    Masked autoencoder for chess boards.
    - Encoder: (B, 19, 8, 8) -> (B, 128)
    - Decoder: (B, 128) + (B, 1, 8, 8) -> (B, 12, 8, 8)
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Encoder: 19 -> 64 -> 128 -> 256 -> 512 -> embedding_dim
        self.encoder = nn.Sequential(
            # Layer 1: Keep it wide from the start
            nn.Conv2d(ENCODER_INPUT_CHANNELS, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 2: No stride! Keep 8x8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: Increase depth, still no stride
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: Final spatial processing
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Flatten(),
            # Now input is 256 * 8 * 8 = 16,384 (Massive signal compared to your current 1,024)
            nn.Linear(256 * 8 * 8, 1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), # Prevent overfitting on this big linear layer
            nn.Linear(1024, embedding_dim),
        )
        # Decoder: embedding (B, 128) broadcast to (B, 128, 8, 8), concat mask -> (B, 129, 8, 8) -> (B, 12, 8, 8)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim + 1, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, PIECE_PLANES, 3, padding=1),
        )

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
        emb_spatial = emb.view(B, -1, 1, 1).expand(B, -1, BOARD_HEIGHT, BOARD_WIDTH)  # (B, 128, 8, 8)
        dec_input = torch.cat([emb_spatial, mask], dim=1)  # (B, 129, 8, 8)
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
