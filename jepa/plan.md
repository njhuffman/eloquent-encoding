Project Title: Chess-JEPA (Conditional-Contrastive Latent Model)

Objective: Build a Transformer-based world model that learns a strategic latent space by predicting the next board state, conditioned on player Elo.

Phase 1: Data Engineering
    Source Data: pgn file
        Parse PGNs using python-chess.
        State Representation: Convert each board to an 8×8×18 bitmask
        Negative Sampling: For every move made (N→N+1real​), generate several negatives by selecting a random legal moves that were not played.

    Storage: Save to HDF5 for fast random access.
        Schema: {'board_t': (8,8,18), 'board_t_plus_1_pos': (8,8,18), 'board_t_plus_1_negs': (N, 8,8,18), 'elo': float}.

Phase 2: Architecture Implementation (PyTorch)

Goal: Implement the Encoder, Predictor, and Target-EMA machinery.
    The Encoder: * Input: 8×8×18.
        Projection: Linear layer to embed_dim (256).
        Architecture: 4-layer Transformer Encoder (nn.TransformerEncoder).
        Output: A single 1×256 vector (use a [CLS] token or Global Average Pooling).
    The Target Encoder: * A structural twin of the Encoder.
        Important: Parameters are updated via EMA (Exponential Moving Average) from the Encoder, not gradients.
    The Predictor:
        Input: Encoder(Board_N) concatenated or added to Linear(Elo_Scalar).
        Architecture: 2-layer Transformer Encoder.
        Output: Predicted latent Z^N+1​.

Phase 3: The Training Loop & Loss Function
    Goal: Optimize for strategic differentiation, not pixel reconstruction.
    Loss Function (Triplet-Margin):
        Zpos​=TargetEncoder(BoardN+1_pos​)
        Zneg​=TargetEncoder(BoardN+1_neg​). Choose the easiest hard negative - within goal contrastive distance, but by the least amount
        Loss=max(0,∥Z^N+1​−Zpos​∥2−∥Z^N+1​−Zneg​∥2+α)
        Note: Use Cosine Distance.
    Regularization (VICReg light):
        Add a variance penalty to the Encoder's output to prevent dimensional collapse.
    Optimization:
        Optimizer: AdamW (lr=1e-4, weight_decay=0.05).
        Scheduler: CosineAnnealingLR.
        EMA Update: θtarget​=0.999⋅θtarget​+0.001⋅θonline​.