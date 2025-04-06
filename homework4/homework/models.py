from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 2 * n_track * 2  # (left + right), each point has (x, z)
        output_dim = n_waypoints * 2  # (x, z) for each waypoint

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        x = torch.cat([track_left, track_right], dim=-1)  # (B, 10, 4)
        x = x.view(x.size(0), -1)  # Flatten to (B, 40)
        x = self.mlp(x)  # (B, 6)
        return x.view(-1, self.n_waypoints, 2)  # Reshape to (B, 3, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,  # or 64?
        nhead: int = 8,  # Increased attention heads
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Enhanced input processing
        self.track_encoder = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # More sophisticated query embeddings
        self.query_embed = nn.Parameter(torch.randn(n_waypoints, d_model))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)

        # Positional encoding with relative distances
        self.pos_embed = nn.Parameter(torch.randn(1, 2 * n_track, d_model))

        # Deeper transformer with improved configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,  # Wider FFN
            dropout=dropout,
            activation='gelu',  # Smoother than ReLU
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model))

        # Output head with stability improvements
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Tanh()  # Constrain outputs to reasonable range
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # B: batch size
        B = track_left.size(0)

        # 1. Process track boundaries
        track = torch.cat([track_left, track_right], dim=1)  # [B, 2*n_track, 2]
        memory = self.track_encoder(track) + self.pos_embed  # [B, 2*n_track, d_model]

        # 2. Prepare queries (waypoints)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, n_waypoints, d_model]

        # 3. Transformer processing
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=self._generate_square_subsequent_mask(self.n_waypoints).to(queries.device)
        )

        # 4. Output prediction with stability
        waypoints = 2.0 * self.output_head(decoded)  # Scale tanh to [-2,2] range
        return waypoints

    def _generate_square_subsequent_mask(self, sz):
        """Prevent waypoints from attending to future waypoints"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
