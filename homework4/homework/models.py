from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models


# Code developed with student effort and
# assistance from generative AI tools (e.g., ChatGPT), per course guidelines.

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
        d_model: int = 160,  # or 64?
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # 1. Input projection
        self.input_proj = nn.Linear(2, d_model)

        # 2. Query embeddings (simpler than before)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # 3. Single transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            batch_first=True  # Simpler batch handling
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        # 4. Output projection
        self.output_proj = nn.Linear(d_model, 2)

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

        # Combine and project tracks
        track = torch.cat([track_left, track_right], dim=1)  # [B, 2*n_track, 2]
        memory = self.input_proj(track)  # [B, 2*n_track, d_model]

        # Get queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, n_waypoints, d_model]

        # Simple transformer processing
        decoded = self.decoder(
            tgt=queries,
            memory=memory
        )

        # Direct output projection
        return self.output_proj(decoded)


class CNNPlanner(torch.nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.n1 = nn.GroupNorm(1, out_channels)
            self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.n2 = nn.GroupNorm(1, out_channels)
            self.relu = nn.ReLU()

            self.skip = (
                nn.Conv2d(in_channels, out_channels, 1, stride, 0)
                if in_channels != out_channels else nn.Identity()
            )

        def forward(self, x0):
            x = self.relu(self.n1(self.c1(x0)))
            x = self.relu(self.n2(self.c2(x)))
            return self.skip(x0) + x

    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        cnn_layers = [
            nn.Conv2d(3, channels_l0, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
        ]

        c1 = channels_l0
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        self.feature_extractor = nn.Sequential(*cnn_layers)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(c1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN feature extraction
        x = self.feature_extractor(x)  # (B, C, H', W')
        x = x.mean(dim=[2, 3])  # Global average pool → (B, C)

        # Fully connected head
        x = self.dropout(x)
        x = self.head(x)  # (B, n_waypoints * 2)
        return x.view(-1, self.n_waypoints, 2)  # (B, n, 2)


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
