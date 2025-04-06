import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
# from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
# print("Time to train")


def train(model_name="mlp_planner", num_epochs=50, batch_size=32, lr=0.0007, exp_dir="logs"):
    # Universal device selection (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")  # For NVIDIA GPUs
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    else:
        device = torch.device("cpu")  # Default fallback

    print(f"Using device: {device}")

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(str(log_dir))

    transform_pipeline = "state_only" if "planner" in model_name else "default"
    train_loader = load_data("drive_data/train", transform_pipeline=transform_pipeline, batch_size=batch_size, shuffle=True)
    val_loader = load_data("drive_data/val", transform_pipeline=transform_pipeline, batch_size=batch_size, shuffle=False)

    model = load_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    loss_fn = torch.nn.SmoothL1Loss()

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_metric = PlannerMetric()

        for batch in train_loader:
            track_left = batch.get("track_left", None)
            track_right = batch.get("track_right", None)
            image = batch.get("image", None)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            inputs = {}
            if track_left is not None:
                inputs["track_left"] = track_left.to(device)
            if track_right is not None:
                inputs["track_right"] = track_right.to(device)
            if image is not None:
                inputs["image"] = image.to(device)

            pred = model(**inputs)
            loss = loss_fn(pred * waypoints_mask[..., None], waypoints * waypoints_mask[..., None])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric.add(pred, waypoints, waypoints_mask)

            logger.add_scalar("train/loss", loss.item(), global_step)
            train_loss += loss.item()
            global_step += 1

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_metric = PlannerMetric()
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = {}
                if "track_left" in batch:
                    inputs["track_left"] = batch["track_left"].to(device)
                if "track_right" in batch:
                    inputs["track_right"] = batch["track_right"].to(device)
                if "image" in batch:
                    inputs["image"] = batch["image"].to(device)

                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)
                pred = model(**inputs)

                val_loss = loss_fn(pred * waypoints_mask[..., None], waypoints * waypoints_mask[..., None])
                # log to visualize in TensorBoard
                logger.add_scalar("val/loss", val_loss.item(), epoch)

                val_metric.add(pred, waypoints, waypoints_mask)

        val_results = val_metric.compute()
        logger.add_scalar("val/longitudinal_error", val_results["longitudinal_error"], epoch)
        logger.add_scalar("val/lateral_error", val_results["lateral_error"], epoch)
        logger.add_scalar("train/avg_loss", avg_train_loss, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Lateral: {val_results['lateral_error']:.4f} | "
              f"Val Longitudinal: {val_results['longitudinal_error']:.4f}")

    scheduler.step()
    save_path = save_model(model)
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()
    train(**vars(args))

