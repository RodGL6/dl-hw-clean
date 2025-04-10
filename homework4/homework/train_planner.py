import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from math import inf

# Code developed with student effort and
# assistance from generative AI tools (e.g., ChatGPT), per course guidelines.

# from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import CNNPlanner  # so isinstance() works


"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
# print("Time to train")


def custom_loss(pred, target, mask, lateral_weight=1.5):
    """Custom loss with extra penalty for lateral errors"""
    base_loss = torch.nn.functional.smooth_l1_loss(
        pred * mask[..., None],
        target * mask[..., None]
    )
    lateral_diff = torch.abs(pred[..., 1] - target[..., 1]) * mask
    lateral_loss = lateral_diff.mean()
    return base_loss + lateral_weight * lateral_loss


def train(model_name="mlp_planner", num_epochs=50, batch_size=32, lr=0.0007, exp_dir="logs",  combined_weights=(0.7, 0.3),  # (longitudinal_weight, lateral_weight)
):
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

    if model_name == "cnn_planner":
        transform_pipeline = "default"  # this loads image + waypoints
    elif "planner" in model_name:
        transform_pipeline = "state_only"
    else:
        transform_pipeline = "default"

    train_loader = load_data("drive_data/train", transform_pipeline=transform_pipeline, batch_size=batch_size, shuffle=True)
    val_loader = load_data("drive_data/val", transform_pipeline=transform_pipeline, batch_size=batch_size, shuffle=False)

    model = load_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # e4?
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    loss_fn = torch.nn.SmoothL1Loss()


    best_metrics = {
        'longitudinal': float('inf'),
        'lateral': float('inf'),
        'combined': float('inf')
    }
    best_model_path = ""

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

            # Dynamically decide what inputs to pass based on model type
            if isinstance(model, CNNPlanner):
                if "image" not in inputs:
                    raise ValueError("CNNPlanner requires 'image' in inputs but it was missing.")
                pred = model(image=inputs["image"])
            else:
                pred = model(**inputs)

            # loss = loss_fn(pred * waypoints_mask[..., None], waypoints * waypoints_mask[..., None])
            loss = custom_loss(pred, waypoints, waypoints_mask, lateral_weight=1.5)


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
        val_loss_total = 0.0
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

                # Dynamically decide what inputs to pass based on model type
                if isinstance(model, CNNPlanner):
                    if "image" not in inputs:
                        raise ValueError("CNNPlanner requires 'image' in inputs but it was missing.")
                    pred = model(image=inputs["image"])
                else:
                    pred = model(**inputs)

                val_loss = loss_fn(pred * waypoints_mask[..., None], waypoints * waypoints_mask[..., None])

                val_loss_total += val_loss.item()
                val_metric.add(pred, waypoints, waypoints_mask)

                # log to visualize in TensorBoard
                logger.add_scalar("val/loss", val_loss.item(), epoch)

                val_metric.add(pred, waypoints, waypoints_mask)

        val_results = val_metric.compute()
        longitudinal_error = val_results["longitudinal_error"]
        lateral_error = val_results["lateral_error"]
        combined = combined_weights[0] * longitudinal_error + combined_weights[1] * lateral_error

        logger.add_scalar("val/longitudinal_error", longitudinal_error, epoch)
        logger.add_scalar("val/lateral_error", lateral_error, epoch)
        logger.add_scalar("train/avg_loss", avg_train_loss, epoch)
        logger.add_scalar("val/combined_score", combined, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Longitudinal: {longitudinal_error:.4f} | "
              f"Val Lateral: {lateral_error:.4f}")

        # Save if all metrics improve or combined is best
        if combined < best_metrics['combined']:
            best_metrics.update({
                'longitudinal': longitudinal_error,
                'lateral': lateral_error,
                'combined': combined
            })

            if best_model_path and Path(best_model_path).exists():
                Path(best_model_path).unlink()

            best_model_path = save_model(model)
            print(f"🔥 New best model saved at epoch {epoch + 1} | Combined: {combined:.4f}")

        # Adjust LR based on performance plateau
        scheduler.step() #combined

    print(f"\n🏁 Training finished. Best model: {best_model_path}")
    print(f"📊 Final Best - Longitudinal: {best_metrics['longitudinal']:.4f}, "
          f"Lateral: {best_metrics['lateral']:.4f}, Combined: {best_metrics['combined']:.4f}")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()
    train(**vars(args))

