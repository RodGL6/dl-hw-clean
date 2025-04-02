import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric


def train(
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Trains the road detection model
    """

    print(f"Using device: {device}")

    # Load dataset
    train_loader = load_data("drive_data/train", batch_size=batch_size, shuffle=True)
    val_loader = load_data("drive_data/val", batch_size=batch_size, shuffle=False)

    # Initialize model
    model = Detector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define loss functions
    segmentation_loss = nn.CrossEntropyLoss()  # For segmentation
    depth_loss = torch.nn.MSELoss()  # Mean Squared Error (penalizes larger errors more)

    # Metrics
    train_metric = DetectionMetric()
    val_metric = DetectionMetric()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_metric.reset()

        for batch in train_loader:
            images = batch["image"].to(device)  # Shape: (B, 3, 96, 128)
            labels = batch["track"].to(device)  # Segmentation labels: (B, 96, 128)
            depths = batch["depth"].to(device).unsqueeze(1)  # Depth values: (B, 1, 96, 128)

            # Forward pass
            logits, depth_preds = model(images)

            # Compute losses
            # loss_seg = segmentation_loss(logits, labels)  # Cross-entropy loss

            road_mask = (labels > 0).float()  # Mask for road pixels
            boundary_mask = ((labels == 1) | (labels == 2)).float()  # Mask for lane boundaries

            # Compute absolute depth difference
            depth_diff = torch.abs(depth_preds - depths.squeeze(1))  # Fix shape mismatch

            # Apply log scaling to focus learning on small errors
            log_depth_loss = (torch.log1p(depth_diff) * road_mask).mean()

            # Compute weighted depth loss
            road_loss = (depth_loss(depth_preds, depths.squeeze(1)) * road_mask).mean()
            lane_loss = (depth_loss(depth_preds, depths.squeeze(1)) * boundary_mask).mean()

            # Combine both depth losses
            loss_depth = road_loss + 2.0 * lane_loss + 0.3 * log_depth_loss  # Add log-scaled loss

            # Compute segmentation and IoU loss
            loss_seg = segmentation_loss(logits, labels)  # Cross-entropy loss
            loss_iou = 1 - train_metric.compute()["iou"]  # IoU loss
            loss_iou = loss_iou / (loss_iou + 1e-6)  # Normalize IoU loss

            # Final loss function
            loss = loss_seg + 0.5 * loss_depth + 1.3 * loss_iou


            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_metric.add(logits.argmax(1), labels, depth_preds, depths)

        # Compute training IoU, Accuracy, Depth Error
        train_results = train_metric.compute()

        # Validation loop
        model.eval()
        val_loss = 0
        val_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["track"].to(device)
                depths = batch["depth"].to(device).unsqueeze(1)

                logits, depth_preds = model(images)

                # Compute losses
                # loss_seg = segmentation_loss(logits, labels)

                road_mask = (labels > 0).float()  # Mask for road pixels
                boundary_mask = ((labels == 1) | (labels == 2)).float()  # Mask for lane boundaries

                # Compute absolute depth difference
                depth_diff = torch.abs(depth_preds - depths.squeeze(1))  # Fix shape mismatch

                # Apply log scaling to focus learning on small errors
                log_depth_loss = (torch.log1p(depth_diff) * road_mask).mean()

                # Compute weighted depth loss
                road_loss = (depth_loss(depth_preds, depths.squeeze(1)) * road_mask).mean()
                lane_loss = (depth_loss(depth_preds, depths.squeeze(1)) * boundary_mask).mean()

                # Combine both depth losses
                loss_depth = road_loss + 2.0 * lane_loss + 0.3 * log_depth_loss  # Add log-scaled loss

                # Compute segmentation and IoU loss
                loss_seg = segmentation_loss(logits, labels)  # Cross-entropy loss
                loss_iou = 1 - train_metric.compute()["iou"]  # IoU loss
                loss_iou = loss_iou / (loss_iou + 1e-6)  # Normalize IoU loss

                # Final loss function
                loss = loss_seg + 0.5 * loss_depth + 1.3 * loss_iou

                val_loss += loss.item()
                val_metric.add(logits.argmax(1), labels, depth_preds, depths)

        val_results = val_metric.compute()

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f} - "
              f"Train IoU: {train_results['iou']:.4f} - Train Accuracy: {train_results['accuracy']:.4f} - "
              f"Train Depth MAE: {train_results['abs_depth_error']:.4f} - Lane Depth MAE: {train_results['tp_depth_error']:.4f} - "
              f"Val IoU: {val_results['iou']:.4f} - Val Accuracy: {val_results['accuracy']:.4f} - "
              f"Val Depth MAE: {val_results['abs_depth_error']:.4f} - Val Lane Depth MAE: {val_results['tp_depth_error']:.4f}")

    # Save model
    save_model(model)
    print("Training complete! Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    train(num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
