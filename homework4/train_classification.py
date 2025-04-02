import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import torch.utils.tensorboard as tb
from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data

def train(exp_dir="logs", num_epochs=30, batch_size=64, learning_rate=0.001, seed=2024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Logging
    log_dir = Path(exp_dir) / f"classifier_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load Data
    train_data = load_data("classification_data/train", transform_pipeline="aug", batch_size=batch_size, shuffle=True)
    val_data = load_data("classification_data/val", transform_pipeline="default", batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = Classifier().to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = loss_func(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == label).sum().item()
                total += label.size(0)

        val_accuracy = correct / total
        logger.add_scalar("val_accuracy", val_accuracy, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}")

    save_model(model)
    torch.save(model.state_dict(), log_dir / "classifier.th")
    print(f"Model saved at {log_dir}/classifier.th")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()
    train(**vars(args))
