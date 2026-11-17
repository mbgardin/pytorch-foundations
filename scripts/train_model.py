"""Train a tabular classifier from CSV to saved model.

Run from the repository root: uv run python scripts/train_model.py
After choosing settings using validation: uv run python scripts/train_model.py --test
Edit the settings below, then rerun. Each run replaces the checkpoint and history.
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# EXPERIMENT SETTINGS: change these values to explore different models.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "observations.csv"
OUTPUT_DIR = ROOT / "artifacts"
FEATURES = ["feature_1", "feature_2", "feature_3"]
LABEL = "label"                 # Integer class indices 0, ..., N_CLASSES - 1.
SPLIT_COLUMN = "split"          # Values: train, validation, test.
N_CLASSES = 2
HIDDEN_UNITS = 16
DROPOUT = 0.0
BATCH_SIZE = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 50
PATIENCE = 8                    # Stop after this many epochs without improvement.
SEED = 486


class Classifier(nn.Module):
    """Change the layers here to try another architecture."""

    def __init__(self, n_features, hidden_units, n_classes, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, n_classes),  # Raw logits for CrossEntropyLoss.
        )

    def forward(self, x):
        return self.layers(x)


def make_loader(rows, scaler, *, shuffle):
    """Apply the training-fitted scaler and create batches of features and labels."""
    x = torch.tensor(scaler.transform(rows[FEATURES]), dtype=torch.float32)
    y = torch.tensor(rows[LABEL].to_numpy(), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)


def evaluate(model, loader, loss_fn):
    """Compute example-weighted loss, accuracy, and a confusion matrix."""
    model.eval()
    total_loss, count = 0.0, 0
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * len(y)
            count += len(y)
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(y.tolist())
    matrix = confusion_matrix(targets, predictions, labels=list(range(N_CLASSES)))
    return total_loss / count, np.mean(np.array(predictions) == np.array(targets)), matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Evaluate the selected model on the held-out test set.")
    args = parser.parse_args()

    # 1. Set seeds and inspect the supplied split before fitting anything.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    frame = pd.read_csv(DATA_PATH)
    required = FEATURES + [LABEL, SPLIT_COLUMN]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not frame[SPLIT_COLUMN].isin(["train", "validation", "test"]).all():
        raise ValueError("Split values must be train, validation, or test.")
    if frame[required].isna().any().any():
        raise ValueError("Missing values found; choose and fit a training-only imputation strategy first.")
    if not frame[LABEL].isin(range(N_CLASSES)).all():
        raise ValueError(f"Labels must be integer class indices from 0 to {N_CLASSES - 1}.")
    train = frame[frame[SPLIT_COLUMN] == "train"]
    validation = frame[frame[SPLIT_COLUMN] == "validation"]
    test = frame[frame[SPLIT_COLUMN] == "test"]
    if any(rows.empty for rows in (train, validation, test)):
        raise ValueError("Each of the train, validation, and test splits must contain rows.")
    print(f"Rows: train={len(train)}, validation={len(validation)}, test={len(test)}")

    # 2. Fit preprocessing on training rows only, then reuse it on other splits.
    scaler = StandardScaler().fit(train[FEATURES])
    train_loader = make_loader(train, scaler, shuffle=True)
    train_eval_loader = make_loader(train, scaler, shuffle=False)
    val_loader = make_loader(validation, scaler, shuffle=False)
    x_batch, y_batch = next(iter(train_loader))
    print(f"Example batch: inputs {tuple(x_batch.shape)} {x_batch.dtype}; labels {tuple(y_batch.shape)} {y_batch.dtype}")

    # 3. Define the model, objective, and optimizer. Try changing any of them.
    model = Classifier(len(FEATURES), HIDDEN_UNITS, N_CLASSES, DROPOUT)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_loss = float("inf")
    best_state = None
    best_epoch, stale_epochs = 0, 0
    history = []

    # 4. Train in batches and select an epoch using validation loss.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy, _ = evaluate(model, train_eval_loader, loss_fn)
        val_loss, val_accuracy, _ = evaluate(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "validation_loss": val_loss, "validation_accuracy": val_accuracy})
        print(f"Epoch {epoch:3d}: train loss={train_loss:.4f}, accuracy={train_accuracy:.3f}; "
              f"validation loss={val_loss:.4f}, accuracy={val_accuracy:.3f}")
        if val_loss < best_loss:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without improvement.")
                break

    model.load_state_dict(best_state)
    val_loss, val_accuracy, matrix = evaluate(model, val_loader, loss_fn)
    print(f"Selected epoch {best_epoch}: validation loss={val_loss:.4f}, accuracy={val_accuracy:.3f}")
    print("Validation confusion matrix (rows: actual; columns: predicted):\n", matrix)

    # 5. Save enough information to rebuild this architecture and preprocessing.
    # This is an inference checkpoint, not a complete training-resume checkpoint.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "training_example.pt"
    torch.save({
        "model_state": model.state_dict(),
        "features": FEATURES,
        "label": LABEL,
        "class_names": [str(i) for i in range(N_CLASSES)],
        "hidden_units": HIDDEN_UNITS,
        "n_classes": N_CLASSES,
        "dropout": DROPOUT,
        "mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "best_epoch": best_epoch,
    }, checkpoint_path)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"Saved checkpoint to {checkpoint_path} and learning history to {OUTPUT_DIR / 'training_history.csv'}")

    # 6. Reserve test results until the architecture and settings are selected.
    if args.test:
        test_loader = make_loader(test, scaler, shuffle=False)
        test_loss, test_accuracy, matrix = evaluate(model, test_loader, loss_fn)
        print(f"Held-out test: loss={test_loss:.4f}, accuracy={test_accuracy:.3f}")
        print("Test confusion matrix (rows: actual; columns: predicted):\n", matrix)
    else:
        print("Test evaluation skipped. Once model choices are final, rerun with --test.")

    # 7. Rebuild the saved model and predict a new raw row in the saved feature order.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = Classifier(len(saved["features"]), saved["hidden_units"], saved["n_classes"], saved["dropout"])
    restored.load_state_dict(saved["model_state"])
    restored.eval()
    raw_row = torch.tensor(validation[FEATURES].iloc[[0]].to_numpy(), dtype=torch.float32)  # Replace with new data.
    with torch.no_grad():
        probabilities = torch.softmax(restored((raw_row - saved["mean"]) / saved["scale"]), dim=1)
    predicted = probabilities.argmax(dim=1).item()
    print(f"Example prediction: class={saved['class_names'][predicted]}, probabilities={probabilities[0].tolist()}")


if __name__ == "__main__":
    main()
