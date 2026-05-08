import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

from .ml_utils import load_hsi_dataset, create_patches
from .model import FastHSIModel


def train_model_for_dataset(dataset_name,
                            epochs=30,
                            batch_size=64,
                            window_size=11,
                            lr=0.0005):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {dataset_name} on {device}")

    # ---------------- Load FULL dataset ----------------
    data, labels = load_hsi_dataset(dataset_name)
    X_train, X_test, y_train, y_test = create_patches(data, labels, window_size)

    # ---------------- Normalize (VERY IMPORTANT) ----------------
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    X_train /= (X_train.max() + 1e-8)
    X_test  /= (X_test.max() + 1e-8)

    # ---------------- Convert to tensors ----------------
    X_train = torch.from_numpy(X_train).unsqueeze(1).to(device)
    y_train = torch.from_numpy(y_train).long().to(device)

    X_test = torch.from_numpy(X_test).unsqueeze(1).to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=batch_size)

    # ---------------- Model ----------------
    num_classes = int(labels.max())
    model = FastHSIModel(input_channels=X_train.shape[-1],
                         num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🔥 -------- CLASS BALANCING (THE FIX) -------- 🔥
    y_np = y_train.cpu().numpy()
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_np),
        y=y_np
    )

    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print("Class weights:", class_weights)

    # ---------------- Training ----------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---------------- Validation ----------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}  Loss:{total_loss:.3f}  Acc:{acc:.4f}")

    # ---------------- Save ----------------
    os.makedirs("ml_models", exist_ok=True)
    save_path = f"ml_models/{dataset_name}_fast.pth"
    torch.save(model.state_dict(), save_path)

    print("✅ Model saved:", save_path)
    return model



def train_multiple_datasets(datasets, epochs=25):
    """
    Train models sequentially for multiple datasets.
    """
    trained_models = {}

    for dataset in datasets:
        trained_models[dataset] = train_model_for_dataset(dataset, epochs=epochs)

    return trained_models


# --------------------------------------------------
# RUN TRAINING FROM COMMAND LINE
# --------------------------------------------------
if __name__ == "__main__":
    datasets = ["indian_pines", "pavia_university", "salinas"]
    train_multiple_datasets(datasets, epochs=25)
