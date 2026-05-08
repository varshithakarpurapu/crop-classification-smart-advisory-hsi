import torch
import numpy as np
import joblib
from .model import FastHSIModel


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
def load_trained_model(model_path, input_channels, num_classes, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastHSIModel(input_channels, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"✅ Model loaded from {model_path}")
    return model


# ==========================================================
# LOAD SAVED PCA MODEL
# ==========================================================
def load_pca_model(dataset_name):
    pca_path = f"ml_models/{dataset_name}_pca.pkl"
    pca = joblib.load(pca_path)
    print(f"✅ PCA loaded from {pca_path}")
    return pca


# ==========================================================
# APPLY PCA TO FULL IMAGE (Used before creating patches)
# ==========================================================

def apply_pca_to_image(data, dataset_name):
    """
    Apply the SAME PCA used during training.
    """

    # Load trained PCA model
    pca_path = f"ml_models/{dataset_name}_pca.pkl"
    pca_model = joblib.load(pca_path)

    # Reshape image to 2D
    h, w, b = data.shape
    reshaped = data.reshape(-1, b)

    # Apply PCA transform
    reduced = pca_model.transform(reshaped)

    # Reshape back to image
    reduced_data = reduced.reshape(h, w, -1)

    return reduced_data



# ==========================================================
# NORMALIZE PATCH (Same as training)
# ==========================================================
def normalize_patch(patch):

    patch = patch.astype(np.float32)

    # Band-wise normalization
    for b in range(patch.shape[2]):
        band = patch[:, :, b]
        patch[:, :, b] = (band - band.min()) / (band.max() - band.min() + 1e-8)

    return patch


# ==========================================================
# PREDICT SINGLE PATCH
# ==========================================================
def predict_patch(model, patch):

    device = next(model.parameters()).device

    patch = normalize_patch(patch)

    # Convert to tensor: (H,W,C) → (1,1,H,W,C)
    x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1)

    return pred.item()


# ==========================================================
# HELPER FUNCTION FOR DJANGO PIPELINE
# ==========================================================
def prepare_image_for_prediction(dataset_name, data):
    """
    Load PCA and transform image before prediction.
    """
    pca_model = load_pca_model(dataset_name)
    data = apply_pca_to_image(data, pca_model)
    return data
