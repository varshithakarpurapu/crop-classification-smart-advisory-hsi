import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ==========================================================
# COLOR MAP (MATCHES DATASET LABELS)
# ==========================================================
COLOR_MAP = ListedColormap([
    "#000000",  # Background
    "#ff0000",  # Alfalfa
    "#00ff00",  # Corn-notill
    "#0000ff",  # Corn-mintill
    "#ffff00",  # Corn
    "#ff00ff",  # Grass-pasture
    "#00ffff",  # Grass-trees
    "#800000",  # Hay-windrowed
    "#808000",  # Oats
    "#800080",  # Soybean-notill
    "#008080",  # Soybean-mintill
    "#c0c0c0",  # Soybean-clean
    "#ffa500",  # Wheat
    "#a52a2a",  # Woods
    "#7fffd4",  # Buildings
    "#ff1493",  # Stone/Steel
    "#32cd32"   # Others
])


CLASS_NAMES = [
    "Background","Alfalfa","Corn-notill","Corn-mintill","Corn",
    "Grass-pasture","Grass-trees","Hay-windrowed","Oats",
    "Soybean-notill","Soybean-mintill","Soybean-clean",
    "Wheat","Woods","Buildings","Stone/Steel","Others"
]


# ==========================================================
# FAST FULL-IMAGE PREDICTION (NO EXTRA NORMALIZATION!)
# ==========================================================
def predict_full_image(model, data, labels, patch_size=11, batch_size=256):

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    pad = patch_size // 2

    # IMPORTANT:
    # Data is already PCA-transformed → DO NOT normalize again
    data = data.astype(np.float32)

    data_padded = np.pad(data, ((pad,pad),(pad,pad),(0,0)), mode='reflect')

    h, w, _ = data.shape
    pred_map = np.zeros((h, w), dtype=np.int32)

    patches = []
    positions = []

    def run_batch():
        nonlocal patches, positions

        if not patches:
            return

        x = np.array(patches, dtype=np.float32)

        # shape → (N,1,H,W,Bands)
        x = torch.from_numpy(x).unsqueeze(1).to(device)

        with torch.no_grad():
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()

        for (i, j), p in zip(positions, preds):
            pred_map[i, j] = p + 1  # shift class index

        patches, positions = [], []

    for i in range(h):
        for j in range(w):

            if labels[i, j] == 0:
                continue

            patch = data_padded[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            positions.append((i, j))

            if len(patches) == batch_size:
                run_batch()

    run_batch()
    return pred_map


# ==========================================================
# CONVERT HSI → RGB FOR DISPLAY
# ==========================================================
def convert_to_rgb(data):

    data = data.astype(np.float32)

    R = data[:,:,30]
    G = data[:,:,20]
    B = data[:,:,10]

    rgb = np.stack([R,G,B], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    return rgb


# ==========================================================
# VISUALIZATION (FORCE MULTI-CLASS COLORS)
# ==========================================================
def plot_comparison(pred_map, labels, data):

    rgb = convert_to_rgb(data)

    cmap = COLOR_MAP
    bounds = np.arange(len(CLASS_NAMES) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(15,5))

    # RGB
    plt.subplot(1,3,1)
    plt.imshow(rgb)
    plt.title("(a) RGB Image")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1,3,2)
    plt.imshow(labels, cmap=cmap, norm=norm)
    plt.title("(b) Ground Truth")
    plt.axis("off")

    # Prediction (Discrete colors!)
    plt.subplot(1,3,3)
    plt.imshow(pred_map, cmap=cmap, norm=norm)
    plt.title("(c) Classified Output")
    plt.axis("off")

    # Legend
    patches = [
        plt.plot([], [], marker="s", ms=10, ls="", color=cmap(i))[0]
        for i in range(len(CLASS_NAMES))
    ]

    plt.legend(patches, CLASS_NAMES,
               bbox_to_anchor=(1.25,1),
               loc="upper left")

    plt.tight_layout()
