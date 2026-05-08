import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

def load_hsi_dataset(name):
    name = name.lower()
    
    if name == "indian_pines":
        data = sio.loadmat("data/Indian_pines_corrected.mat")["indian_pines_corrected"]
        labels = sio.loadmat("data/Indian_pines_gt.mat")["indian_pines_gt"]
    elif name == "paviau" or name == "pavia_university":
        data = sio.loadmat("data/PaviaU.mat")["paviaU"]
        labels = sio.loadmat("data/PaviaU_gt.mat")["paviaU_gt"]
    elif name == "salinas":
        data = sio.loadmat("data/Salinas_corrected.mat")["salinas_corrected"]
        labels = sio.loadmat("data/Salinas_gt.mat")["salinas_gt"]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"✅ Loaded {name} dataset — shape: {data.shape}, classes: {len(np.unique(labels)) - 1}")
    return data, labels

def create_patches(data, labels, window_size=11, test_ratio=0.2):
    pad = window_size // 2
    data_padded = np.pad(data, ((pad,pad),(pad,pad),(0,0)), mode='reflect')

    X, y = [], []
    for i in range(pad, data.shape[0] + pad):
        for j in range(pad, data.shape[1] + pad):
            label = labels[i-pad, j-pad]
            if label != 0:
                patch = data_padded[i-pad:i+pad+1, j-pad:j+pad+1, :]
                X.append(patch)
                y.append(int(label)-1)
    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)

def sample_data(X, y, max_samples=5000):
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        return X[idx], y[idx]
    return X, y

CROP_NAMES = {
    1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill", 4: "Corn",
    5: "Grass-pasture", 6: "Grass-trees", 7: "Grass-pasture-mowed",
    8: "Hay-windrowed", 9: "Oats", 10: "Soybean-notill",
    11: "Soybean-mintill", 12: "Soybean-clean", 13: "Wheat",
    14: "Woods", 15: "Buildings-Grass-Trees-Drives", 16: "Stone-Steel-Towers"
}

PAVIA_CLASSES = {1:"Asphalt",2:"Meadows",3:"Gravel",4:"Trees",5:"Painted metal sheets",
6:"Bare Soil",7:"Bitumen",8:"Self-Blocking Bricks",9:"Shadows"}

SALINAS_CLASSES = {1:"Broccoli green weeds 1",2:"Broccoli green weeds 2",3:"Fallow",
4:"Fallow rough plow",5:"Fallow smooth",6:"Stubble",7:"Celery",8:"Grapes untrained",
9:"Soil vineyards develop",10:"Corn senesced green weeds",11:"Lettuce romaine 4wk",
12:"Lettuce romaine 5wk",13:"Lettuce romaine 6wk",14:"Lettuce romaine 7wk",
15:"Vineyard untrained",16:"Vineyard vertical trellis"}



from sklearn.decomposition import PCA

def apply_pca(data, n_components=30):
    """
    Apply PCA to reduce spectral bands.
    Input: (H, W, Bands)
    Output: (H, W, n_components)
    """
    h, w, b = data.shape
    reshaped = data.reshape(-1, b)

    pca = PCA(n_components=n_components, whiten=True)
    reduced = pca.fit_transform(reshaped)

    reduced_data = reduced.reshape(h, w, n_components)

    print(f"✅ PCA applied: {b} → {n_components} bands")
    return reduced_data, pca
