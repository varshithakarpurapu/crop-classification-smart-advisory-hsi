import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .ml_utils import CROP_NAMES, PAVIA_CLASSES, SALINAS_CLASSES


HEALTH_CLASSES = [
    "Background",
    "Healthy crop",
    "Possible stressed crop",
    "Low vegetation area",
    "Dry/bare soil region",
]

HEALTH_COLORS = ListedColormap([
    "#111111",
    "#2e7d32",
    "#f9a825",
    "#8bc34a",
    "#8d6e63",
])

FERTILIZER_RULES = {
    "Alfalfa": ("Phosphorus and potassium based fertilizer", "Light to moderate irrigation; avoid waterlogging.", "Avoid excess nitrogen because alfalfa fixes nitrogen naturally."),
    "Corn": ("Nitrogen-rich fertilizer with balanced phosphorus", "Moderate to high water requirement during growth stage.", "Apply nitrogen in split doses for better crop uptake."),
    "Corn-notill": ("Nitrogen-rich fertilizer with zinc support", "Moderate irrigation; monitor soil moisture under residue cover.", "Split fertilizer application is useful in no-till fields."),
    "Corn-mintill": ("Nitrogen and phosphorus balanced fertilizer", "Moderate irrigation at early vegetative and tasseling stages.", "Check soil compaction and maintain organic matter."),
    "Soybean-notill": ("Phosphorus and potassium fertilizer", "Moderate irrigation, especially during flowering and pod filling.", "Avoid excess nitrogen; soybean benefits from biological nitrogen fixation."),
    "Soybean-mintill": ("Phosphorus and potassium fertilizer with micronutrients", "Moderate irrigation during flowering and pod development.", "Use rhizobium inoculation where suitable for better nitrogen fixation."),
    "Soybean-clean": ("Phosphorus and potassium fertilizer", "Moderate irrigation; avoid prolonged dry stress.", "Monitor weeds early because clean fields expose soil quickly."),
    "Wheat": ("Nitrogen-rich fertilizer with phosphorus at sowing", "Moderate irrigation at crown root initiation and grain filling.", "Split nitrogen application improves yield and reduces nutrient loss."),
    "Hay-windrowed": ("Potassium-rich fertilizer after cutting", "Light to moderate irrigation after harvest/cutting.", "Potassium helps regrowth and improves forage quality."),
    "Grass-pasture": ("Balanced NPK fertilizer", "Light regular irrigation during dry periods.", "Avoid overgrazing and maintain ground cover."),
    "Grass-trees": ("Organic manure or balanced slow-release fertilizer", "Low to moderate irrigation depending on canopy cover.", "Mixed vegetation areas benefit from soil organic matter improvement."),
}

ROTATION_RULES = {
    "Corn": ("Soybean", "Soybean helps restore nitrogen balance after a cereal crop."),
    "Corn-notill": ("Soybean", "Legume rotation improves soil nitrogen and breaks pest cycles."),
    "Corn-mintill": ("Soybean", "Soybean can improve soil fertility after corn."),
    "Soybean-notill": ("Wheat", "A cereal crop after soybean uses residual nitrogen efficiently."),
    "Soybean-mintill": ("Wheat", "Wheat is a good follow-up crop after soybean."),
    "Soybean-clean": ("Corn", "Corn benefits from the nitrogen contribution of soybean."),
    "Wheat": ("Soybean", "Soybean after wheat diversifies the field and supports soil nitrogen."),
    "Alfalfa": ("Corn", "Corn can use nitrogen left by alfalfa in the soil."),
    "Hay-windrowed": ("Corn", "Corn can benefit from improved soil structure after forage crops."),
    "Grass-pasture": ("Legume crop", "A legume crop can improve nitrogen levels in pasture soil."),
}


def get_class_names(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in ["paviau", "pavia_university"]:
        return PAVIA_CLASSES
    if dataset_name == "salinas":
        return SALINAS_CLASSES
    return CROP_NAMES


def get_crop_name(class_id, dataset_name):
    return get_class_names(dataset_name).get(int(class_id), "Unknown")


def get_recommendation(crop_name):
    fertilizer, irrigation, tip = FERTILIZER_RULES.get(
        crop_name,
        ("Balanced NPK fertilizer based on soil test", "Moderate irrigation based on local soil moisture.", "Use a soil test before final fertilizer application."),
    )
    rotation_crop, rotation_reason = ROTATION_RULES.get(
        crop_name,
        ("Legume or cereal crop based on local season", "A balanced rotation helps maintain soil fertility and reduce disease risk."),
    )
    return {
        "fertilizer": fertilizer,
        "irrigation": irrigation,
        "tip": tip,
        "rotation_crop": rotation_crop,
        "rotation_reason": rotation_reason,
    }


def spectral_vegetation_index(data):        # CROP HEALTH FORMULA 
    data = data.astype(np.float32)
    band_count = data.shape[2]
    red_idx = min(30, band_count - 1)
    nir_idx = min(50, band_count - 1)
    red = data[:, :, red_idx]
    nir = data[:, :, nir_idx]
    return (nir - red) / (nir + red + 1e-8)


def classify_health_from_index(index_values):
    health = np.zeros(index_values.shape, dtype=np.int32)
    valid_values = index_values[np.isfinite(index_values)]

    if valid_values.size == 0:
        return health

    dry_limit, low_limit, healthy_limit = np.percentile(valid_values, [15, 35, 60])
    health[index_values >= healthy_limit] = 1
    health[(index_values >= low_limit) & (index_values < healthy_limit)] = 2
    health[(index_values >= dry_limit) & (index_values < low_limit)] = 3
    health[index_values < dry_limit] = 4
    return health


def analyze_full_image(data, pred_map, dataset_name):
    valid_mask = pred_map > 0
    total_pixels = int(valid_mask.sum())

    if total_pixels == 0:
        return {
            "dominant_crop": "Unknown",
            "coverage_percent": 0,
            "health_summary": [],
            "recommendation": get_recommendation("Unknown"),
            "health_map": np.zeros(pred_map.shape, dtype=np.int32),
        }

    class_ids, counts = np.unique(pred_map[valid_mask], return_counts=True)
    dominant_id = int(class_ids[np.argmax(counts)])
    dominant_crop = get_crop_name(dominant_id, dataset_name)
    coverage_percent = round((int(counts.max()) / total_pixels) * 100, 2)

    index_map = spectral_vegetation_index(data)
    health_map = classify_health_from_index(index_map)
    health_map[~valid_mask] = 0

    crop_summaries = []
    for class_id, count in sorted(zip(class_ids, counts), key=lambda item: item[1], reverse=True):
        class_mask = pred_map == class_id
        health_ids, health_counts = np.unique(health_map[class_mask], return_counts=True)
        health_counts_by_id = dict(zip(health_ids.tolist(), health_counts.tolist()))
        dominant_health_id = max(
            [health_id for health_id in health_counts_by_id if health_id != 0],
            key=lambda health_id: health_counts_by_id[health_id],
            default=0,
        )
        crop_name = get_crop_name(int(class_id), dataset_name)
        crop_summaries.append({
            "crop": crop_name,
            "pixel_count": int(count),
            "coverage_percent": round((int(count) / total_pixels) * 100, 2),
            "health_status": HEALTH_CLASSES[dominant_health_id],
            "recommendation": get_recommendation(crop_name),
        })

    health_summary = []
    for class_id, label in enumerate(HEALTH_CLASSES[1:], start=1):
        count = int(np.sum(health_map == class_id))
        health_summary.append({
            "status": label,
            "pixel_count": count,
            "percent": round((count / total_pixels) * 100, 2),
        })

    return {
        "dominant_crop": dominant_crop,
        "coverage_percent": coverage_percent,
        "crop_summaries": crop_summaries,
        "health_summary": health_summary,
        "recommendation": get_recommendation(dominant_crop),
        "health_map": health_map,
    }


def analyze_patch(patch, crop_name):
    index_map = spectral_vegetation_index(patch)
    mean_index = float(np.nanmean(index_map))

    if mean_index >= 0.35:
        health_status = "Healthy crop"
    elif mean_index >= 0.15:
        health_status = "Possible stressed crop"
    elif mean_index >= 0:
        health_status = "Low vegetation area"
    else:
        health_status = "Dry/bare soil region"

    return {
        "dominant_crop": crop_name,
        "coverage_percent": 100,
        "patch_index": round(mean_index, 4),
        "crop_summaries": [{
            "crop": crop_name,
            "pixel_count": int(patch.shape[0] * patch.shape[1]),
            "coverage_percent": 100,
            "health_status": health_status,
            "recommendation": get_recommendation(crop_name),
        }],
        "health_summary": [{
            "status": health_status,
            "pixel_count": int(patch.shape[0] * patch.shape[1]),
            "percent": 100,
        }],
        "recommendation": get_recommendation(crop_name),
    }


def save_health_map(health_map, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bounds = np.arange(len(HEALTH_CLASSES) + 1) - 0.5
    norm = BoundaryNorm(bounds, HEALTH_COLORS.N)

    plt.figure(figsize=(8, 6))
    plt.imshow(health_map, cmap=HEALTH_COLORS, norm=norm)
    plt.title("Crop Health / Stress Map")
    plt.axis("off")

    legend_items = [
        plt.plot([], [], marker="s", ms=10, ls="", color=HEALTH_COLORS(i))[0]
        for i in range(1, len(HEALTH_CLASSES))
    ]
    plt.legend(legend_items, HEALTH_CLASSES[1:], bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
