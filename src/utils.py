"""
Visualisation and I/O utilities for the auto-cropping pipeline.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional

from .cropper import CropResult


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(image: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)


def visualize_pipeline(original: np.ndarray,
                       sal_map: np.ndarray,
                       edge_map: np.ndarray,
                       crop_result: CropResult,
                       output_path: Optional[str] = None) -> plt.Figure:
    """
    Four-panel diagnostic figure:
      1. Original image with crop overlay
      2. Saliency map (heatmap)
      3. Edge density map
      4. Cropped result
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Automatic Image Cropping Pipeline", fontsize=14, fontweight="bold")

    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    rect = patches.Rectangle(
        (crop_result.x, crop_result.y),
        crop_result.w, crop_result.h,
        linewidth=3, edgecolor="red", facecolor="none",
        label=f"Crop ({crop_result.aspect_ratio})"
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"Original + Crop Region\n"
                      f"Score: {crop_result.score:.3f}")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axis("off")

    im1 = axes[1].imshow(sal_map, cmap="hot")
    axes[1].set_title(f"Saliency Map\n"
                      f"Coverage: {crop_result.saliency_coverage:.3f}")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(edge_map, cmap="Blues")
    axes[2].set_title(f"Edge Density Map\n"
                      f"Edge score: {crop_result.edge_coverage:.3f}")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    y, x, h, w = crop_result.y, crop_result.x, crop_result.h, crop_result.w
    cropped_rgb = cv2.cvtColor(original[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    axes[3].imshow(cropped_rgb)
    axes[3].set_title(f"Auto-Cropped Result\n"
                      f"Aspect ratio: {crop_result.aspect_ratio}")
    axes[3].axis("off")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def draw_crop_overlay(image: np.ndarray, result: CropResult,
                      color: tuple = (0, 0, 255), thickness: int = 3) -> np.ndarray:
    """Draw the crop rectangle on a copy of the image."""
    overlay = image.copy()
    cv2.rectangle(overlay,
                  (result.x, result.y),
                  (result.x + result.w, result.y + result.h),
                  color, thickness)
    label = f"Crop {result.aspect_ratio} | Score:{result.score:.3f}"
    cv2.putText(overlay, label, (result.x + 5, result.y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return overlay


def compare_crops(original: np.ndarray,
                  crops: dict[str, tuple[np.ndarray, CropResult]],
                  output_path: Optional[str] = None) -> plt.Figure:
    """Grid showing crops for all aspect ratios side by side."""
    n = len(crops)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle("Crops by Aspect Ratio", fontsize=14, fontweight="bold")

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (ar, (cropped, result)) in enumerate(crops.items()):
        ax = axes_flat[idx]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{ar}\nScore: {result.score:.3f}")
        ax.axis("off")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def print_crop_report(result: CropResult) -> None:
    print(f"\n{'='*50}")
    print(f"  Crop Report")
    print(f"{'='*50}")
    print(f"  Aspect Ratio : {result.aspect_ratio}")
    print(f"  Position     : ({result.x}, {result.y})")
    print(f"  Size         : {result.w} x {result.h}")
    print(f"  Composite Score       : {result.score:.4f}")
    print(f"  Saliency Coverage     : {result.saliency_coverage:.4f}")
    print(f"  Edge Coverage         : {result.edge_coverage:.4f}")
    print(f"{'='*50}\n")
