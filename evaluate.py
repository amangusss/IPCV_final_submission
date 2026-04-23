"""
Quantitative evaluation: compares AutoCropper against two baselines.

  1. Centre Crop  — fixed crop centred on the image
  2. Saliency Only — crop maximising saliency mean only (no edge, no centre bias)
  3. Ours (combined) — full pipeline

Metrics:
  - Saliency Coverage: mean saliency value inside the chosen crop window
  - Edge Coverage    : mean edge-density inside the chosen crop window
  - Composite Score  : weighted sum used for crop selection

Run:
    python evaluate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.saliency import SaliencyDetector
from src.edge_analysis import EdgeAnalyzer
from src.cropper import AutoCropper, ASPECT_RATIOS
from src.utils import save_image


def centre_crop(image: np.ndarray, ar: str) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) for a centred crop."""
    h, w = image.shape[:2]
    ar_w, ar_h = ASPECT_RATIOS[ar]
    if w / h >= ar_w / ar_h:
        cw, ch = int(round(h * ar_w / ar_h)), h
    else:
        cw, ch = w, int(round(w * ar_h / ar_w))
    x = (w - cw) // 2
    y = (h - ch) // 2
    return x, y, cw, ch


def saliency_only_crop(sal_map: np.ndarray, ar: str,
                       img_w: int, img_h: int,
                       stride_frac: float = 0.05) -> tuple[int, int, int, int]:
    """Sliding window using saliency mean only."""
    ar_w, ar_h = ASPECT_RATIOS[ar]
    if img_w / img_h >= ar_w / ar_h:
        cw, ch = int(round(img_h * ar_w / ar_h)), img_h
    else:
        cw, ch = img_w, int(round(img_w * ar_h / ar_w))

    sx = max(1, int(img_w * stride_frac))
    sy = max(1, int(img_h * stride_frac))
    best_score, bx, by = -1.0, 0, 0
    for y in range(0, img_h - ch + 1, sy):
        for x in range(0, img_w - cw + 1, sx):
            score = float(np.mean(sal_map[y:y+ch, x:x+cw]))
            if score > best_score:
                best_score, bx, by = score, x, y
    return bx, by, cw, ch


def compute_metrics(sal_map: np.ndarray, edge_map: np.ndarray,
                    center_map: np.ndarray,
                    x: int, y: int, w: int, h: int,
                    sal_w=0.55, edge_w=0.25, ctr_w=0.20) -> dict:
    sal = float(np.mean(sal_map[y:y+h, x:x+w]))
    edge = float(np.mean(edge_map[y:y+h, x:x+w]))
    ctr = float(np.mean(center_map[y:y+h, x:x+w]))
    composite = sal_w * sal + edge_w * edge + ctr_w * ctr
    return {"sal_coverage": sal, "edge_coverage": edge,
            "center_coverage": ctr, "composite": composite}


def _center_bias(h, w):
    cx, cy = w / 2.0, h / 2.0
    xv, yv = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    bias = np.exp(-0.5 * (((xv-cx)/(w*0.35))**2 + ((yv-cy)/(h*0.35))**2))
    return bias.astype(np.float32)


def evaluate_image(image: np.ndarray, name: str,
                   ar: str = "4:3") -> dict:
    sal_det = SaliencyDetector()
    edge_det = EdgeAnalyzer()
    cropper = AutoCropper()

    h, w = image.shape[:2]
    sal_map = sal_det.combined(image)
    edge_map = edge_det.combined_edge_map(image)
    center_map = _center_bias(h, w)

    # Baseline 1: Centre crop
    cx, cy, cw, ch = centre_crop(image, ar)
    centre_m = compute_metrics(sal_map, edge_map, center_map, cx, cy, cw, ch)

    # Baseline 2: Saliency-only crop
    sx, sy, sw, sh = saliency_only_crop(sal_map, ar, w, h)
    sal_only_m = compute_metrics(sal_map, edge_map, center_map, sx, sy, sw, sh)

    # Ours
    _, result = cropper.crop(image, ar)
    ours_m = compute_metrics(sal_map, edge_map, center_map,
                             result.x, result.y, result.w, result.h)

    return {
        "name": name,
        "aspect_ratio": ar,
        "centre_crop": centre_m,
        "saliency_only": sal_only_m,
        "ours": ours_m,
    }


def make_demo_images():
    imgs = []

    # Nature: salient sun is strongly off-center (top-right)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    for y in range(600):
        b = int(max(50, 200 - y * 0.25))
        img[y] = [b, max(b - 20, 50), 200]
    # Sun far to the top-right
    cv2.ellipse(img, (680, 80), (70, 70), 0, 0, 360, (0, 230, 255), -1)
    cv2.rectangle(img, (0, 400), (800, 600), (34, 139, 34), -1)
    imgs.append((img, "Nature"))

    # Portrait: face in lower-left quadrant (off-centre)
    img = np.full((600, 800, 3), 160, dtype=np.uint8)
    np.random.seed(42)
    for _ in range(500):
        x, y = np.random.randint(0, 800), np.random.randint(0, 600)
        cv2.circle(img, (x, y), np.random.randint(2, 10),
                   tuple(int(v) for v in (np.random.randint(130, 190),) * 3), -1)
    # Face cluster at (180, 420) — bottom-left
    cv2.ellipse(img, (180, 420), (90, 110), 0, 0, 360, (210, 180, 140), -1)
    cv2.ellipse(img, (150, 405), (14, 10), 0, 0, 360, (50, 50, 150), -1)
    cv2.ellipse(img, (210, 405), (14, 10), 0, 0, 360, (50, 50, 150), -1)
    imgs.append((img, "Portrait"))

    # Mixed: bright object top-left + text block bottom-right
    img = np.full((600, 800, 3), 240, dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, (0, 200, 255), -1)    # salient circle top-left
    for row in range(3):
        y = 370 + row * 70
        cv2.rectangle(img, (500, y), (760, y + 50), (210, 210, 220), -1)
        for ln in range(3):
            cv2.line(img, (510, y + 10 + ln * 13), (750, y + 10 + ln * 13),
                     (60, 60, 60), 1)
    imgs.append((img, "Mixed"))

    return imgs


def plot_comparison(results: list[dict], output_path: str) -> None:
    metrics = ["sal_coverage", "edge_coverage", "composite"]
    labels = ["Saliency Coverage", "Edge Coverage", "Composite Score"]
    methods = ["centre_crop", "saliency_only", "ours"]
    method_labels = ["Centre Crop", "Saliency Only", "Ours (Combined)"]
    colors = ["#5B8DB8", "#E87D5A", "#4CAF50"]

    n_scenes = len(results)
    n_metrics = len(metrics)
    x = np.arange(n_scenes)
    width = 0.25

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    fig.suptitle("Crop Method Comparison Across Scenes", fontsize=14, fontweight="bold")

    for mi, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[mi]
        for i, (method, ml, color) in enumerate(zip(methods, method_labels, colors)):
            vals = [r[method][metric] for r in results]
            ax.bar(x + i * width, vals, width, label=ml, color=color, alpha=0.85)
        ax.set_title(label)
        ax.set_xticks(x + width)
        ax.set_xticklabels([r["name"] for r in results])
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1] * 1.1))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Comparison chart saved: {output_path}")


def print_table(results: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"{'Scene':<12} {'Method':<20} {'SAL Cov':>10} {'EDGE Cov':>10} {'Composite':>12}")
    print(f"{'='*80}")
    method_map = {"centre_crop": "Centre Crop",
                  "saliency_only": "Saliency Only",
                  "ours": "Ours (Combined)"}
    for r in results:
        for method, label in method_map.items():
            m = r[method]
            print(f"{r['name']:<12} {label:<20} "
                  f"{m['sal_coverage']:>10.4f} {m['edge_coverage']:>10.4f} "
                  f"{m['composite']:>12.4f}")
        print(f"{'-'*80}")


def main():
    images = make_demo_images()
    results = []
    # Use 1:1 crop so the window is meaningfully smaller than the image,
    # forcing a real placement decision.
    for img, name in images:
        r = evaluate_image(img, name, ar="1:1")
        results.append(r)

    print_table(results)
    plot_comparison(results, "results/evaluation/comparison_chart.png")

    print("\nImprovement of 'Ours' vs Centre Crop (composite score):")
    for r in results:
        base = r["centre_crop"]["composite"]
        diff = r["ours"]["composite"] - base
        pct = diff / base * 100 if base > 0 else 0.0
        sign = "+" if diff >= 0 else ""
        print(f"  {r['name']:<12}: {sign}{pct:.1f}%")

    print("\nImprovement of 'Ours' vs Saliency Only (composite score):")
    for r in results:
        base = r["saliency_only"]["composite"]
        diff = r["ours"]["composite"] - base
        pct = diff / base * 100 if base > 0 else 0.0
        sign = "+" if diff >= 0 else ""
        print(f"  {r['name']:<12}: {sign}{pct:.1f}%")


if __name__ == "__main__":
    main()
