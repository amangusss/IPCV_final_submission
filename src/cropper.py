"""
Automatic image cropper.

Combines saliency and edge maps to score candidate crop windows
and select the optimal crop for each requested aspect ratio.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .saliency import SaliencyDetector
from .edge_analysis import EdgeAnalyzer


@dataclass
class CropResult:
    x: int
    y: int
    w: int
    h: int
    score: float
    aspect_ratio: str
    saliency_coverage: float
    edge_coverage: float


ASPECT_RATIOS = {
    "1:1":  (1.0, 1.0),
    "4:3":  (4.0, 3.0),
    "3:4":  (3.0, 4.0),
    "16:9": (16.0, 9.0),
    "9:16": (9.0, 16.0),
    "3:2":  (3.0, 2.0),
    "2:3":  (2.0, 3.0),
}


class AutoCropper:
    """
    Scores candidate crop windows using a weighted sum of:
      - Saliency coverage  (content importance)
      - Edge density       (structural richness)
      - Centre bias        (prefer crops near image centre)
    """

    def __init__(self,
                 saliency_weight: float = 0.55,
                 edge_weight: float = 0.25,
                 center_weight: float = 0.20,
                 stride_fraction: float = 0.05,
                 min_crop_fraction: float = 0.40):
        self.saliency_weight = saliency_weight
        self.edge_weight = edge_weight
        self.center_weight = center_weight
        self.stride_fraction = stride_fraction
        self.min_crop_fraction = min_crop_fraction

        self._saliency = SaliencyDetector()
        self._edges = EdgeAnalyzer()

    def _center_bias_map(self, h: int, w: int) -> np.ndarray:
        """Gaussian centre-bias prior: central pixels are preferred."""
        cx, cy = w / 2.0, h / 2.0
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xv, yv = np.meshgrid(x_coords, y_coords)
        sigma_x = w * 0.35
        sigma_y = h * 0.35
        bias = np.exp(-0.5 * (((xv - cx) / sigma_x) ** 2 + ((yv - cy) / sigma_y) ** 2))
        return bias.astype(np.float32)

    def _score_window(self,
                      sal_map: np.ndarray,
                      edge_map: np.ndarray,
                      center_map: np.ndarray,
                      x: int, y: int, w: int, h: int) -> float:
        sal_region = sal_map[y:y+h, x:x+w]
        edge_region = edge_map[y:y+h, x:x+w]
        center_region = center_map[y:y+h, x:x+w]

        sal_score = float(np.mean(sal_region))
        edge_score = float(np.mean(edge_region))
        center_score = float(np.mean(center_region))

        return (self.saliency_weight * sal_score
                + self.edge_weight * edge_score
                + self.center_weight * center_score)

    def _find_best_crop(self,
                        sal_map: np.ndarray,
                        edge_map: np.ndarray,
                        center_map: np.ndarray,
                        img_h: int, img_w: int,
                        crop_w: int, crop_h: int) -> tuple[int, int, float]:
        stride_x = max(1, int(img_w * self.stride_fraction))
        stride_y = max(1, int(img_h * self.stride_fraction))

        best_score = -1.0
        best_x, best_y = 0, 0

        for y in range(0, img_h - crop_h + 1, stride_y):
            for x in range(0, img_w - crop_w + 1, stride_x):
                score = self._score_window(sal_map, edge_map, center_map, x, y, crop_w, crop_h)
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y

        return best_x, best_y, best_score

    def crop(self,
             image: np.ndarray,
             aspect_ratio: str = "4:3",
             sal_sr_weight: float = 0.4) -> tuple[np.ndarray, CropResult]:
        """
        Crop `image` to `aspect_ratio` using saliency + edge scoring.

        Returns the cropped image and a CropResult with metadata.
        """
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Unknown aspect ratio '{aspect_ratio}'. "
                             f"Choose from: {list(ASPECT_RATIOS.keys())}")

        img_h, img_w = image.shape[:2]
        ar_w, ar_h = ASPECT_RATIOS[aspect_ratio]

        sal_map = self._saliency.combined(image, sr_weight=sal_sr_weight)
        edge_map = self._edges.combined_edge_map(image)
        center_map = self._center_bias_map(img_h, img_w)

        crop_w, crop_h = self._compute_crop_size(img_w, img_h, ar_w, ar_h)

        x, y, score = self._find_best_crop(sal_map, edge_map, center_map,
                                           img_h, img_w, crop_w, crop_h)

        cropped = image[y:y+crop_h, x:x+crop_w]

        sal_coverage = float(np.mean(sal_map[y:y+crop_h, x:x+crop_w]))
        edge_coverage = float(np.mean(edge_map[y:y+crop_h, x:x+crop_w]))

        result = CropResult(x=x, y=y, w=crop_w, h=crop_h,
                            score=score,
                            aspect_ratio=aspect_ratio,
                            saliency_coverage=sal_coverage,
                            edge_coverage=edge_coverage)
        return cropped, result

    def _compute_crop_size(self, img_w: int, img_h: int,
                           ar_w: float, ar_h: float) -> tuple[int, int]:
        """Largest crop of given aspect ratio that fits in the image."""
        if img_w / img_h >= ar_w / ar_h:
            crop_h = img_h
            crop_w = int(round(crop_h * ar_w / ar_h))
        else:
            crop_w = img_w
            crop_h = int(round(crop_w * ar_h / ar_w))
        crop_w = min(crop_w, img_w)
        crop_h = min(crop_h, img_h)
        return crop_w, crop_h

    def crop_all_ratios(self,
                        image: np.ndarray) -> dict[str, tuple[np.ndarray, CropResult]]:
        """Compute crops for all supported aspect ratios."""
        return {ar: self.crop(image, ar) for ar in ASPECT_RATIOS}

    def best_ratio_crop(self, image: np.ndarray,
                        candidates: Optional[list[str]] = None) -> tuple[np.ndarray, CropResult]:
        """
        Among candidate aspect ratios, return the crop with the highest score.
        Useful when the desired output ratio is flexible.
        """
        ratios = candidates if candidates else list(ASPECT_RATIOS.keys())
        results = {ar: self.crop(image, ar) for ar in ratios}
        best_ar = max(results, key=lambda ar: results[ar][1].score)
        return results[best_ar]
