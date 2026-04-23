"""
Edge analysis module.

Uses Canny edge detection and Sobel gradient magnitude to build
an edge-saliency contribution map.
"""

import cv2
import numpy as np


class EdgeAnalyzer:
    """Detects and analyses edges to produce an edge-density saliency contribution."""

    def __init__(self,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 blur_kernel: int = 15):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel

    def canny_map(self, image: np.ndarray) -> np.ndarray:
        """Returns a binary Canny edge map (float32 in [0, 1])."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges.astype(np.float32) / 255.0

    def gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude map normalised to [0, 1]."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
        return mag

    def edge_density_map(self, image: np.ndarray) -> np.ndarray:
        """
        Smoothed edge-density map: blurs the Canny edge map so that
        regions rich in edges score higher. Used as an additive term
        in the crop scoring function.
        """
        edges = self.canny_map(image)
        density = cv2.GaussianBlur(edges,
                                   (self.blur_kernel, self.blur_kernel), 0)
        density = cv2.normalize(density, None, 0, 1, cv2.NORM_MINMAX)
        return density

    def combined_edge_map(self, image: np.ndarray,
                          canny_weight: float = 0.6) -> np.ndarray:
        """
        Weighted combination of edge density and gradient magnitude.
        Canny highlights crisp contours; Sobel captures soft transitions.
        """
        density = self.edge_density_map(image)
        gradient = self.gradient_magnitude(image)
        combined = canny_weight * density + (1 - canny_weight) * gradient
        combined = cv2.GaussianBlur(combined, (self.blur_kernel, self.blur_kernel), 0)
        combined = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
        return combined

    def adaptive_canny(self, image: np.ndarray) -> np.ndarray:
        """
        Canny with automatically chosen thresholds based on median pixel intensity.
        Robust to varying image brightness.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        v = float(np.median(blurred))
        sigma = 0.33
        low = max(0, int((1.0 - sigma) * v))
        high = min(255, int((1.0 + sigma) * v))
        edges = cv2.Canny(blurred, low, high)
        return edges.astype(np.float32) / 255.0
