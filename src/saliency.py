"""
Saliency detection module.

Implements two complementary saliency methods:
  - Spectral Residual (SR): Hou & Zhang, CVPR 2007
  - Fine-Grained (FG): Montabone & Soto, 2010 (via OpenCV)
"""

import cv2
import numpy as np


class SaliencyDetector:
    """Computes saliency maps using Spectral Residual and Fine-Grained methods."""

    def __init__(self, blur_kernel: int = 11, sr_scale: int = 64):
        self.blur_kernel = blur_kernel
        self.sr_scale = sr_scale

    def spectral_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Spectral Residual saliency map.

        Steps:
          1. Convert to grayscale and resize for efficiency.
          2. Apply log spectrum and smooth it.
          3. Spectral residual = log spectrum - smoothed log spectrum.
          4. IFT back to spatial domain; saliency = magnitude squared.
        """
        detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, saliency_map = detector.computeSaliency(gray)
        saliency_map = cv2.GaussianBlur(saliency_map.astype(np.float32),
                                        (self.blur_kernel, self.blur_kernel), 0)
        saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
        return saliency_map

    def fine_grained(self, image: np.ndarray) -> np.ndarray:
        """
        Fine-Grained saliency map via local contrast analysis.
        More stable for structured scenes with prominent objects.
        """
        detector = cv2.saliency.StaticSaliencyFineGrained_create()
        _, saliency_map = detector.computeSaliency(image)
        saliency_map = saliency_map.astype(np.float32)
        saliency_map = cv2.GaussianBlur(saliency_map,
                                        (self.blur_kernel, self.blur_kernel), 0)
        saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
        return saliency_map

    def combined(self, image: np.ndarray, sr_weight: float = 0.4) -> np.ndarray:
        """
        Weighted combination of SR and FG saliency maps.
        SR is good at global contrast; FG captures local structure.
        """
        sr = self.spectral_residual(image)
        fg = self.fine_grained(image)
        combined = sr_weight * sr + (1 - sr_weight) * fg
        combined = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
        return combined

    def get_salient_region(self, saliency_map: np.ndarray,
                           threshold: float = 0.5) -> tuple[int, int, int, int]:
        """
        Returns bounding box (x, y, w, h) of the most salient region
        using the saliency map thresholded at `threshold`.
        """
        binary = (saliency_map > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = saliency_map.shape[:2]
            return 0, 0, w, h
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)
