"""
Automatic Image Cropping Using Saliency Detection and Edge Analysis
===================================================================
Usage:
    python main.py --input <image_path> [--ratio 4:3] [--output results/]
    python main.py --input <image_path> --all-ratios
    python main.py --input <image_path> --best          # auto-pick best ratio
"""

import argparse
import sys
from pathlib import Path

import cv2

from src.saliency import SaliencyDetector
from src.edge_analysis import EdgeAnalyzer
from src.cropper import AutoCropper, ASPECT_RATIOS
from src.utils import (load_image, save_image, visualize_pipeline,
                       compare_crops, print_crop_report, draw_crop_overlay)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-crop images using saliency detection and edge analysis"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--ratio", "-r", default="4:3",
                        choices=list(ASPECT_RATIOS.keys()),
                        help="Target aspect ratio (default: 4:3)")
    parser.add_argument("--all-ratios", action="store_true",
                        help="Produce crops for all supported aspect ratios")
    parser.add_argument("--best", action="store_true",
                        help="Automatically choose the best-scoring aspect ratio")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip saving the pipeline visualisation figure")
    parser.add_argument("--sr-weight", type=float, default=0.4,
                        help="Spectral Residual weight in combined saliency (0-1)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading image: {args.input}")
    image = load_image(args.input)
    stem = Path(args.input).stem

    sal_detector = SaliencyDetector()
    edge_analyzer = EdgeAnalyzer()
    cropper = AutoCropper()

    sal_map = sal_detector.combined(image, sr_weight=args.sr_weight)
    edge_map = edge_analyzer.combined_edge_map(image)

    if args.all_ratios:
        crops = cropper.crop_all_ratios(image)
        for ar, (cropped, result) in crops.items():
            ar_str = ar.replace(":", "x")
            crop_path = str(out_dir / f"{stem}_crop_{ar_str}.jpg")
            save_image(cropped, crop_path)
            print_crop_report(result)
            print(f"  Saved: {crop_path}")

        compare_path = str(out_dir / f"{stem}_all_ratios.png")
        compare_crops(image, crops, output_path=compare_path)
        print(f"\nComparison grid saved: {compare_path}")

        # Use 4:3 for the pipeline visualisation
        main_result = crops.get("4:3", list(crops.values())[0])[1]

    elif args.best:
        cropped, result = cropper.best_ratio_crop(image)
        print_crop_report(result)
        crop_path = str(out_dir / f"{stem}_best_crop.jpg")
        save_image(cropped, crop_path)
        print(f"Best crop saved: {crop_path}")
        main_result = result

    else:
        cropped, result = cropper.crop(image, aspect_ratio=args.ratio,
                                       sal_sr_weight=args.sr_weight)
        print_crop_report(result)
        ar_str = args.ratio.replace(":", "x")
        crop_path = str(out_dir / f"{stem}_crop_{ar_str}.jpg")
        save_image(cropped, crop_path)
        print(f"Cropped image saved: {crop_path}")
        main_result = result

    if not args.no_viz:
        viz_path = str(out_dir / f"{stem}_pipeline.png")
        visualize_pipeline(image, sal_map, edge_map, main_result,
                           output_path=viz_path)
        print(f"Pipeline visualisation saved: {viz_path}")

        overlay = draw_crop_overlay(image, main_result)
        overlay_path = str(out_dir / f"{stem}_overlay.jpg")
        save_image(overlay, overlay_path)
        print(f"Overlay image saved: {overlay_path}")


if __name__ == "__main__":
    main()
