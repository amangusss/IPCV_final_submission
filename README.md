# Automatic Image Cropping Using Saliency Detection and Edge Analysis

Applied Image Processing — Final Project

## Overview

This project implements an end-to-end, training-free automatic image cropping pipeline that combines:

- **Spectral Residual (SR) Saliency** — global contrast novelty (Hou & Zhang, 2007)
- **Fine-Grained (FG) Saliency** — local contrast saliency (Montabone & Soto, 2010)
- **Canny Edge Detection + Sobel Gradient** — structural richness map
- **Gaussian Centre-Bias Prior** — photographic composition preference

Crop windows are scored by a weighted sum of these signals. The highest-scoring window for the requested aspect ratio is returned as the output crop.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Basic crop

```bash
python main.py --input path/to/image.jpg
python main.py --input path/to/image.jpg --ratio 16:9
```

### All aspect ratios

```bash
python main.py --input path/to/image.jpg --all-ratios
```

### Auto-select best ratio

```bash
python main.py --input path/to/image.jpg --best
```

### Built-in demo (no external images needed)

```bash
python demo/demo.py
```

Results are saved to `results/demo/`.

### Evaluation (baseline comparison)

```bash
python evaluate.py
```

Results are saved to `results/evaluation/`.

### Build presentation

```bash
python build_presentation.py
```

Output: `presentation/AutoCropping_Presentation.pptx` (14 slides with embedded charts and pipeline images).

## Supported Aspect Ratios

`1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `3:2`, `2:3`

## Project Structure

```
FinalTask/
├── src/
│   ├── saliency.py        # SaliencyDetector
│   ├── edge_analysis.py   # EdgeAnalyzer
│   ├── cropper.py         # AutoCropper
│   └── utils.py           # Visualisation helpers
├── demo/
│   └── demo.py            # Synthetic scene demo
├── report/
│   └── final_report.md    # Full project report
├── results/               # Generated outputs
├── main.py                # CLI entry point
├── evaluate.py            # Baseline comparison
└── requirements.txt
```

## Output Files (per image)

| File | Description |
|------|-------------|
| `*_crop_<ratio>.jpg` | Cropped image |
| `*_pipeline.png` | 4-panel diagnostic: original, saliency, edges, crop |
| `*_overlay.jpg` | Original with crop rectangle drawn |
| `*_all_ratios.png` | Grid of all aspect ratio crops |
