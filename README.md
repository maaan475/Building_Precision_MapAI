# Building Precision: Efficient Encoder–Decoder Networks for Remote Sensing Based on Aerial RGB and LiDAR Data

**IEEE Access**, vol. 12, pp. 60329–60346, 2024  
[DOI: 10.1109/ACCESS.2024.3391416](https://doi.org/10.1109/ACCESS.2024.3391416) · [IEEE Xplore](https://ieeexplore.ieee.org/document/10507195)

**Authors:** Muhammad Sulaiman, Erik Finnesand, Mina Farmanbar, Ahmed Nabil Belbachir, Chunming Rong

---

## Abstract

Precision in building delineation is essential for population analysis, urban management, policy making, and disaster response. Deep learning–based semantic segmentation has advanced automatic building extraction from remote sensing imagery, yet many state-of-the-art methods remain suboptimal for accurate footprint and boundary recovery. This repository implements the official experimental pipeline accompanying the IEEE Access paper, which adapts **U-Net** and **Context-Transfer U-Net (CT-UNet)** to fuse aerial **RGB** with **LiDAR** for building segmentation on the **MapAI** benchmark.

Models are evaluated with **Intersection over Union (IoU)** for region overlap and **Boundary IoU (BIoU)** for edge fidelity. The official competition score is the average of IoU and BIoU. Fusing LiDAR with RGB improves the combined score by **13.33%** relative to RGB alone (IoU +9.53%, BIoU +17.14%). Weighted ensembles with test-time augmentation (TTA) further improve robustness over single models.

---

## Highlights

| Contribution | Description |
|---|---|
| Dual-task design | Task 1: RGB-only (3-channel); Task 2: RGB + LiDAR (4-channel) |
| Architectures | Modified U-Net and CT-UNet with Dense Boundary Blocks (DBB) and Spatial Channel Attention Blocks (SCAB) |
| Backbones | EfficientNetB4, EfficientNetV2S, ResNet50V2, DenseNet201 (ImageNet transfer learning for Task 1) |
| Regularization | Training-time geometric augmentation; batch normalization after ReLU; Dice loss |
| Inference | Test-time augmentation (horizontal/vertical flips) and weighted 3-model ensembles |
| Evaluation | MapAI official metrics: IoU, BIoU, and Score = (IoU + BIoU) / 2 |

---

## Dataset: MapAI

The [MapAI](https://huggingface.co/datasets/sjyhne/mapai_dataset) dataset supports the NORA / University of Agder *Precision in Building Segmentation* challenge. It contains aerial RGB imagery, LiDAR height maps, and binary building masks spanning diverse urban and rural scenes (Denmark for train/val; Norway for test).

| Split | Count | Notes |
|---|---|---|
| Train | 7,500 | Denmark |
| Validation | 1,500 | Denmark |
| Task 1 test | 1,368 | Norway (RGB) |
| Task 2 test | 978 | Norway (RGB + LiDAR) |

Non-building objects (trees, vehicles, roads, solar panels, power lines) share colors and textures with rooftops, making LiDAR especially valuable for disambiguation.

**Preprocessing (as implemented):** all modalities are resized to **512 × 512**. RGB and masks are stored as PNG; RGB+LiDAR stacks are saved as 4-channel TIFF (`uint8`).

---

## Method Overview

```
MapAI (Hugging Face)
        │
        ▼
 Load_Dataset.py  ──►  Preprocess_Dataset.py  ──►  512×512 RGB / RGB+LiDAR / masks
        │
        ▼
 Train_Model.py   ──►  U-Net / CT-UNet (+ backbones)  ──►  models/task{1,2}/
        │
        ▼
 Test_Model.py / Model_Ensemble.py  ──►  IoU · BIoU · Score  (+ optional TTA)
```

### Architectures

1. **U-Net** (`code/UNet.py`)  
   Classic encoder–decoder with skip connections. Modifications include batch normalization after ReLU, 3×3 transposed convolutions, and a 1×1 sigmoid head. Backbone variants replace the encoder with EfficientNetB4, EfficientNetV2S, ResNet50V2, or DenseNet201.

2. **CT-UNet** (`code/CTUNet.py`)  
   Context-Transfer U-Net with:
   - **Dense Blocks (DB)** for feature reuse  
   - **Dense Boundary Blocks (DBB)** for sharper boundaries  
   - **Spatial Channel Attention Blocks (SCAB)** for intra-class consistency  

   The same four backbones are supported as encoders.

### Training configuration (code defaults)

| Setting | U-Net | CT-UNet |
|---|---|---|
| Optimizer | Adam | Adam |
| Initial learning rate | 1×10⁻⁴ | 5×10⁻⁵ |
| Batch size | 6 (baseline) / 12 (backbone) | 10–12 |
| Loss | Dice coefficient loss | Dice coefficient loss |
| Metric (train) | Jaccard / IoU | Jaccard / IoU |
| Max epochs | 100 | 100 |
| Early stopping | patience = 10 (on `val_loss`) | same |
| LR schedule | ReduceLROnPlateau (factor 0.1, patience 5) | same |
| Input size | 512 × 512 × 3 (Task 1) or × 4 (Task 2) | same |
| Pretrained weights | ImageNet (Task 1); from scratch (Task 2) | same |

**Augmentation:** rotation (90°), width/height shift (0.3), shear (0.5), zoom (0.3), horizontal/vertical flip, reflect fill. Masks are binarized after augmentation.

**TTA:** average of original, left–right flip, up–down flip, and both flips. Thresholds: **0.5** without TTA, **0.3** with TTA.

---

## Repository Structure

```
Building_Precision_MapAI/
├── README.md
├── requirements.txt
├── code/
│   ├── Load_Dataset.py       # Download MapAI from Hugging Face
│   ├── Preprocess_Dataset.py # Resize & build RGB / RGB+LiDAR folders
│   ├── UNet.py               # U-Net + backbone variants
│   ├── CTUNet.py             # CT-UNet + backbone variants
│   ├── Train_Model.py        # Interactive training entry point
│   ├── Test_Model.py         # Single-model evaluation
│   ├── Model_Ensemble.py     # Weighted / search ensembles
│   ├── Loss_Metrics.py       # Dice / Jaccard loss & metrics
│   ├── eval_functions.py     # Official IoU, BIoU, Score
│   └── utils.py              # Test loaders & TTA helpers
├── dataset/                  # Created at runtime (gitignored)
├── models/                   # Saved checkpoints (gitignored)
└── logs/                     # CSV training logs
```

---

## Requirements

- Python 3.8+ (recommended)
- CUDA-capable GPU strongly recommended (TensorFlow 2.11)
- Dependencies listed in `requirements.txt`:

```
datasets==2.8.0
numpy==1.23.5
opencv_python==4.6.0.66
Pillow==9.4.0
tensorflow==2.11.0
tifffile==2023.2.3
tqdm==4.64.1
```

### Installation

```bash
git clone <repository-url>
cd Building_Precision_MapAI
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing the Pipeline

All scripts are interactive and expect to be run from the `code/` directory.

### 1. Download MapAI

```bash
cd code
python Load_Dataset.py
```

Writes `dataset/MapAI/{train,validation,task1_test,task2_test}/{image,mask,lidar}/`.

### 2. Preprocess

```bash
python Preprocess_Dataset.py
```

Produces `dataset/MapAI/preprocessed_*` at 512 × 512 with RGB and RGB+LiDAR (`.tif`) inputs.

### 3. Train

```bash
python Train_Model.py
```

Prompts for:

1. GPU index  
2. Task — `1` RGB or `2` RGB+LiDAR  
3. Model architecture:

| Input | Model |
|---|---|
| 1 | U-Net |
| 2 | EfficientNetB4 U-Net |
| 3 | EfficientNetV2S U-Net |
| 4 | ResNet50V2 U-Net |
| 5 | DenseNet201 U-Net |
| 6 | EfficientNetB4 CT-UNet |
| 7 | EfficientNetV2S CT-UNet |
| 8 | ResNet50V2 CT-UNet |
| 9 | DenseNet201 CT-UNet |

4. Save name for the checkpoint  

Artifacts:

- Best checkpoint: `models/task{1|2}/<name>_Checkpoint.h5`
- Final model: `models/task{1|2}/<name>/`
- Training log: `logs/<name>_log`

### 4. Evaluate a single model

```bash
python Test_Model.py
```

Select task, test vs. validation split, TTA on/off, and the model folder name under `models/task{1|2}/`. Prints `score`, `iou`, and `biou`.

### 5. Ensemble evaluation

```bash
python Model_Ensemble.py
```

Modes:

1. Fixed weights for three models  
2. Grid search over weights summing to 1.0 for three models  
3. Exhaustive search over all 3-model combinations from a candidate list (reports top 10)

---

## Evaluation Metrics

Official MapAI metrics (implemented in `eval_functions.py`):

\[
\mathrm{IoU} = \frac{|I_g \cap I_p|}{|I_g \cup I_p|}
\qquad
\mathrm{BIoU} = \frac{|(I_{gd} \cap I_g) \cap (I_{pd} \cap I_p)|}{|(I_{gd} \cap I_g) \cup (I_{pd} \cap I_p)|}
\qquad
\mathrm{Score} = \frac{\mathrm{IoU} + \mathrm{BIoU}}{2}
\]

Boundary masks use morphological erosion with dilation ratio \(0.02 \times\) image diagonal (MapAI competition toolkit).

---

## Key Results (from the paper)

### Effect of LiDAR (best proposed configurations)

| Setting | IoU | BIoU | Score | Δ vs RGB |
|---|---|---|---|---|
| Task 1 (RGB) | — | — | baseline | — |
| Task 2 (RGB + LiDAR) | +9.53% | +17.14% | **+13.33%** | vs Task 1 |

### Task 1 — best ensemble with TTA (threshold 0.3)

| Models (weights) | IoU | BIoU | Score |
|---|---|---|---|
| U-Net DenseNet201 (0.4) + CT-UNet EfficientNetB4 (0.3) + CT-UNet EfficientNetV2S (0.3) | 0.8011 | 0.6295 | **0.7153** |

TTA improved IoU / BIoU by approximately **0.61% / 0.74%** on the top ensembles.

### Qualitative findings

- Ensembles outperform single models; TTA yields consistent further gains.  
- **CT-UNet** is stronger on Task 1 (RGB); **U-Net** is stronger on Task 2 (RGB+LiDAR).  
- Dense Boundary / attention blocks matter less when LiDAR is available, because height cues simplify foreground–background separation.  
- LiDAR reduces false positives from trucks, shadows, and tree-obscured roofs.

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{Sulaiman2024BuildingPrecision,
  author  = {Sulaiman, Muhammad and Finnesand, Erik and Farmanbar, Mina
             and Belbachir, Ahmed Nabil and Rong, Chunming},
  title   = {Building Precision: Efficient Encoder--Decoder Networks for
             Remote Sensing Based on Aerial {RGB} and {LiDAR} Data},
  journal = {IEEE Access},
  year    = {2024},
  volume  = {12},
  pages   = {60329--60346},
  doi     = {10.1109/ACCESS.2024.3391416}
}
```

### Related resources

- MapAI dataset: [Hugging Face — `sjyhne/mapai_dataset`](https://huggingface.co/datasets/sjyhne/mapai_dataset)  
- MapAI evaluation toolkit: [Sjyhne/MapAI-Competition](https://github.com/Sjyhne/MapAI-Competition)  
- CT-UNet (Liu et al., 2021): Context-Transfer-UNet for building segmentation  

---

## Acknowledgments

This work builds on the MapAI *Precision in Building Segmentation* challenge organized by the Norwegian Artificial Intelligence Research Consortium (NORA) in cooperation with the University of Agder (CAIR). Evaluation routines adapt the official MapAI competition toolkit.

---

## License

Please refer to the repository license file (if provided) and the IEEE Access article copyright for reuse of paper content. Dataset usage is subject to the MapAI / Hugging Face dataset terms.
