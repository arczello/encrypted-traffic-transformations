# Structured 2D Data Transformations for Encrypted Video Traffic Classification

This repository provides the reference implementation of the experimental pipeline for comparing 1D-to-2D data transformation schemes in encrypted video traffic classification.

---

## 1. Overview of Transformation Schemes

The benchmark implements nine 1D-to-2D representations across mathematical time-series encodings and statistical feature–time representations:

* **Mathematical Encodings (Baselines):**
  * `GASF`: Gramian Angular Summation Field
  * `GADF`: Gramian Angular Difference Field
  * `MTF`: Markov Transition Field ($Q = 8$ quantile bins)
  * `RP`: Recurrence Plot ($\epsilon = 0.1$, phase-space recurrence)

* **Statistical and Structural Feature–Time Matrices:**
  * `Direct`: Unnormalized $5 \times W$ matrix of basic statistical window metrics (Mean, Max, Min, Std, Median)
  * `Differential`: First-order temporal differences capturing segment gradients
  * `Z-Score`: Flow-level row-wise standard normal standardization
  * `Feature-Grouped`: Semantically clustered row ordering aligning central tendency (Mean, Median) and dynamic range (Max, Min) adjacently
  * `Multi-Scale`: Dual-resolution representation combining multi-granularity sliding windows

---

## 2. Software Dependencies

Required Python packages are listed in `requirements.txt`:
* Python $\ge$ 3.10
* TensorFlow $\ge$ 2.15 / Keras 3
* NumPy, SciPy, Scikit-learn
* Pandas
* PyTS (Time-Series transformation toolkit)

Install dependencies into your target environment via:
```bash
pip install -r requirements.txt
```

---

## 3. Data Specification and Ingestion

The evaluation routines expect pre-segmented numerical NumPy arrays:
* **Feature Array:** Shape `(N, T)`, where $N$ denotes the number of flow segments and $T$ denotes the segment length in seconds (e.g., $T = 128$).
* **Label Array:** 1D integer vector of shape `(N,)` corresponding to the traffic class indices.

Raw traffic traces can be obtained from their respective public repositories:
* Video traces: [SysSec-KAIST / WatchingTheWatchers](https://github.com/SysSec-KAIST/WatchingTheWatchers)
* Non-video traces: [arczello / non_video_traces](https://github.com/arczello/non_video_traces)

---

## 4. Module Reference and API Usage

The repository is structured as modular Python libraries:

```
code_publish/
├── transformations.py      # Core implementations of the 9 transformation schemes
├── models.py                # Baseline CNN, ResNet-50, and ViT-Base/16 architectures
├── train_and_evaluate.py   # Stratified splitting, checkpointing, and evaluation protocol
├── explainability.py       # Grad-CAM heatmap extraction and AtB calculation
├── requirements.txt        # Package dependency specification
└── README.md               # Methodological reference and API documentation
```

### Transformation API (`transformations.py`)

Transform a batch of 1D intensity sequences into 2D representations:

```python
import numpy as np
from transformations import apply_transformation_batch

# X_raw: numpy.ndarray of shape (N, T)
x_2d = apply_transformation_batch(
    x_raw, scheme="Feature-Grouped", window_size=10, stride=5
)
# Returns numpy.ndarray of shape (N, 5, W)
```

### Model Architectures (`models.py`)

Instantiate standardized baseline CNN or benchmark Large Vision Models:

```python
from models import build_cnn, build_resnet50, build_vit

# Standardized baseline 2D CNN
cnn_model = build_cnn(input_shape=(5, 24, 1), num_classes=6)

# Benchmark Large Vision Models
resnet_model = build_resnet50(input_shape=(224, 224, 3), num_classes=6)
vit_model = build_vit(input_shape=(224, 224, 3), num_classes=6)
```

### Training and Evaluation (`train_and_evaluate.py`)

Execute stratified 70/15/15 cross-validation with early stopping and checkpoint restoration:

```python
from train_and_evaluate import evaluate_scheme, set_seed

set_seed(42)

results = evaluate_scheme(
    raw_x=x_raw,
    labels=y_labels,
    num_classes=6,
    scheme="Feature-Grouped",
    model_type="cnn",
    epochs=50,
    batch_size=32,
    lr=0.001,
    patience=10,
    seed=42,
)

print(f"Global Macro F1: {results['global_f1']:.4f}")
print(f"Inter-Bin F1:   {results['inter_bin_f1']:.4f}")
```

### Saliency and Explainability (`explainability.py`)

Extract Grad-CAM activation heatmaps and compute the Attention-to-Burst (AtB) ratio:

```python
from explainability import compute_atb_ratio, compute_gradcam_heatmap, upsample_heatmap

# sample_tensor: shape (1, H, W, 1)
raw_cam = compute_gradcam_heatmap(
    cnn_model, sample_tensor, last_conv_layer_name="last_conv"
)
cam_upsampled = upsample_heatmap(raw_cam, target_shape=(H, W))

temporal_attention = np.sum(cam_upsampled, axis=0)
atb = compute_atb_ratio(raw_series_1d, temporal_attention)
print(f"Attention-to-Burst (AtB) Ratio: {atb:.2f}%")
```

---

## 5. Citation

If you find this benchmark or transformation methodology useful in your research, please cite:

```bibtex
@article{biernacki2026structured,
  title={Structured 2D Data Transformations for Resource-Efficient and Explainable CNN-Based Encrypted Video Traffic Classification},
  author={Biernacki, Arkadiusz},
  journal={Electronics},
  year={2026}
}
```
