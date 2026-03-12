# Pix2Text: Point Clouds → Text Descriptions

**Pix2Text** learns a shared embedding space for **3D point clouds** and **natural language**, so that a 3D shape can be "described" by matching it to the closest text (e.g. class name) in that space. It uses contrastive learning (InfoNCE) on [ModelNet40](https://modelnet.cs.princeton.edu/) with a PointNet-style encoder for geometry and DistilBERT for text.

---

## Features

- **Point cloud encoder**: PointNet-style backbone that maps 3D point sets to embeddings
- **Text encoder**: Pretrained DistilBERT + linear projection for class descriptions
- **Contrastive training**: InfoNCE loss aligning point cloud and text in a shared space
- **Inference**: Describe a point cloud by retrieving the best-matching class (or top‑k)
- **Evaluation**: Top-1 and Top-5 retrieval accuracy
- **Visualization**: 3D point cloud plots (Plotly) and side-by-side comparison with predictions

---

## Requirements

- **Python** 3.8+
- **PyTorch** (with CUDA recommended for training)
- **Transformers** (Hugging Face) for DistilBERT
- **NumPy**, **tqdm**, **Plotly** (for visualization)

---

## Installation

1. **Clone or navigate to the project**  
   ```bash
   cd pix2text
   ```

2. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux / macOS
   ```

3. **Install dependencies**  
   ```bash
   pip install torch numpy transformers tqdm plotly
   ```

4. **Install the package in editable mode** (so `pix2text` is importable)  
   ```bash
   pip install -e .
   ```  
   If you don't have a `setup.py` or `pyproject.toml`, run from the project root and add it to `PYTHONPATH`, or install manually:  
   ```bash
   pip install torch numpy transformers tqdm plotly
   ```
   Then in your script/notebook, ensure the parent of the `pix2text` folder is on `sys.path`.

---

## Dataset: ModelNet40

- Download **ModelNet40** (OFF meshes) and place it so that the structure is:
  ```
  ModelNet40/
  ├── airplane/
  │   ├── train/
  │   │   └── *.off
  │   └── test/
  │       └── *.off
  ├── bathtub/
  ├── bed/
  └── ... (40 classes)
  ```

- Set the path in `pix2text.config` (see [Configuration](#configuration)).

- The code can **build a precomputed NPZ** from OFF files (with FPS sampling and normalization) for faster training.

---

## Configuration

Edit `pix2text/config.py` (or override in your script/notebook):

| Variable | Default | Description |
|----------|---------|-------------|
| `DRIVE_MODELNET_PATH` | `"/content/ModelNet40"` | Root path to ModelNet40 OFF files |
| `EXTRACTED_DATASET_PATH` | `"/content/ModelNet40"` | Same as above (used for building NPZ) |
| `PRECOMPUTED_PATH` | `Path("/content/point2text_features.npz")` | Output path for precomputed NPZ |
| `USE_NUM_POINTS` | `1024` | Points per shape (sampled per batch at train time) |
| `BATCH_SIZE` | `128` | Training batch size |
| `EMBED_DIM` | `256` | Shared embedding dimension |
| `LR` | `1e-3` | Learning rate |
| `NUM_EPOCHS` | `200` | Number of training epochs |
| `SAVE_DIR` | `"/content/point2text_checkpoints"` | Checkpoint directory |

Call `config.setup(seed=42)` before training to create `SAVE_DIR`, set device (CUDA/CPU), and fix RNG seeds.

---

## Running in Google Colab

If you clone this repo into Colab, follow these steps. You can paste each block into a separate cell.

### 1. Clone the repo and install

```python
# Clone (replace with your repo URL)
!git clone https://github.com/YOUR_USERNAME/pix2text.git
%cd pix2text

# Install the package and dependencies (Colab has PyTorch; we need transformers, plotly, etc.)
!pip install -e .
```

### 2. Get ModelNet40

You need the ModelNet40 OFF files in Colab. Two options:

**Option A — Upload from Google Drive**  
If you have `ModelNet40.zip` in your Drive (e.g. at `MyDrive/ModelNet40.zip`):

```python
from google.colab import drive
drive.mount("/content/drive")

# Unzip to /content (config expects /content/ModelNet40)
!unzip -q "/content/drive/MyDrive/ModelNet40.zip" -d /content
# If the zip contains a top-level folder "ModelNet40", you now have /content/ModelNet40
```

**Option B — Download (if you have a direct link)**  
Some mirrors host ModelNet40; adjust the URL if you have one:

```python
# Example: download and unzip (URL may vary)
# !wget -q "https://some-mirror.com/ModelNet40.zip" -O /content/ModelNet40.zip
# !unzip -q /content/ModelNet40.zip -d /content
```

After this, ensure the folder `/content/ModelNet40` exists and contains class folders (`airplane/`, `bed/`, etc.) each with `train/` and `test/` subfolders of `.off` files.

### 3. Set paths and build the NPZ (one-time)

The default `config` uses `/content/ModelNet40` and `/content/point2text_features.npz`. If your data is elsewhere (e.g. on Drive), override before calling `setup()`:

```python
import numpy as np
from pathlib import Path
import pix2text.config as config

# Optional: use a path on Drive to avoid re-uploading every run
# config.EXTRACTED_DATASET_PATH = "/content/drive/MyDrive/ModelNet40"
# config.PRECOMPUTED_PATH = Path("/content/drive/MyDrive/point2text_features.npz")
# config.SAVE_DIR = "/content/drive/MyDrive/point2text_checkpoints"

config.setup()

from pix2text.data import get_classes, build_npz_fast

classes = get_classes(config.EXTRACTED_DATASET_PATH)
print("Classes:", len(classes), classes[:5])

# Build NPZ from OFF files (run once; then you can skip this cell)
build_npz_fast(
    config.EXTRACTED_DATASET_PATH,
    str(config.PRECOMPUTED_PATH),
    classes,
    points_per_model=config.USE_NUM_POINTS,
    val_ratio=0.1,
    device=config.device,
)
```

### 4. Load data and train

```python
import torch
import pix2text.config as config
from pix2text.data import get_classes
from pix2text.models import PointNetFeat, TextEncoder, MLP, init_text_models
from pix2text.training import run_training

config.setup()

# Load precomputed NPZ
data = np.load(config.PRECOMPUTED_PATH)
train_points = torch.tensor(data["train_points"], dtype=torch.float32)
train_labels = torch.tensor(data["train_labels"], dtype=torch.long)
val_points   = torch.tensor(data["val_points"], dtype=torch.float32)
val_labels   = torch.tensor(data["val_labels"], dtype=torch.long)

classes = get_classes(config.EXTRACTED_DATASET_PATH)
CLASS_DESC = [f"A 3D object from the class {c}." for c in classes]

# Models
init_text_models(config.device)
pc_encoder = PointNetFeat(out_dim=config.EMBED_DIM).to(config.device)
txt_encoder = TextEncoder(out_dim=config.EMBED_DIM).to(config.device)
pc_proj = MLP(config.EMBED_DIM, config.EMBED_DIM).to(config.device)
txt_proj = MLP(config.EMBED_DIM, config.EMBED_DIM).to(config.device)
opt = torch.optim.Adam(
    list(pc_encoder.parameters()) + list(txt_encoder.parameters()) +
    list(pc_proj.parameters()) + list(txt_proj.parameters()),
    lr=config.LR,
)

# Train (saves to config.SAVE_DIR; resume from latest ckpt_*.pt if present)
run_training(
    pc_encoder, txt_encoder, pc_proj, txt_proj, opt,
    train_points, train_labels, val_points, val_labels,
    CLASS_DESC, config.device,
    BATCH_SIZE=config.BATCH_SIZE,
    NUM_EPOCHS=config.NUM_EPOCHS,
)
```

To train for fewer epochs in Colab (e.g. 10), pass `NUM_EPOCHS=10` to `run_training`.

### 5. Inference and visualization (after training)

```python
from pix2text.inference import describe_pointcloud_topk
from pix2text.viz import plot_side_by_side
import numpy as np

# Load best checkpoint if you restarted runtime
# ckpt = torch.load(f"{config.SAVE_DIR}/best_ckpt.pt", map_location=config.device)
# pc_encoder.load_state_dict(ckpt["pc_encoder"])
# txt_encoder.load_state_dict(ckpt["txt_encoder"])
# pc_proj.load_state_dict(ckpt["pc_proj"])
# txt_proj.load_state_dict(ckpt["txt_proj"])

# Pick a validation sample
idx = 0
pc_np = val_points[idx].numpy()
gt_class = classes[val_labels[idx].item()]

top_classes, top_scores, top_descriptions, pc_sampled = describe_pointcloud_topk(
    pc_np, pc_encoder, pc_proj, txt_encoder, txt_proj,
    classes, config.device, use_num_points=1024, top_k=5
)
print("Ground truth:", gt_class)
print("Top-5:", top_descriptions)

plot_side_by_side(pc_np, pc_sampled, gt_class, top_classes, top_scores)
```

**Summary:** Clone repo → install with `pip install -e .` → put ModelNet40 in `/content/ModelNet40` (or set paths) → build NPZ once → run training → run inference/viz.

---

## Project structure

```
pix2text/
├── __init__.py          # Package exports
├── config.py            # Paths, hyperparameters, device, setup()
├── data/
│   ├── utils.py         # load_off(), normalize_pc(), batched_fps()
│   └── modelnet.py      # get_classes(), build_npz_fast(), ModelNetNPZ
├── models/
│   ├── pointnet.py      # PointNetFeat (point cloud encoder)
│   ├── text_encoder.py  # TextEncoder, init_text_models (DistilBERT)
│   └── mlp.py           # MLP projection head
├── training/
│   ├── loss.py          # info_nce()
│   └── loop.py          # run_training(), checkpoint resume
├── evaluation/
│   └── retrieval.py     # retrieval_accuracy() (top-1, top-5)
├── inference/
│   └── describe.py      # describe_pointcloud(), describe_pointcloud_topk()
└── viz/
    └── plot.py          # plot_pointcloud(), plot_side_by_side()
```

---

## Usage

### 1. Build precomputed NPZ (optional but recommended)

From OFF files to a single NPZ with train/val split:

```python
from pix2text import config
from pix2text.data import get_classes, build_npz_fast

config.setup()
classes = get_classes(config.EXTRACTED_DATASET_PATH)
build_npz_fast(
    config.EXTRACTED_DATASET_PATH,
    config.PRECOMPUTED_PATH,
    classes,
    points_per_model=config.USE_NUM_POINTS,
    val_ratio=0.1,
    device=config.device,
)
```

### 2. Training

Load data, build models, and run the training loop (with automatic checkpoint resume):

```python
import torch
from pix2text import config
from pix2text.data import get_classes, build_npz_fast  # or load existing NPZ
from pix2text.models import PointNetFeat, TextEncoder, MLP, init_text_models

config.setup()
# Load NPZ: train_points, train_labels, val_points, val_labels
data = np.load(config.PRECOMPUTED_PATH)
train_points = torch.tensor(data["train_points"], dtype=torch.float32)
train_labels = torch.tensor(data["train_labels"], dtype=torch.long)
val_points   = torch.tensor(data["val_points"], dtype=torch.float32)
val_labels   = torch.tensor(data["val_labels"], dtype=torch.long)

classes = get_classes(config.EXTRACTED_DATASET_PATH)
CLASS_DESC = [f"A 3D object from the class {c}." for c in classes]

init_text_models(config.device)
pc_encoder = PointNetFeat(out_dim=config.EMBED_DIM).to(config.device)
txt_encoder = TextEncoder(out_dim=config.EMBED_DIM).to(config.device)
pc_proj = MLP(config.EMBED_DIM, config.EMBED_DIM).to(config.device)
txt_proj = MLP(config.EMBED_DIM, config.EMBED_DIM).to(config.device)
opt = torch.optim.Adam(
    list(pc_encoder.parameters()) + list(txt_encoder.parameters()) +
    list(pc_proj.parameters()) + list(txt_proj.parameters()),
    lr=config.LR
)

from pix2text.training import run_training
run_training(
    pc_encoder, txt_encoder, pc_proj, txt_proj, opt,
    train_points, train_labels, val_points, val_labels,
    CLASS_DESC, config.device,
    BATCH_SIZE=config.BATCH_SIZE,
    NUM_EPOCHS=config.NUM_EPOCHS,
)
```

Checkpoints are saved as `ckpt_<epoch>.pt` and the best validation loss as `best_ckpt.pt` in `config.SAVE_DIR`.

### 3. Inference: describe a point cloud

Single best class:

```python
from pix2text.inference import describe_pointcloud

pred_class, score = describe_pointcloud(
    pc_np, pc_encoder, pc_proj, txt_encoder, txt_proj,
    classes, config.device, use_num_points=1024
)
print(pred_class, score)
```

Top‑k classes and scores:

```python
from pix2text.inference import describe_pointcloud_topk

top_classes, top_scores, top_descriptions, pc_sampled = describe_pointcloud_topk(
    pc_tensor, pc_encoder, pc_proj, txt_encoder, txt_proj,
    classes, config.device, use_num_points=1024, top_k=5
)
```

### 4. Evaluation: retrieval accuracy

Compute top-1 and top-5 retrieval accuracy (point cloud → text) on validation data:

```python
from pix2text.evaluation import retrieval_accuracy

# Get embeddings for val point clouds and for all class texts
with torch.no_grad():
    pc_emb = pc_proj(pc_encoder(val_points.to(device)))
    texts = [f"A 3D object from the class {c}." for c in classes]
    txt_emb = txt_proj(txt_encoder(texts))
# train_labels_ref: indices 0..39 for the 40 class texts
train_labels_ref = torch.arange(len(classes), device=device)
acc = retrieval_accuracy(pc_emb, txt_emb, val_labels, train_labels_ref, topk=(1, 5))
print("Top-1:", acc[1], "Top-5:", acc[5])
```

### 5. Visualization

```python
from pix2text.viz import plot_pointcloud, plot_side_by_side

plot_pointcloud(pc_np, title="My shape")
plot_side_by_side(gt_pc, pred_pc, gt_class, top_classes, top_scores)
```

---

## How it works

1. **Point clouds** are normalized and (optionally) sub-sampled with FPS to a fixed number of points (e.g. 1024). **PointNetFeat** produces a global feature vector, then an **MLP** projects it into the shared space.
2. **Text** for each class is of the form: `"A 3D object from the class <class_name>."` **DistilBERT** encodes it; **TextEncoder** (linear layer) maps the [CLS] output to the same dimension; another **MLP** projects to the shared space.
3. **InfoNCE** treats each point cloud–text pair in the batch as positive and all others as negatives (symmetric for pc→text and text→pc). The model is trained to maximize cosine similarity for positives and minimize it for negatives.
4. **At test time**, a point cloud is embedded and compared via cosine similarity to all class text embeddings; the best match (or top‑k) is returned as the description.

---

## License

See the repository for license information.
