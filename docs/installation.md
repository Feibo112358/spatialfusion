# Installation

We provide pretrained weights for the **multimodal autoencoder (AE)** and **graph convolutional masked autoencoder (GCN)** under `data/`.

SpatialFusion depends on **PyTorch** and **DGL**, which have different builds for CPU and GPU systems. You can install it using **pip** or inside a **conda/mamba** environment.

---

### 1. Create mamba environment

```bash
mamba create -n spatialfusion python=3.10 -y
mamba activate spatialfusion
# Then install GPU or CPU version below
```

### 2. Install platform-specific libraries (GPU vs CPU)

#### GPU (CUDA 12.4)

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cu124
conda install -c dglteam/label/th24_cu124 dgl
```

**Note:** TorchText issues exist for this version:
[https://github.com/pytorch/text/issues/2272](https://github.com/pytorch/text/issues/2272) — this may affect scGPT.

---

#### GPU (CUDA 12.1) — *Recommended if using scGPT*

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
conda install -c dglteam/label/th21_cu121 dgl

# Optional: embeddings used by scGPT
pip install --no-cache-dir torchtext==0.18.0 torchdata==0.9.0

# Optional: UNI (H&E embedding model)
pip install timm
```

---

#### CPU-only

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cpu
conda install -c dglteam -c conda-forge dgl

# Optional, used for scGPT
pip install --no-cache-dir torchtext==0.18.0 torchdata==0.9.0

# Optional, used for UNI
pip install timm
```

> 💡 Replace `cu124` with the CUDA version matching your system (e.g., `cu121`).

---

### 3. Install SpatialFusion package

#### Basic installation — *Recommended for users*
```bash
cd spatialfusion/
pip install -e .
```
---
#### Developer installation - *Recommended for contributors*
Includes: `pytest`, `black`, `ruff`, `sphinx`, `matplotlib`, `seaborn`.

```bash
cd spatialfusion/
pip install -e ".[dev,docs]"
```



---

### 4. Verify Installation

```bash
python - <<'PY'
import torch, dgl, spatialfusion
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("DGL:", dgl.__version__)
print("SpatialFusion OK")
PY
```

---

### 5. Notes

* Default output directory is:

  ```
  $HOME/spatialfusion_runs
  ```

  Override with:

  ```
  export SPATIALFUSION_ROOT=/your/path
  ```
* CPU installations work everywhere but are significantly slower.