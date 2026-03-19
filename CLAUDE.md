# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpatialFusion is a deep learning package for spatial omics data analysis that integrates spatial transcriptomics (ST) with H&E histopathology at single-cell resolution. It uses a two-stage embedding pipeline:

1. **Paired Autoencoder (AE):** Combines UNI morphology embeddings + scGPT molecular embeddings into joint latent representations
2. **Graph Convolutional Network (GCN):** Refines embeddings using spatial graph topology to discover reproducible spatial niches

## Build & Development Commands

```bash
# Install in development mode
pip install -e .
pip install -e ".[dev,docs]"

# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ -v --cov=src/spatialfusion

# Run a single test file
pytest tests/test_basic.py

# Run a single test function
pytest tests/test_basic.py::test_safe_standardize

# Build package
python -m build

# Validate built package
python -m twine check dist/*

# Serve docs locally
mkdocs serve
```

## Architecture

### Source Layout

All source code is under `src/spatialfusion/` (src-layout packaging).

### Key Modules

- **`embed/embed.py`** — Main orchestration. `run_full_embedding()` is the primary entry point, supporting both in-memory (`AEInputs` dict) and disk-based (`sample_list` + `base_path`) data loading modes.
- **`models/multi_ae.py`** — `PairedAE` autoencoder with dual encoders/decoders for learning joint multimodal embeddings. Supports single-modality (UNI-only) mode when `d2=None`.
- **`models/gcn.py`** — `GCNAutoencoder` with learnable mask tokens, stacked GCN layers, and optional classification head.
- **`models/baseline_multi_ae.py`** — Baseline AE variant for comparison experiments.
- **`finetune/finetune.py`** — Functions for fine-tuning both AE (`finetune_autoencoder()`) and GCN (`finetune_gcn()`) on new data. Auto-selects device (MPS → CUDA → CPU) via `get_device()`.
- **`utils/pkg_ckpt.py`** — Resolves packaged pretrained checkpoint paths via `importlib.resources`.
- **`utils/ae_data_loader.py`** — Data loading with automatic CSV/Parquet format detection.
- **`utils/baseline_ae_data_loader.py`** — Data loader for the baseline AE variant.
- **`utils/embed_ae_utils.py`** — AE embedding extraction; defines `LABEL_CANDIDATES` for automatic celltype column detection.
- **`utils/embed_gcn_utils.py`** — GCN inference with optional memory-efficient subgraph batching.
- **`utils/gcn_utils.py`** — Spatial k-NN graph construction and subgraph generation.

### Pretrained Checkpoints

Packaged under `src/spatialfusion/data/`:
- `checkpoint_dir_ae/spatialfusion-multimodal-ae.pt` — Pretrained paired AE
- `checkpoint_dir_gcn/spatialfusion-full-gcn.pt` — Pretrained GCN (ST + H&E)
- `checkpoint_dir_gcn/spatialfusion-he-gcn.pt` — Pretrained GCN (H&E only)

### Data Flow Patterns

- **Dual-mode data loading:** In-memory via `AEInputs(adata, z_uni, z_scgpt)` or disk-based via `sample_list` + `base_path`.
- **Disk-based directory convention:** Each sample must live at `{base_path}/{sample}/` with `embeddings/UNI.csv` (or `.parquet`), `embeddings/scGPT.csv` (or `.parquet`), and `adata.h5ad`.
- **Embedding combination modes:** "average", "concat", "z1", or "z2" for combining encoder outputs.
- **Cell indexing:** Cells use string indices; multi-sample data uses composite `f"{sample}_{cell_id}"` format.
- **`safe_standardize()`** is used pervasively: casts float16→float32, handles low-variance columns (< 1e-5 std), prevents NaN/inf.
- **GCN loading uses fixed hyperparameters:** `load_gcn()` always instantiates `GCNAutoencoder(hidden_dim=10, num_layers=2)` regardless of arguments — must match the pretrained checkpoint architecture.
- **`spatial_key` default mismatch:** `run_full_embedding()` defaults to `"spatial_px"`, while `graphs_from_embeddings_and_adata()` defaults to `"spatial"`. Always pass `spatial_key` explicitly to avoid silent errors.

## Dependencies

GPU workflows require PyTorch+CUDA and DGL (Deep Graph Library) installed separately from pip — see README.md for platform-specific installation. Optional deps: `timm` (UNI model), `torchtext`/`torchdata` (scGPT).

## Environment Setup (uv)

Use `setup_env.sh` to create the environment. It auto-detects CUDA and installs the correct wheels. **Always activate `.venv` before running tests or scripts.**

```bash
bash setup_env.sh
source .venv/bin/activate
pytest tests/ --ignore=tests/test_finetune.py   # test_finetune.py imports a removed legacy module
```

### Known DGL / Platform Pitfalls

**Check the OS first.** The original upstream repo (`uhlerlab/spatialfusion`) was developed on Linux. DGL wheel availability differs critically by platform:

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| DGL 2.2.1+cu121 不可用（Linux） | 该版本只有 Windows wheel，Linux 最高是 2.1.0 | 用 DGL 2.1.0+cu121 |
| `libgraphbolt_pytorch_X.Y.Z.so` not found | DGL 2.1.0 graphbolt 只编译了 torch 2.1.x / 2.2.x | 降级到 torch 2.2.2；脚本自动 symlink patch 版本差异 |
| `No module named 'torchdata.datapipes'` | torchdata ≥ 0.8.0 删除了 datapipes，DGL graphbolt 依赖它 | 锁定 `torchdata==0.7.1` |
| `numpy.core.multiarray failed to import` | torch 2.2.x 使用 NumPy 1.x ABI，NumPy 2.x 不兼容 | 锁定 `numpy<2` |
| `uv pip install` 装到了系统 Python | uv 默认解析系统 Python，忽略当前 venv | 始终传 `--python .venv/bin/python` |

### Validated version combination (Linux, CUDA 12.3)

```
torch==2.2.2+cu121   torchvision==0.17.2+cu121
dgl==2.1.0+cu121     torchdata==0.7.1
numpy==1.26.4        (numpy<2)
```

### Pretrained checkpoint input dimensions

Inferred from `spatialfusion-multimodal-ae.pt`:
- `d1_dim = 1536` (UNI morphology embeddings)
- `d2_dim = 512`  (scGPT molecular embeddings)

## Environment

Default output directory: `$HOME/spatialfusion_runs` (override with `SPATIALFUSION_ROOT` env var).
