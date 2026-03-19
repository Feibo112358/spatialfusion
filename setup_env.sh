#!/usr/bin/env bash
# setup_env.sh — Create and populate a uv virtualenv for SpatialFusion
# Auto-detects CUDA version and selects the matching PyTorch + DGL wheels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── 1. Detect CUDA version ────────────────────────────────────────────────────
detect_cuda_tag() {
    local cuda_ver
    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    elif command -v nvidia-smi &>/dev/null; then
        cuda_ver=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    else
        echo "cpu"; return
    fi

    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    # Map installed CUDA → highest compatible PyTorch cu-tag
    # (cu121 works on CUDA 12.1–12.3; cu124 requires CUDA ≥ 12.4)
    if   [[ $major -eq 12 && $minor -ge 4 ]]; then echo "cu124"
    elif [[ $major -eq 12 ]];                  then echo "cu121"
    elif [[ $major -eq 11 && $minor -ge 8 ]];  then echo "cu118"
    else                                            echo "cpu"
    fi
}

CUDA_TAG=$(detect_cuda_tag)
echo "Detected CUDA tag: ${CUDA_TAG}"

# ── 2. Version pins ───────────────────────────────────────────────────────────
# DGL 2.1.0 is the latest Linux build available for cu121 (2.2.1 is Windows-only).
# DGL 2.1.0 ships libgraphbolt compiled for torch 2.1.x / 2.2.x — torch 2.2.2
# is the newest patch release with a compatible graphbolt .so.
# torchdata 0.7.1 is required: newer versions (0.8+) removed datapipes used by DGL.
# numpy <2 is required: torch 2.2.2 was built against NumPy 1.x ABI.
TORCH_VERSION="2.2.2"
TORCHVISION_VERSION="0.17.2"
TORCHDATA_VERSION="0.7.1"

case "$CUDA_TAG" in
    cu121) TORCH_INDEX="https://download.pytorch.org/whl/cu121"
           DGL_INDEX="https://data.dgl.ai/wheels/cu121/repo.html" ;;
    cu118) TORCH_INDEX="https://download.pytorch.org/whl/cu118"
           DGL_INDEX="https://data.dgl.ai/wheels/cu118/repo.html" ;;
    *)     TORCH_INDEX="https://download.pytorch.org/whl/cpu"
           DGL_INDEX="https://data.dgl.ai/wheels/repo.html" ;;
esac

# ── 3. Create / reuse venv (Python 3.10) ─────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating uv virtualenv at ${VENV_DIR} (Python 3.10)..."
    uv venv "$VENV_DIR" --python 3.10
else
    echo "Reusing existing virtualenv at ${VENV_DIR}"
fi

VENV_PY="$VENV_DIR/bin/python"

# ── 4. Remove any conflicting torch / nvidia packages in the venv ─────────────
echo ""
echo "Removing any existing torch/nvidia packages from the venv..."
EXISTING=$("$VENV_PY" -m pip list --format=freeze 2>/dev/null \
    | grep -E "^(torch|torchvision|nvidia-)" | cut -d= -f1 || true)
if [[ -n "$EXISTING" ]]; then
    # shellcheck disable=SC2086
    "$VENV_PY" -m pip uninstall -y $EXISTING
else
    echo "  (nothing to remove)"
fi

# ── 5. Install PyTorch ────────────────────────────────────────────────────────
echo ""
echo "Installing PyTorch ${TORCH_VERSION}+${CUDA_TAG} ..."
uv pip install \
    --python "$VENV_PY" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "$TORCH_INDEX"

# ── 6. Install DGL ────────────────────────────────────────────────────────────
echo ""
echo "Installing DGL (${CUDA_TAG}) ..."
uv pip install \
    --python "$VENV_PY" \
    dgl \
    --find-links "$DGL_INDEX"

# ── 6a. Patch: torchdata downgrade (DGL 2.1 needs datapipes, removed in 0.8+)
echo ""
echo "Pinning torchdata to ${TORCHDATA_VERSION} (DGL 2.1 requires datapipes) ..."
uv pip install --python "$VENV_PY" "torchdata==${TORCHDATA_VERSION}"

# ── 6b. Patch: numpy <2 (torch 2.2 ABI requires numpy 1.x)
echo "Pinning numpy<2 (torch 2.2 ABI requirement) ..."
uv pip install --python "$VENV_PY" "numpy<2"

# ── 6c. Patch: symlink graphbolt .so for exact torch patch version
# DGL 2.1.0 ships .so files for 2.2.0 and 2.2.1; create a symlink for 2.2.2.
GRAPHBOLT_DIR="$VENV_DIR/lib/python3.10/site-packages/dgl/graphbolt"
TORCH_VER_FULL=$("$VENV_PY" -c "import torch; print(torch.__version__.split('+')[0])")
SO_TARGET="$GRAPHBOLT_DIR/libgraphbolt_pytorch_${TORCH_VER_FULL}.so"
if [[ ! -f "$SO_TARGET" ]]; then
    # Find the closest existing .so with the same minor version
    MINOR_PREFIX="libgraphbolt_pytorch_$(echo "$TORCH_VER_FULL" | cut -d. -f1,2)"
    BEST_SO=$(ls "$GRAPHBOLT_DIR/${MINOR_PREFIX}"*.so 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "$BEST_SO" ]]; then
        ln -sf "$BEST_SO" "$SO_TARGET"
        echo "Symlinked $(basename "$BEST_SO") → $(basename "$SO_TARGET")"
    else
        echo "WARNING: No matching graphbolt .so found for torch ${TORCH_VER_FULL}" >&2
    fi
fi

# ── 7. Install SpatialFusion + dev deps ───────────────────────────────────────
echo ""
echo "Installing spatialfusion[dev] in editable mode..."
uv pip install \
    --python "$VENV_PY" \
    -e "$SCRIPT_DIR/.[dev]"

# ── 8. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "Running smoke tests..."
"$VENV_PY" - <<'EOF'
import torch, dgl, spatialfusion
print(f"  torch       : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device : {torch.cuda.get_device_name(0)}")
print(f"  dgl         : {dgl.__version__}")
print(f"  spatialfusion loaded OK")
EOF

echo ""
echo "All done. Activate with:"
echo "  source .venv/bin/activate"
