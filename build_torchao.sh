#!/bin/bash
# Script to build torchao from source with CUDA extensions for torchtitan
# This is needed for mxfp8_quantize_cuda and other CUDA kernels
#
# Usage:
#   ./build_torchao.sh                    # Auto-detect GPU architecture
#   ./build_torchao.sh 8.0                # Build for SM 8.0 (A100)
#   CUDA_ARCH=8.0 ./build_torchao.sh      # Alternative syntax

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building torchao from source with CUDA extensions${NC}"

# Allow user to specify architecture
if [ -n "$1" ]; then
    USER_CUDA_ARCH="$1"
elif [ -n "$CUDA_ARCH" ]; then
    USER_CUDA_ARCH="$CUDA_ARCH"
else
    USER_CUDA_ARCH=""
fi

# Check if ao repo exists
AO_PATH="${AO_PATH:-$HOME/git/ao}"
if [ ! -d "$AO_PATH" ]; then
    echo -e "${RED}Error: torchao repository not found at $AO_PATH${NC}"
    echo "Please clone it with: git clone https://github.com/pytorch/ao $AO_PATH"
    exit 1
fi

# Activate venv
VENV_PATH="$(dirname "$0")/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    exit 1
fi

source "$VENV_PATH/bin/activate"

# Check PyTorch and CUDA
echo -e "${YELLOW}Checking environment...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else \"N/A\"}')"

# Patch PyTorch's CUDA version check if needed
CPP_EXTENSION_FILE="$VENV_PATH/lib/python3.11/site-packages/torch/utils/cpp_extension.py"
if [ ! -f "$CPP_EXTENSION_FILE" ]; then
    echo -e "${RED}Error: cpp_extension.py not found at expected location${NC}"
    exit 1
fi

# Check if already patched
if grep -q "Temporarily commented out for torchao build" "$CPP_EXTENSION_FILE"; then
    echo -e "${GREEN}PyTorch CUDA version check already patched${NC}"
else
    echo -e "${YELLOW}Patching PyTorch CUDA version check to allow CUDA 13.0...${NC}"
    # Create backup
    if [ ! -f "$CPP_EXTENSION_FILE.backup.original" ]; then
        cp "$CPP_EXTENSION_FILE" "$CPP_EXTENSION_FILE.backup.original"
    fi

    # Patch using Python for reliability
    python -c "
import sys

cpp_file = '$CPP_EXTENSION_FILE'

with open(cpp_file, 'r') as f:
    content = f.read()

# Find and replace the version check
old_code = '''        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)'''

new_code = '''        # Temporarily commented out for torchao build with CUDA 13.0
        # if cuda_ver.major != torch_cuda_version.major:
        #     raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(cpp_file, 'w') as f:
        f.write(content)
    print('Patched successfully')
elif new_code in content:
    print('Already patched')
else:
    print('ERROR: Could not find code to patch')
    sys.exit(1)
"
    echo -e "${GREEN}Patch applied (original saved to $CPP_EXTENSION_FILE.backup.original)${NC}"
fi

# Install wheel if not present
if ! python -c "import wheel" 2>/dev/null; then
    echo -e "${YELLOW}Installing wheel package...${NC}"
    pip install wheel
fi

# Detect or use specified GPU architecture
if [ -n "$USER_CUDA_ARCH" ]; then
    CUDA_ARCH="$USER_CUDA_ARCH"
    echo -e "${GREEN}Using user-specified CUDA architecture: $CUDA_ARCH${NC}"
else
    echo -e "${YELLOW}Detecting GPU architecture...${NC}"
    CUDA_ARCH=$(python -c "import torch; cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None; print(f'{cap[0]}.{cap[1]}' if cap else '')" 2>/dev/null || echo "")

    if [ -z "$CUDA_ARCH" ]; then
        echo -e "${YELLOW}Warning: Could not detect GPU architecture. Using default (9.0;10.0)${NC}"
        echo -e "${YELLOW}If this is wrong, run: ./build_torchao.sh <arch>  (e.g., ./build_torchao.sh 8.0)${NC}"
        CUDA_ARCH="9.0;10.0"
    else
        echo -e "${GREEN}Detected CUDA compute capability: $CUDA_ARCH${NC}"
        # Extract major version for architecture list
        MAJOR_VER=$(echo $CUDA_ARCH | cut -d. -f1)
        CUDA_ARCH="${MAJOR_VER}.0"
    fi
fi

# Build and install torchao
echo -e "${YELLOW}Building torchao from source (this may take several minutes)...${NC}"
cd "$AO_PATH"
# Set TORCH_CUDA_ARCH_LIST to avoid auto-detection issues when CUDA runtime isn't fully initialized
TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" USE_CPP=1 pip install --no-build-isolation -e .

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "import torchao; from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda; print('torchao version:', torchao.__version__); print('✓ mxfp8_quantize_cuda imported successfully')"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}torchao built successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "torchao is installed in editable mode from: $AO_PATH"
echo "Any changes to the source will be reflected immediately."
echo ""
echo "Note: If you reinstall PyTorch, you may need to run this script again"
echo "to rebuild torchao's CUDA extensions."
