#!/bin/bash
# Script to download HuggingFace model assets if not present
# This downloads the tokenizer and config files needed for training
#
# Usage:
#   ./download_qwen_assets.sh                    # Downloads Qwen/Qwen3-0.6B (default)
#   ./download_qwen_assets.sh Qwen/Qwen3-1.7B    # Downloads specified model

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default to Qwen3-0.6B if no argument provided
REPO_ID="${1:-Qwen/Qwen3-0.6B}"

# Extract model name from repo_id (part after "/")
if [[ ! "$REPO_ID" =~ / ]]; then
    echo "Error: Invalid repo format. Expected 'org/model' (e.g., 'Qwen/Qwen3-0.6B')"
    exit 1
fi

MODEL_NAME="${REPO_ID##*/}"
ASSETS_PATH="./assets/hf/$MODEL_NAME"

# Check if assets already exist
if [ -d "$ASSETS_PATH" ] && [ "$(ls -A $ASSETS_PATH 2>/dev/null)" ]; then
    echo -e "${GREEN}$MODEL_NAME assets already exist at $ASSETS_PATH${NC}"
    exit 0
fi

echo -e "${YELLOW}$MODEL_NAME assets not found at $ASSETS_PATH${NC}"
echo -e "${YELLOW}Downloading tokenizer and config from HuggingFace ($REPO_ID)...${NC}"

# Download tokenizer and config files
python scripts/download_hf_assets.py \
    --repo_id "$REPO_ID" \
    --assets tokenizer config

echo -e "${GREEN}Successfully downloaded $MODEL_NAME assets to $ASSETS_PATH${NC}"
