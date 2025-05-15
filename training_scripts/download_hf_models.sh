#!/bin/bash

# Usage: ./download_models.sh /path/to/model_dir YOUR_HF_TOKEN

# Check for 2 inputs
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 model_dir hf_token"
  exit 1
fi

MODEL_DIR="$1"
HF_TOKEN="$2"

echo "[INFO] Using model directory: $MODEL_DIR"
echo "[INFO] Using provided Hugging Face token"

# Change to the specified directory
cd "$MODEL_DIR" || { echo "[ERROR] Directory not found: $MODEL_DIR"; exit 1; }

# Declare model names and local folder names
declare -A models=(
  ["CohereLabs/aya-101"]="aya-101"
  ["google/gemma-2-9b"]="gemma-2-9b"
  ["meta-math/MetaMath-Mistral-7B"]="MetaMath-Mistral-7B"
  ["LLaMAX/LLaMAX2-7B-XNLI"]="LLaMAX2-7B-XNLI"
  ["facebook/nllb-200-3.3B"]="nllb-200-3.3B"
)

# Loop through models and download in parallel
for model_id in "${!models[@]}"; do
  local_dir="${models[$model_id]}"
  echo "[INFO] Starting download for $model_id in background..."
  huggingface-cli download "$model_id" \
    --local-dir "./$local_dir" \
    --local-dir-use-symlinks False \
    --token "$HF_TOKEN" &

  # Optional: Limit concurrent jobs (e.g., max 3)
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do
    sleep 1
  done
done

# Wait for all background jobs to finish
wait

echo "[DONE] All downloads completed."
