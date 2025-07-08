#!/bin/bash
MODEL=$1

PROJECT_PATH="$(pwd)"
SCRATCH_PATH=~/scratch/QAlign
MODEL_PATH=$SCRATCH_PATH/model/$MODEL

# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/evaluate/scripts/afrixnli_test.py \
    --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 16 \
    # --lang_only Bengali Thai Swahili Japanese Chinese German French Russian Spanish English