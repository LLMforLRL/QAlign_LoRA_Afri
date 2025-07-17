#!/bin/bash
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/finetuning-gemma_2-afrixnli-%j.out

#############################################################
# install the environment by loading in python and required packages
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2 arrow/17.0.0 

source /home/bumie304/scratch/QAlign/ENV/bin/activate
#############################################################

# stage 1: question alignment
# finetuning model on question translation data
bash training_scripts/finetune_modified.sh gemma-2-9b-it snli-query-translation_all

# stage 2: response alignment
# finetuning stage 1 model with MetaMathQA dataset
bash training_scripts/finetune_modified.sh gemma-2-9b-it.snli-query-translation_all.finetune snli_[:395000]

# EVALUATION MGSM
MODEL="gemma-2-9b-it.snli-query-translation_all.finetune.snli_[:395000].finetune"

PROJECT_PATH="$(pwd)"
SCRATCH_PATH=~/scratch/QAlign
MODEL_PATH=$SCRATCH_PATH/model/$MODEL

# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/evaluate/scripts/afrixnli_test.py \
    --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 32 \