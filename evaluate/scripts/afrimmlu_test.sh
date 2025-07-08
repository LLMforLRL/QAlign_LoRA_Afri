#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/eval_afrimmlu-%j.out

#############################################################
# install the environment by loading in python and required packages
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2 arrow/17.0.0 

source ~/scratch/QAlign/ENV/bin/activate
#############################################################
MODEL1="gemma-2-9b-it.gsm8kafri_all.finetune.metamath_all.finetune"
MODEL2="gemma-2-9b-it"

PROJECT_PATH="$(pwd)"
SCRATCH_PATH=~/scratch/QAlign
MODEL_PATH1=$SCRATCH_PATH/model/$MODEL1
MODEL_PATH2=$SCRATCH_PATH/model/$MODEL2

# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/evaluate/scripts/afrimmlu_test.py \
    --model_path $MODEL_PATH1 \
    --batch_size 8 \

# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/evaluate/scripts/afrimmlu_test.py \
    --model_path $MODEL_PATH2 \
    --batch_size 8 \
