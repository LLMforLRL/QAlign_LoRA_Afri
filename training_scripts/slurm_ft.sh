#!/bin/bash
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=finetuning-%j.out

#############################################################
# install the environment by loading in python and required packages
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2 arrow/17.0.0 

source ~/scratch/QAlign/ENV/bin/activate
#############################################################

# stage 1: question alignment
# finetuning on question translation data
bash training_scripts/finetune_modified.sh gemma-2-9b gsm8kafri_all

# stage 2: response alignment
# finetuning stage 1 model with MetaMathQA dataset
bash training_scripts/finetune_modified.sh gemma-2-9b.gsm8kafri_all.finetune metamath_all