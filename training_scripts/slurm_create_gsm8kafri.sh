#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=create_gsm8kafri-%j.out

#############################################################
# install the environment by loading in python and required packages
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2 arrow/17.0.0 

source ~/scratch/QAlign/ENV/bin/activate
#############################################################

python kosei_ds_script/datas/nllb_translation_mgsm.py