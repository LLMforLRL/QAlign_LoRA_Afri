#!/bin/bash

# Load necessary modules
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2 arrow/17.0.0 

# Create and activate virtual environment
cd ~/scratch/QAlign
virtualenv --no-download ENV
source ~/scratch/QAlign/ENV/bin/activate

# Upgrade pip 
pip install --no-index --upgrade pip

# Install torch and deepspeed 
pip install --no-index torch deepspeed

# Install specific and required packages
pip install packaging==25.0
pip install -U transformers accelerate
pip install -r fixed_requirements.txt
