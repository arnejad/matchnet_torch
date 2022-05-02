#!/bin/bash

#SBATCH --time=10-00:00:00



module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4

module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4

python3 /home/p306627/codes/matchnet_torch/generate_db.py

