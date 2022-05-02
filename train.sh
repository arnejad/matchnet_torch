#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1



module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4

module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4

source /data/p306627/.envs/torch/bin/activate

cp /data/p306627/DBs/matchnet.tar.xz /local/tmp/


python3 /home/p306627/codes/matchnet_torch/train.py 


rm -r /local/tmp/matchnet
rm /local/tmp/matchnet.tar.xz