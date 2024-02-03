#!/bin/bash
#SBATCH -o ./zslurm/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
export LD_LIBRARY_PATH=/vol/bitbucket/jg2619/augmenting_llms/dependencies/OpenBlas/lib/:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
echo $(date)
SECONDS=0
source activate
python "preprocess_data copy.py"
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."