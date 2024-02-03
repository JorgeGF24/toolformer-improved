#!/bin/bash
#SBATCH -o ./zslurm/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuC
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=24mokies2@gmail.com # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
echo $(date)
SECONDS=0
source activate
python "preprocess_data copy.py"
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."