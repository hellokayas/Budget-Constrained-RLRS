#!/bin/bash
#SBATCH --account=p32726
#SBATCH --partition=gengpu
#SBATCH --time=04:00:00
#SBATCH --job-name=ucontrol
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
##SBATCH --constraint=rhel8
#SBATCH --cpus-per-task=4
#SBATCH --output=ucontrol.log

module purge
module load mamba/24.3.0
eval "$('/hpc/software/mamba/24.3.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
source "/hpc/software/mamba/24.3.0/etc/profile.d/mamba.sh"
mamba activate /projects/p32726/pythonenvs/llm3

export LOGLEVEL=INFO
python3 hashcontrol.py