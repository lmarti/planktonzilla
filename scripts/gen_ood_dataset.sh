#!/bin/bash
#SBATCH --account=tec@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/planktonzilla/
module purge
module load arch/h100
source .venv/bin/activate
cd notebooks

export SLURM_JOB_ID=${SLURM_JOB_ID:-local}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

srun python gen_planktonzilla_ood.py