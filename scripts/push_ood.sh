#!/bin/bash
#SBATCH --partition=compil
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/planktonzilla/
module purge
source .venv/bin/activate
cd notebooks

# === Ejecutar torchrun ===
srun python push_planktonzilla.py