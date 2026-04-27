#!/bin/bash
#SBATCH --partition=compil
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/am/planktonzilla/
module purge
source .venv/bin/activate
cd notebooks


# === Ejecutar torchrun ===
srun python add_planktonzilla.py