#!/bin/bash
#OAR -l nodes=2,walltime=00:20:00
#OAR -n PlanktonZilla_Training
#OAR -p gpu_count >= 2 AND gpu_model = 'A40'
#OAR -q besteffort
#OAR -O logs/job.%jobid%.out
#OAR -E logs/job.%jobid%.err

# 1. FIX: Set the correct path to where your code AND .venv are
PROJECT_DIR="/home/svasquez/planktonzilla"
cd "$PROJECT_DIR" || exit 1

# Export Environment Variables
export OAR_JOB_ID=${OAR_JOB_ID:-local}
export HF_HUB_OFFLINE=0
export WANDB_MODE=online

# === Multi-Node Configuration ===
sort -u "$OAR_NODEFILE" > unique_nodes.txt
MASTER_ADDR=$(head -n 1 unique_nodes.txt)
MASTER_PORT=29500
NNODES=$(wc -l < unique_nodes.txt)

echo "==================================================="
echo "Job ID: $OAR_JOB_ID"
echo "Master Node: $MASTER_ADDR"
echo "Total Nodes: $NNODES"
echo "Project Dir: $PROJECT_DIR"
echo "==================================================="

# === Construct the Command ===
# Use the full path to activate the venv to be safe
CMD="cd $PROJECT_DIR || exit 1; \
     source .venv/bin/activate; \
     export WANDB_MODE=$WANDB_MODE; \
     export HF_HUB_OFFLINE=$HF_HUB_OFFLINE; \
     torchrun \
       --nproc_per_node=2 \
       --nnodes=$NNODES \
       --rdzv_id=$OAR_JOB_ID \
       --rdzv_backend=c10d \
       --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
       planktonzilla/train_ood.py $@"

# === Execution ===
echo "Starting distributed training..."
export PARALLEL_SSH=oarsh

# Ensure the logs directory exists before starting
mkdir -p logs

parallel --tag --nonall --sshloginfile unique_nodes.txt "$CMD"