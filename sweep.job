#!/bin/bash
#SBATCH --partition cpulong
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 3-0:00:00
#SBATCH --job-name sweep
#SBATCH --output slogs/sweep-%J.log
module purge
singularity exec ${OPENSPIEL_IMG} wandb agent ${WANDB_PROJECT}
