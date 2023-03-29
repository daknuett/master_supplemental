#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=extract
#SBATCH -M hpd
#SBATCH --partition=hpd
#SBATCH --array=0-43%5

srun python3 handle_observable_extratction.py


