#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name=GEVP_dm
#SBATCH -M XXXXX
#SBATCH --partition=XXXXX
#SBATCH --array=0-100

srun python3 extract_GEVP_q3q5_delta_mod.py
