#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name=autocorr
#SBATCH -M qp3
#SBATCH --partition=qp3
#SBATCH --array=0-100

srun python3 extract_autocorrelations.py

