#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --job-name=1st_q3
#SBATCH -M XXXXX
#SBATCH --partition=XXXXX

srun python3 extract_1st_q3.py
