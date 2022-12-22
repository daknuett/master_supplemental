#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --job-name=3rd_q3
#SBATCH -M XXXXX
#SBATCH --partition=XXXXX

srun python3 extract_3rd_q3.py
