#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=eval_only_%j.out
#SBATCH --error=eval_only_%j.err

echo "Running Sandbox evaluation..."
cd evaluation/benchmarks
bash run_sandbox_eval_full.sh

echo "Re-evaluation finished!"
