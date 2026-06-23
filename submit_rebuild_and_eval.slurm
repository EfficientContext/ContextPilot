#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=rebuild_and_eval_%j.out
#SBATCH --error=rebuild_and_eval_%j.err

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "Rebuilding Apptainer Image..."
cd evaluation/sandbox
apptainer build --force --fakeroot my_project.sif recipe.def

echo "Running Sandbox evaluation..."
cd ../benchmarks
bash run_sandbox_eval_full.sh

echo "Re-evaluation finished!"
