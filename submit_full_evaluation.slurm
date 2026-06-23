#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=evaluation_full_%j.out
#SBATCH --error=evaluation_full_%j.err

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "Starting ELM dataset generation..."
python3 evaluation/benchmarks/run_bigcodebench_elm_full.py

echo "Starting Sandbox evaluation..."
bash evaluation/benchmarks/run_sandbox_eval_full.sh

echo "Full evaluation pipeline finished!"
