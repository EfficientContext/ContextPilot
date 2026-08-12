#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=debug_import_%j.out
#SBATCH --error=debug_import_%j.err

echo "Debugging http_server import on compute node..."
export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

python -v -c "import contextpilot.server.http_server" 2>&1 | tail -n 100
echo "Done."
