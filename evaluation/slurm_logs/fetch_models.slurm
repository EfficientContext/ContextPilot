#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00
#SBATCH --output=fetch_models_full_%j.out
#SBATCH --error=fetch_models_full_%j.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

python -c '
import os
import requests

url = os.environ.get("BASE_URL", "https://api.openai.com/v1") + "/models"
api_key = os.environ.get("OPENAI_API_KEY", "dummy")

try:
    response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
    data = response.json()
    for item in data.get("data", []):
        print(item.get("id"))
except Exception as e:
    print("Error:", e)
'
