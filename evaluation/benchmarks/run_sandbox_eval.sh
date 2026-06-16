#!/bin/bash

# Define paths (Git Bash / WSL compatible Windows paths or native Windows paths work with cp)
SOURCE="D:/AI4Coding/ContextPilot/evaluation/benchmarks/elm_samples.jsonl"
DEST_DIR="D:/AI4Coding/ContextPilot/evaluation/sandbox/results"
DEST="$DEST_DIR/elm_samples.jsonl"

echo "Preparing Sandbox Environment..."

# Ensure the results directory exists
mkdir -p "$DEST_DIR"

# Copy the samples to the volume-mounted folder
cp "$SOURCE" "$DEST"
echo "Successfully copied elm_samples.jsonl to the Docker volume mount ($DEST_DIR)."

echo -e "\nChecking if bigcodebench-sandbox is running..."
if [ "$(docker inspect -f '{{.State.Running}}' bigcodebench-sandbox 2>/dev/null)" != "true" ]; then
    echo "Container is not running. Starting it now..."
    docker start bigcodebench-sandbox
else
    echo "Container is already running."
fi

echo -e "\nExecuting BigCodeBench Evaluation securely inside the Docker Sandbox..."
# Run the evaluation command inside the existing running container
docker exec bigcodebench-sandbox bigcodebench.evaluate --samples /app/results/elm_samples.jsonl

echo -e "\nEvaluation Complete!"
