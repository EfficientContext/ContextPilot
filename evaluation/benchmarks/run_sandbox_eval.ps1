$source = "D:\AI4Coding\ContextPilot\evaluation\benchmarks\elm_samples.jsonl"
$destDir = "D:\AI4Coding\ContextPilot\evaluation\sandbox\results"
$destination = Join-Path $destDir "elm_samples.jsonl"

Write-Host "Preparing Sandbox Environment..."

# Ensure the results directory exists
if (-not (Test-Path -Path $destDir)) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
}

# Copy the samples to the volume-mounted folder
Copy-Item -Path $source -Destination $destination -Force
Write-Host "Successfully copied elm_samples.jsonl to the Docker volume mount ($destDir)."

Write-Host "`nChecking if bigcodebench-sandbox is running..."
$containerStatus = docker inspect -f '{{.State.Running}}' bigcodebench-sandbox 2>$null
if ($containerStatus -ne "true") {
    Write-Host "Container is not running. Starting it now..."
    docker start bigcodebench-sandbox
} else {
    Write-Host "Container is already running."
}

Write-Host "`nExecuting BigCodeBench Evaluation securely inside the Docker Sandbox..."
# Run the evaluation command inside the existing running container
docker exec bigcodebench-sandbox bigcodebench.evaluate --samples /app/results/elm_samples.jsonl

Write-Host "`nEvaluation Complete!"
