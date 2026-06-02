# Ensure dependencies are installed
Write-Host "Installing snakeviz..." -ForegroundColor Cyan
pip install snakeviz --quiet

# Run the script with cProfile
Write-Host "Running dummy_agent.py with cProfile..." -ForegroundColor Cyan
python -m cProfile -o agent_profile.prof dummy_agent.py

# Visualize the results
Write-Host "Launching snakeviz for visualization..." -ForegroundColor Cyan
snakeviz agent_profile.prof
