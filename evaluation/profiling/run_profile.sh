#!/bin/bash

# Ensure dependencies are installed
echo "Installing snakeviz..."
pip install snakeviz --quiet

# Run the script with cProfile
# -o agent_profile.prof: outputs the binary profile data to a file
echo "Running dummy_agent.py with cProfile..."
python -m cProfile -o agent_profile.prof dummy_agent.py

# Visualize the results
# This will open a browser tab with a flame graph/icicle graph
echo "Launching snakeviz for visualization..."
snakeviz agent_profile.prof
