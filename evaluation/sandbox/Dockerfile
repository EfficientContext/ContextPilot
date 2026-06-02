# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Install system dependencies that might be needed for common code execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -s /bin/bash evaluser

# Set working directory
WORKDIR /app

# Install bigcodebench
RUN pip install --no-cache-dir bigcodebench

# Create results directory and set ownership
RUN mkdir -p /app/results && chown -R evaluser:evaluser /app

# Switch to the non-root user
USER evaluser

# Set entrypoint to bash for interactive testing/persistence
ENTRYPOINT ["/bin/bash"]
