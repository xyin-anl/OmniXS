# Use Python base image
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# # Set environment variables for Lightning installation in Mac
# ENV GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
# ENV GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Copy and run installation script
COPY install.sh /workspace/
RUN chmod +x /workspace/install.sh && \
    ./install.sh && \
    rm install.sh

# Create a non-root user
RUN useradd -m -s /bin/bash user
USER user

# Set Python path
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

# Default command
CMD ["bash"]
