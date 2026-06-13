# Use the official PyTorch image with CUDA support pre-installed
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables to prevent interactive prompts and buffer delays
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for ML libraries (like LightGBM and FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install all Python dependencies
# Using --no-cache-dir to prevent bloating the container image size
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase into the container
COPY . .

# Set the default command to open an interactive bash shell
CMD ["/bin/bash"]
