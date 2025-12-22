# HuggingFace Spaces - Docker SDK
# Uses Astral's uv to manage Python and project dependencies

# Lightweight Debian base with uv preinstalled
FROM ghcr.io/astral-sh/uv:debian

SHELL ["/bin/bash", "-lc"]

# Environment for predictable behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

# Workdir inside container
WORKDIR /app

# System dependencies
# - ffmpeg required by video/audio processing
# - OpenCV runtime libraries to support opencv-python wheels
# Note: packages.txt includes several llvm-10 entries which are typically
# not required for CPU wheels and may not be available on Debian.
COPY packages.txt ./
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only project metadata first to leverage Docker layer caching for deps
COPY pyproject.toml README.md ./

# Sync project dependencies with uv
# Will install Python 3.10.13 as specified in pyproject, create a venv, and
# install all runtime deps (excluding dev deps).
RUN uv sync --no-dev

# Copy the rest of the application source
COPY . .

# Download required checkpoints from HuggingFace
RUN mkdir -p checkpoints/whisper && \
    curl -L --fail -o checkpoints/latentsync_unet.pt "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_unet.pt" && \
    curl -L --fail -o checkpoints/whisper/tiny.pt "https://huggingface.co/ByteDance/LatentSync/resolve/main/whisper/tiny.pt"

# Expose Gradio default port
EXPOSE 7860

# Start the app. Gradio CLI respects GRADIO_SERVER_NAME and PORT.
# Use uv to ensure the correct virtual environment and Python version.
CMD ["uv", "run", "gradio", "app.py"]