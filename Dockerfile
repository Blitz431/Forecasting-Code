FROM python:3.11-slim

WORKDIR /app

# System deps needed to build/import pyarrow, lightgbm, xgboost, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch's CUDA build first (host has an RTX 3060 Ti passed through via
# Docker Desktop's WSL2 GPU support). cu124 wheels work fine on newer drivers —
# NVIDIA's driver/runtime compatibility is backwards-compatible.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Dependency layer cached separately from source so editing .py files doesn't
# force a reinstall of these (mostly large, slow-to-build) packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Actual process is chosen per-service via docker-compose's `command:`.
CMD ["python", "cli/scheduler.py", "--status"]
