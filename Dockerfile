# Reproducibility image for the CRC metagenomics meta-analysis.
#
# Builds a self-contained environment that can run scripts/verify_results.py
# against the committed data/processed/ and results/ CSVs.
#
# Build:
#   docker build -t crc-metagenomics .
# Run (default = verify headline numbers):
#   docker run --rm crc-metagenomics
# Drop into a shell for an interactive re-run of any pipeline step:
#   docker run --rm -it crc-metagenomics bash

FROM python:3.11-slim

# Avoid interactive tzdata prompts and keep image small.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Minimal system deps for numpy/scipy/xgboost wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches independently of source code.
COPY requirements.lock ./
RUN pip install -r requirements.lock

# Copy the rest of the repo (respecting .dockerignore).
COPY . .

CMD ["python3", "scripts/verify_results.py"]
