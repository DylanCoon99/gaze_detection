FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (slim — runner only)
COPY requirements-runner.txt .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-runner.txt

# Copy runner and metrics code
COPY src/infra/runner/ /app/runner/
COPY src/infra/metrics/ /app/metrics/

# Set Python path so runner can find metrics
ENV PYTHONPATH="/app/runner:/app/metrics"
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"
ENV MLFLOW_ALLOW_FILE_STORE="true"
ENV GIT_PYTHON_REFRESH="quiet"

ENTRYPOINT ["python", "/app/runner/main.py"]
