# Build context is the repo root: docker build -f src/serving/Dockerfile .
# Model artifact must be downloaded first: python src/serving/download_model.py
FROM python:3.12-slim

WORKDIR /app

# psutil==5.9.0 (a model dependency) has no prebuilt wheel for this base
# image and needs to compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the model's own captured dependencies first (exact versions from
# training -- a version drift here silently corrupted predictions once
# already, see src/serving/model/requirements.txt), then the serving layer.
COPY src/serving/model/requirements.txt ./model-requirements.txt
RUN pip install --no-cache-dir -r model-requirements.txt fastapi "uvicorn[standard]"

COPY src/serving/app.py ./
COPY src/serving/model ./model

# SageMaker runs `docker run <image> serve` -- that overrides any CMD (no
# ENTRYPOINT is set), so "serve" must exist as an executable on PATH.
COPY src/serving/serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# SageMaker's container contract expects the server on port 8080.
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
