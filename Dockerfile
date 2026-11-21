# SQLForge API + UI image (CPU only; the model itself is served by vLLM on the GPU host).
#   docker build -t sqlforge-app .
# Used by observability/docker-compose.yml (services `api` and `ui`); databases are mounted from ./data.
FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv/sqlforge

COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

# only the code the service needs (training / serving tooling stays out of the image)
COPY app/ app/
COPY data/__init__.py data/prompt.py data/schema.py data/schema_linker.py data/prepare.py data/
COPY eval/__init__.py eval/metrics.py eval/generate.py eval/self_correct.py eval/
RUN useradd --create-home --uid 10001 sqlforge && mkdir -p data/raw data/processed && chown -R sqlforge /srv/sqlforge
USER sqlforge

EXPOSE 8080 8501
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
