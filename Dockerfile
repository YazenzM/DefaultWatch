# Scoring API for DefaultWatch.
# Build after `defaultwatch train` has written models/defaultwatch_bundle.joblib:
#   docker build -t defaultwatch-api .
#   docker run --rm -p 8000:8000 defaultwatch-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/defaultwatch_bundle.joblib

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY models ./models

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json,sys,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); sys.exit(0 if r.get('status')=='ok' else 1)"

CMD ["uvicorn", "defaultwatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
