FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_CACHE_DIR=/opt/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the embedding model into the image so the API starts without a network fetch
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"

EXPOSE 8000
CMD ["uvicorn", "recon_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
