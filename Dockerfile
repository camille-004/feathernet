FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    make \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt caches.

RUN pip install poetry

WORKDIR /app

COPY . .

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi
