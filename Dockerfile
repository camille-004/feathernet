FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    python3.11-dev \
    make \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt caches.

WORKDIR /app

COPY . .

RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip \
    && pip install -r requirements.txt
