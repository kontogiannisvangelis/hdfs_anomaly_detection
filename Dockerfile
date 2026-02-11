FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY pyproject.toml README.md /app/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

COPY . /app
