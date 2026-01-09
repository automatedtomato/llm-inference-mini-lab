# CUDA 12.4 Dev image include nvcc
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    bash-completion \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /usr/local/bin/uv

WORKDIR /llm-inference-mini-lab

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system -r pyproject.toml --extra dev