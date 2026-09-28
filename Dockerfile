FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv \
    curl ca-certificates git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

ENV PATH="/opt/venv/bin:$PATH" \
    CUDA_HOME=/usr/local/cuda

COPY . .

CMD ["bash"]
