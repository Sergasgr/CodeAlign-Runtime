FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    curl git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

WORKDIR /app

COPY pyproject.toml .

RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv -e .

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

CMD ["bash"]
