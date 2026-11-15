# =============================================================================
# CPU Build Stage
# =============================================================================
FROM ubuntu:24.04 AS builder-cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Fetch ggml v0.9.11 if submodule is not populated, and apply OmniVoice patch
RUN if [ ! -f vendor/ggml/CMakeLists.txt ]; then \
        rm -rf vendor/ggml && \
        git clone --branch v0.9.11 --depth 1 https://github.com/ggml-org/ggml.git vendor/ggml; \
    fi && \
    sed -i 's/\r$//' patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch && \
    git -C vendor/ggml apply --check /src/patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch && \
    git -C vendor/ggml apply /src/patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build -j"$(nproc)"

# =============================================================================
# CUDA Build Stage (use `--target runtime-cuda` for NVIDIA GPUs)
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder-cuda

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN if [ ! -f vendor/ggml/CMakeLists.txt ]; then \
        rm -rf vendor/ggml && \
        git clone --branch v0.9.11 --depth 1 https://github.com/ggml-org/ggml.git vendor/ggml; \
    fi && \
    sed -i 's/\r$//' patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch && \
    git -C vendor/ggml apply --check /src/patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch && \
    git -C vendor/ggml apply /src/patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build -j"$(nproc)"

# =============================================================================
# CUDA Runtime Stage
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-cuda

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-cuda /src/build/omnivoice-server /usr/local/bin/omnivoice-server
COPY --from=builder-cuda /src/build/omnivoice-cli /usr/local/bin/omnivoice-cli

RUN mkdir -p /models /voices /output
WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["omnivoice-server"]
CMD ["--model", "/models/omnivoice-q8_0.gguf", "--host", "0.0.0.0", "--port", "8000", "--voices-dir", "/voices", "--device", "cuda:0"]

# =============================================================================
# CPU Runtime Stage (Default)
# =============================================================================
FROM ubuntu:24.04 AS runtime-cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-cpu /src/build/omnivoice-server /usr/local/bin/omnivoice-server
COPY --from=builder-cpu /src/build/omnivoice-cli /usr/local/bin/omnivoice-cli

RUN mkdir -p /models /voices /output
WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["omnivoice-server"]
CMD ["--model", "/models/omnivoice-q8_0.gguf", "--host", "0.0.0.0", "--port", "8000", "--voices-dir", "/voices", "--device", "cpu", "--threads", "4"]
