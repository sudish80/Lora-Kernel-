# Production Dockerfile for LoraKernel serving
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake libssl-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/ ./src/
COPY main.cpp ./

RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) lora_kernel && \
    make test_gradient_checker test_convergence test_mha

FROM ubuntu:22.04 AS runtime
RUN apt-get update && apt-get install -y libssl3 libomp5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/build/lora_kernel .
COPY --from=builder /build/build/test_* ./

EXPOSE 8080
ENTRYPOINT ["./lora_kernel"]
