FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace