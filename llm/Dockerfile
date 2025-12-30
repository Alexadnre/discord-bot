FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1. Installation des outils système
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Préparation de l'environnement de compilation CUDA
# On indique au compilateur où se trouvent les bibliothèques CUDA Stubs
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUDA=on
ENV CMAKE_ARGS="-DLLAMA_CUDA=on"
ENV FORCE_CMAKE=1

# AJOUT CRUCIAL : On lie les stubs CUDA pour que le linker les trouve pendant le build
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# 3. Installation des outils de build Python
RUN pip3 install --upgrade pip
RUN pip3 install setuptools wheel scikit-build-core ninja

# 4. Installation des dépendances
COPY requirements.txt .

# On crée un lien symbolique temporaire pour libcuda.so (souvent manquant dans l'env de build)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && pip3 install --no-cache-dir -r requirements.txt \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

COPY . .

EXPOSE 8000

CMD ["python3", "main.py"]