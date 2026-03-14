FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# -----------------------------
# 1️⃣ Basic system packages
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 2️⃣ Install Micromamba
# -----------------------------
# Install Micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

RUN curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && mv bin/micromamba /usr/local/bin/micromamba


# -----------------------------
# 3️⃣ Copy environment files
# -----------------------------
WORKDIR /tmp
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt

# -----------------------------
# 4️⃣ Create environment
# -----------------------------
RUN micromamba create -y -n fastbook -f /tmp/environment.yml -c conda-forge \
    && micromamba clean --all --yes

# -----------------------------
# 5️⃣ Set environment as default
# -----------------------------
SHELL ["/bin/bash", "-lc"]

# Initialize micromamba for bash and auto-activate fastbook in interactive shells
# Use single quotes to avoid expanding the hook during build.
RUN echo "export MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX" >> /etc/bash.bashrc \
    && echo "export MAMBA_EXE=/usr/local/bin/micromamba" >> /etc/bash.bashrc \
    && echo 'eval "$("$MAMBA_EXE" shell hook --shell bash)"' >> /etc/bash.bashrc \
    && echo "micromamba activate fastbook" >> /etc/bash.bashrc

# -----------------------------
# 6️⃣ Optional: test PyTorch + CUDA
# -----------------------------
RUN micromamba run -n fastbook python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
EOF

# -----------------------------
# 7️⃣ Default command
# -----------------------------
# CMD ["bash"]
EXPOSE 8888
CMD ["bash"]