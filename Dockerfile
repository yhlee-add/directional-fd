# syntax=docker/dockerfile:1


# ============================================================================
# Stage 1: base — CUDA 12.8 toolkit + python + uv + torch 2.9.0/cu128 venv
# ============================================================================
ARG PY_VERSION=3.10
ARG DOCKER_CUDA_VERSION=12.8.1
FROM nvidia/cuda:${DOCKER_CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS base
ARG PY_VERSION
ENV PY_VERSION=${PY_VERSION}

ENV TORCH_CUDA_ARCH_LIST='8.0 8.6 9.0 10.0+PTX'
ENV MAX_JOBS=2 NVCC_THREADS=8 DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda SETUPTOOLS_USE_DISTUTILS=stdlib

# system + python
RUN apt-get update -y && apt-get install -y \
    vim git bzip2 tmux wget tar htop unzip curl \
    ssh openssh-server software-properties-common \
    gcc-12 g++-12 ninja-build cmake build-essential autoconf libtool automake \
    ffmpeg libsm6 libxext6 libgl1 \
    mesa-utils x11-apps freeglut3-dev libglu1-mesa-dev mesa-common-dev \
    libxkbfile-dev libgl1-mesa-glx
RUN mkdir -p /var/run/sshd && echo "root:root" | chpasswd \
    && echo "Port 22" >> /etc/ssh/sshd_config \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
EXPOSE 22
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update -y \
    && apt-get install -y python${PY_VERSION} python${PY_VERSION}-dev python${PY_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PY_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PY_VERSION} \
    && ln -sf /usr/bin/python${PY_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PY_VERSION} \
    && apt-get install -y python3-setuptools libpython${PY_VERSION}-dev

# mongodb
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
    gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
    --dearmor
RUN echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-8.0.list
RUN apt-get update && apt-get install -y mongodb-org
RUN mkdir -p /data/db

# uv + common python tools
RUN curl -LsSf https://astral.sh/uv/0.8.14/install.sh | sh
ENV PATH=/root/.local/bin:$PATH
SHELL ["/bin/bash", "-c"]
RUN uv python pin ${PY_VERSION}
RUN uv pip install --no-cache --system ipython tqdm rich jupyter jupyterlab ipykernel pandas \
    einops safetensors pyyaml requests psutil opencv-python-headless matplotlib seaborn \
    scikit-learn scipy pillow tensorboard h5py triton

# torch 2.9.0 + cu128 venv
ENV UV_NO_CACHE=1
COPY <<'EOF' /usr/local/bin/install_torch_env
#!/bin/bash
set -e
TORCH_VER=$1; VISION_VER=$2; CUDA_VER=$3; FLASH_ATTN_LINK=$4
BASE_DIR=/ddiff-base/py${PY_VERSION}-torch${TORCH_VER}
mkdir -p "$BASE_DIR" && cd "$BASE_DIR"
uv venv --python "${PY_VERSION}" --system-site-packages --seed
uv pip install --no-cache torch=="${TORCH_VER}" torchvision=="${VISION_VER}" torchaudio=="${TORCH_VER}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VER}"
if [ -n "$FLASH_ATTN_LINK" ]; then uv pip install --no-cache "$FLASH_ATTN_LINK"; fi
printf 'ln -sfn %s/.venv ./.venv && [ -f pyproject.toml ] && uv add torch==%s torchvision==%s torchaudio==%s; echo torch %s ready\n' \
    "$BASE_DIR" "$TORCH_VER" "$VISION_VER" "$TORCH_VER" "$TORCH_VER" > /usr/local/bin/uv_init_torch${TORCH_VER}
chmod +x /usr/local/bin/uv_init_torch${TORCH_VER}
EOF

RUN chmod +x /usr/local/bin/install_torch_env
RUN install_torch_env 2.9.0 0.24.0 cu128 \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

# VBench venv, isolated from the generation venv
ENV VBENCH_VENV=/ddiff-base/py${PY_VERSION}-torch2.5.1
RUN install_torch_env 2.5.1 0.20.1 cu121
RUN cd $VBENCH_VENV && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0 8.6 9.0+PTX' \
    uv pip install --no-cache --no-build-isolation \
    git+https://github.com/facebookresearch/detectron2.git
# VBench's setup.py requires torch.cuda.is_available(), which is false during
# docker build (no GPU); comment the check out before installing.
RUN git clone https://github.com/Vchitect/VBench.git /tmp/VBench \
    && git -C /tmp/VBench checkout 45e79ec14e69a2187202c675d2dbce1a71843d53 \
    && sed -i 's/^check_torch_version()/# &/' /tmp/VBench/setup.py \
    && cd $VBENCH_VENV && uv pip install --no-cache --no-build-isolation /tmp/VBench \
    && rm -rf /tmp/VBench

# La-Proteina venv, isolated with its own torch 2.7/cu118 env, for protein
# generation. environment.yaml is a conda file, but every dep is pip-installable
# except the eval-only mmseqs2; python is 3.10 here vs the upstream 3.11.
ENV LAPROTEINA_VENV=/ddiff-base/py${PY_VERSION}-torch2.7.0
ENV LAPROTEINA_REPO=/opt/la-proteina
RUN install_torch_env 2.7.0 0.22.0 cu118
COPY docker/laproteina/pyproject.toml /tmp/laproteina/pyproject.toml
RUN cd $LAPROTEINA_VENV && uv pip install --no-cache -r /tmp/laproteina/pyproject.toml \
    && uv pip install --no-cache --no-deps graphein==1.7.7 \
    && uv pip install --no-cache \
        torch_geometric torch_scatter torch_sparse torch_cluster \
        -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
RUN git clone https://github.com/NVIDIA-BioNeMo/la-proteina $LAPROTEINA_REPO \
    && git -C $LAPROTEINA_REPO checkout cde5de3ead6e4d76f367da6dc5174be9913ef6ca


# ============================================================================
# Stage 2: app — experiment tooling + project deps
# ============================================================================
FROM base AS app

WORKDIR /root/

RUN apt update -y && apt-get install -y zsh zip bmon supervisor
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && sed -i 's/^ZSH_THEME=".*"/ZSH_THEME="frisk"/' ~/.zshrc
RUN sed -i 's|^#[ \t]*export PATH=$HOME/bin|export PATH=$HOME/bin|' ~/.zshrc && printf '\nexport LC_ALL=C.UTF-8\nexport LANG=C.UTF-8\n' >> ~/.zshrc && chsh -s /usr/bin/zsh root

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
RUN bash -c "source ~/.nvm/nvm.sh && nvm install 22 && npm install -g omniboard"

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord"]

ENV commit_id=1a5daa3a0231a0fbba4f14db7ec463cf99d7768e
RUN curl -sSL "https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable" \
    -o vscode-server-linux-x64.tar.gz \
    && mkdir -p ~/.vscode-server/bin/${commit_id} \
    && tar zxvf vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${commit_id} --strip 1 \
    && touch ~/.vscode-server/bin/${commit_id}/0

RUN apt-get install -y libgl1 libxcb-cursor0 libxcb-xinerama0 libxcb-xfixes0 libxkbcommon-x11-0 libfontconfig1 dbus-x11
RUN git clone https://github.com/ingydotnet/git-subrepo && echo 'source /root/git-subrepo/.rc' >> ~/.zshrc

WORKDIR /root/build/
RUN bash -c "uv_init_torch2.9.0"
ADD https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl ./flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl
COPY pyproject.toml pyproject.toml
RUN uv sync --no-install-project
