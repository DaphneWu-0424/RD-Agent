FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Qlib currently uses MLflow's filesystem tracking backend.
# MLflow 3.x requires explicit opt-in for this legacy backend.
ENV MLFLOW_ALLOW_FILE_STORE=true

# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

# Local build proxy for GitHub access.
# Build with host networking so 127.0.0.1 points to the Docker host.
ARG GIT_PROXY=http://127.0.0.1:7897

# Use a mainland China Ubuntu mirror for apt packages.
RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
    /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y \
        curl \
        vim \
        git \
        build-essential \
        coreutils \
    && rm -rf /var/lib/apt/lists/*

# GitHub goes through the local proxy.
RUN git -c http.proxy="${GIT_PROXY}" \
    clone https://github.com/microsoft/qlib.git

WORKDIR /workspace/qlib

RUN git -c http.proxy="${GIT_PROXY}" fetch \
    && git reset 2fb9380b342556ddb50a4b24e4fe8655d548b2b8 --hard

# Use Tsinghua PyPI mirror for Python dependencies.
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install --upgrade cython
RUN python -m pip install -e .

RUN pip install catboost
RUN pip install xgboost
RUN pip install tables
