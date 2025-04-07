# Set build arguments
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.2.0
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Base image with NVIDIA CUDA
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

# Set environment variables
ENV TZ=Etc/UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Set timezone and install system dependencies
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget curl git cmake build-essential pkg-config bash \
    xvfb x11-apps ca-certificates \
    libx11-6 libxext6 libxrender1 libxft2 libtcl8.6 libtk8.6 python3-tk \
    libusb-1.0-0-dev libgl1-mesa-dev libboost-all-dev pybind11-dev \
    libopencv-dev libprotobuf-dev protobuf-compiler libhdf5-dev hdf5-tools \
    libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg && \ 
    rm -rf /var/lib/apt/lists/*

# Install Python and dependencies
ARG PYTHON_VERSION
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    h5py numpy matplotlib opencv-python pandas scipy dtaidistance \
    pytorch-lightning torchvision torch fire wandb torchprofile onnx scikit-learn dotenv \
    pybind11 tensorboard tensorboardX

# Set working directory
WORKDIR /workspace/code/ 

# Attach a volume for persistent storage
VOLUME ["/workspace/data"]

# Copy project files
COPY . /workspace/code

# Set the default command
CMD ["/bin/bash"]