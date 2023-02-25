# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG PYT_VER=22.08
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3

# Specify poetry version
ENV POETRY_VERSION=1.2

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Cache dependencies
COPY pyproject.toml ./

# Copy files into container
COPY . /modulus/

# Extract OptiX 7.0.0 SDK and CMake 3.18.2
RUN cd /modulus && ./NVIDIA-OptiX-SDK-7.0.0-linux64.sh --skip-license --include-subdir --prefix=/root
RUN cd /root && \
    wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.tar.gz && \
    tar xvfz cmake-3.24.1-linux-x86_64.tar.gz

# Build libsdf.so
RUN mkdir /modulus/external/pysdf/build && \
    cd /modulus/external/pysdf/build && \
    /root/cmake-3.24.1-linux-x86_64/bin/cmake .. -DGIT_SUBMODULE=OFF -DOptiX_INSTALL_DIR=/root/NVIDIA-OptiX-SDK-7.0.0-linux64 -DCUDA_CUDA_LIBRARY="" && \
    make -j && \
    mkdir /modulus/external/lib && \
    cp libpysdf.so /modulus/external/lib/

ENV LD_LIBRARY_PATH="/modulus/external/lib:${LD_LIBRARY_PATH}" \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,video \
    _CUDA_COMPAT_TIMEOUT=90

# Install pysdf
RUN cd /modulus/external/pysdf && python setup.py install

# Install tiny-cuda-nn
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/@master#subdirectory=bindings/torch

# Install functorch
# Notes about how to find the corresponding functorch commit for a PyTorch container:
# 1. Find out the PyTorch commit of the NGC container (e.g., 1.13.0a0+d321be6), it shows on the top of the message after you log in to the container.
# 2. Find out the date of this PyTorch commit.
# 3. Find a functorch commit near this date, and it should build successfully.
RUN pip install git+https://github.com/pytorch/functorch.git@8a5465a72330a2d82df577e7211d57b3cfde664e

# Install modulus and dependencies
RUN cd /modulus && \
     poetry config virtualenvs.create false && \
     poetry install --no-interaction

# Copy Pysdf egg file
RUN mkdir /modulus/external/eggs
RUN cp -r /modulus/external/pysdf/dist/pysdf-0.1-py3.8-linux-x86_64.egg /modulus/external/eggs

# Cleanup
RUN rm -rf /root/NVIDIA-OptiX-SDK-7.0.0-linux64 /root/cmake-3.24.1-linux-x86_64 /modulus/external/pysdf  /modulus/.git*
RUN rm -fv /modulus/setup.py /modulus/setup.cfg /modulus/MANIFEST.in

WORKDIR /examples
