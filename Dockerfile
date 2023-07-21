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

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:23.06-py3
FROM $BASE_CONTAINER as builder

# Update pip and setuptools
RUN pip install --upgrade pip setuptools  

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

# Install tiny-cuda-nn
ENV TCNN_CUDA_ARCHITECTURES="60;70;75;80;86;90"
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/@master#subdirectory=bindings/torch

FROM builder as pysdf-install
# Extract OptiX 7.0.0 SDK and CMake 3.18.2
COPY ./deps/NVIDIA-OptiX-SDK-7.0.0-linux64.sh /modulus-sym/
RUN cd /modulus-sym && ./NVIDIA-OptiX-SDK-7.0.0-linux64.sh --skip-license --include-subdir --prefix=/root
RUN cd /root && \
     wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.tar.gz && \
     tar xvfz cmake-3.24.1-linux-x86_64.tar.gz

# Build libsdf.so
COPY ./deps/external /external/
RUN mkdir /external/pysdf/build/ && \
        cd /external/pysdf/build && \
        /root/cmake-3.24.1-linux-x86_64/bin/cmake .. -DGIT_SUBMODULE=OFF -DOptiX_INSTALL_DIR=/root/NVIDIA-OptiX-SDK-7.0.0-linux64 -DCUDA_CUDA_LIBRARY="" && \
        make -j && \
        mkdir /external/lib && \
        cp libpysdf.so /external/lib/ && \
        cd /external/pysdf && pip install .

ENV LD_LIBRARY_PATH="/external/lib:${LD_LIBRARY_PATH}" \
     NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,video \
    _CUDA_COMPAT_TIMEOUT=90

# CI Image
FROM pysdf-install as ci
RUN pip install "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0"
COPY . /modulus-sym/
RUN cd /modulus-sym/ && pip install -e . && rm -rf /modulus-sym/

# Image without pysdf
FROM builder as no-pysdf

# Install modulus sym
COPY . /modulus-sym/
RUN cd /modulus-sym/ && pip install .
RUN rm -rf /modulus-sym/

# Image with pysdf 
# Install pysdf
FROM pysdf-install as deploy

# Install modulus sym
COPY . /modulus-sym/
RUN cd /modulus-sym/ && pip install .
RUN rm -rf /modulus-sym/

# Docs image
FROM deploy as docs
# Install packages for Sphinx build
RUN pip install "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.2/pandoc-3.1.2-linux-amd64.tar.gz && tar xvzf pandoc-3.1.2-linux-amd64.tar.gz --strip-components 1 -C /usr/local/ 
