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

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:22.12-py3
FROM $BASE_CONTAINER as no-pysdf

# Update pip and setuptools
RUN pip install --upgrade pip setuptools  

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

# Install tiny-cuda-nn
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/@master#subdirectory=bindings/torch

# Install modulus sym
COPY . /modulus-sym/
RUN cd /modulus-sym/ && pip install .
RUN rm -rf /modulus-sym/

# Install pysdf
FROM no-pysdf as with-pysdf
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
