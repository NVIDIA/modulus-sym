# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:24.03-py3
FROM $BASE_CONTAINER as builder

ARG TARGETPLATFORM

# Update pip and setuptools
RUN pip install "pip==23.2.1" "setuptools==68.2.2"  

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

# install vtk
COPY . /modulus-sym/
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ -e "/modulus-sym/deps/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl" ]; then \
	echo "VTK wheel for $TARGETPLATFORM exists, installing!" && \
	pip install --no-cache-dir /modulus-sym/deps/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
	echo "Installing vtk for: $TARGETPLATFORM" && \
	pip install --no-cache-dir "vtk>=9.2.6"; \ 
    else \
	echo "Installing vtk for: $TARGETPLATFORM from source" && \
	apt-get update && apt-get install -y libgl1-mesa-dev && \
	git clone https://gitlab.kitware.com/vtk/vtk.git && cd vtk && git checkout tags/v9.2.6 && git submodule update --init --recursive && \
	mkdir build && cd build && cmake -GNinja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON /workspace/vtk/ && ninja && \
	python setup.py bdist_wheel && \
	pip install --no-cache-dir dist/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl && \
	cd ../../ && rm -r vtk; \
    fi

# Install modulus sym dependencies
RUN pip install --no-cache-dir "hydra-core>=1.2.0" "termcolor>=2.1.1" "chaospy>=4.3.7" "Cython==0.29.28" "numpy-stl==2.16.3" "opencv-python==4.5.5.64" \
    "scikit-learn==1.0.2" "symengine>=0.10.0" "sympy==1.12" "timm==0.5.4" "torch-optimizer==0.3.0" "transforms3d==0.3.1" \
    "typing==3.7.4.3" "pillow==10.0.1" "notebook==6.4.12" "mistune==2.0.3" "pint==0.19.2" "tensorboard>=2.8.0"

# Install tiny-cuda-nn
ENV TCNN_CUDA_ARCHITECTURES="60;70;75;80;86;90"
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ -e "/modulus-sym/deps/tinycudann-1.7-cp310-cp310-linux_x86_64.whl" ]; then \
        echo "Tiny CUDA NN wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /modulus-sym/deps/tinycudann-1.7-cp310-cp310-linux_x86_64.whl; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ -e "/modulus-sym/deps/tinycudann-1.7-cp310-cp310-linux_aarch64.whl" ]; then \
        echo "Tiny CUDA NN wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /modulus-sym/deps/tinycudann-1.7-cp310-cp310-linux_aarch64.whl; \
    else \
        echo "No Tiny CUDA NN wheel present, building from source" && \
	pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/@master#subdirectory=bindings/torch; \	
    fi


FROM builder as pysdf-install

ARG TARGETPLATFORM

COPY . /modulus-sym/
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
	cp /modulus-sym/deps/NVIDIA-OptiX-SDK-7.3.0-linux64-aarch64.sh /modulus-sym/ && \
	cd /modulus-sym && ./NVIDIA-OptiX-SDK-7.3.0-linux64-aarch64.sh --skip-license --include-subdir --prefix=/root && \
	cd /root && \
	wget  https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-aarch64.tar.gz && \
	tar xvfz cmake-3.24.1-linux-aarch64.tar.gz && \
	cp -r /modulus-sym/deps/external /external/ && \
	mkdir /external/pysdf/build/ && \
	cd /external/pysdf/build && \
	/root/cmake-3.24.1-linux-aarch64/bin/cmake .. -DGIT_SUBMODULE=OFF -DOptiX_INSTALL_DIR=/root/NVIDIA-OptiX-SDK-7.3.0-linux64-aarch64 -DCUDA_CUDA_LIBRARY="" && \
	make -j && \
	mkdir /external/lib && \
	cp libpysdf.so /external/lib/ && \
	cd /external/pysdf && pip install . ; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
	cp /modulus-sym/deps/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64.sh /modulus-sym/ && \
	cd /modulus-sym && ./NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64.sh --skip-license --include-subdir --prefix=/root && \
	cd /root && \
	wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.tar.gz && \
	tar xvfz cmake-3.24.1-linux-x86_64.tar.gz && \
	cp -r /modulus-sym/deps/external /external/ && \
	mkdir /external/pysdf/build/ && \
	cd /external/pysdf/build && \
	/root/cmake-3.24.1-linux-x86_64/bin/cmake .. -DGIT_SUBMODULE=OFF -DOptiX_INSTALL_DIR=/root/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64 -DCUDA_CUDA_LIBRARY="" && \
	make -j && \
	mkdir /external/lib && \
	cp libpysdf.so /external/lib/ && \
	cd /external/pysdf && pip install . ; \
    fi

# Cleanup
RUN rm -rf /root/NVIDIA-OptiX-SDK* /root/cmake* /external/pysdf

ENV LD_LIBRARY_PATH="/external/lib:${LD_LIBRARY_PATH}" \
     NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,video \
    _CUDA_COMPAT_TIMEOUT=90

# CI Image
FROM pysdf-install as ci

# Install Modulus
RUN pip install --upgrade --no-cache-dir git+https://github.com/NVIDIA/modulus.git

RUN pip install --no-cache-dir "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0"
COPY . /modulus-sym/
RUN cd /modulus-sym/ && pip install -e . --no-deps && rm -rf /modulus-sym/

# Image without pysdf
FROM builder as no-pysdf

# Install modulus sym
COPY . /modulus-sym/
RUN cd /modulus-sym/ && pip install --no-cache-dir . --no-deps
RUN rm -rf /modulus-sym/

# Image with pysdf 
# Install pysdf
FROM pysdf-install as deploy

# Install modulus sym
COPY . /modulus-sym/
RUN cd /modulus-sym/ && pip install --no-cache-dir . --no-deps
RUN rm -rf /modulus-sym/

# Set Git Hash as a environment variable
ARG MODULUS_SYM_GIT_HASH
ENV MODULUS_SYM_GIT_HASH=${MODULUS_SYM_GIT_HASH:-unknown}

# Docs image
FROM deploy as docs
# Install packages for Sphinx build
RUN pip install --no-cache-dir "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.2/pandoc-3.1.2-linux-amd64.tar.gz && tar xvzf pandoc-3.1.2-linux-amd64.tar.gz --strip-components 1 -C /usr/local/ 
