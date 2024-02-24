# Use the NVIDIA Modulus image as the base image
FROM nvcr.io/nvidia/modulus/modulus:23.09

# Set user and group arguments
ARG MYUSER=user
ARG MYUID=1000
ARG MYGID=1000
ARG PROCS=8

# Set build type argument and configure environment
ARG BUILD_TYPE=Release
ENV DEBIAN_FRONTEND noninteractive

# Install cmake, add user, and other dependencies
RUN cd /workspace && \
    if [ -f /usr/bin/yum ]; then yum install -y make wget; fi && \
    if [ -f /usr/bin/apt-get ]; then apt-get update && apt-get install -y apt-utils make wget vim; fi && \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.23.0/cmake-3.23.0-linux-x86_64.tar.gz && \
    tar xf cmake-3.23.0-linux-x86_64.tar.gz && rm cmake-3.23.0-linux-x86_64.tar.gz && \
    groupadd -f -g ${MYGID} ${MYUSER} && \
    useradd -rm -u $MYUID -g $MYUSER -p "" $MYUSER && \
    chown ${MYUSER}:${MYGID} /workspace

# Clone and build MPSlib
RUN cd /workspace && git clone https://github.com/ergosimulation/mpslib.git MPSlib && cd MPSlib && make -j ${PROCS}

# Install Flow OPM simulator
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:opm/ppa && apt-get update
RUN apt-get install -y opm-simulators

# Install Python packages
RUN pip3 install tensorflow==2.9.1 scikit-mps kneed pyDOE FyeldGenerator py-cpuinfo gdown gstools scikit-image ema-pytorch accelerate loky xgboost scikit-learn pandas

# Set the working directory
WORKDIR /workspace/project

# Define the CMD or ENTRYPOINT as needed for your application
# CMD /workspace/compile_amgx.sh
