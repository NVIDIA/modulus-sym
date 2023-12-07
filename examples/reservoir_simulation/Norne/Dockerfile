# Use the NVIDIA Modulus image as the base image
FROM nvcr.io/nvidia/modulus/modulus:23.09

# Clone and build MPSlib
RUN cd /workspace && git clone https://github.com/ergosimulation/mpslib.git MPSlib && cd MPSlib && make -j 

# Install Python packages
RUN pip3 install tensorflow==2.9.1 scikit-mps kneed pyDOE FyeldGenerator py-cpuinfo gdown gstools scikit-image ema-pytorch accelerate loky xgboost scikit-learn pandas

# Set the working directory
WORKDIR /workspace/project

COPY set_env.sh /opt/nvidia/set_env.sh

ENTRYPOINT ["/opt/nvidia/set_env.sh"]
