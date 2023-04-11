To build release image, you will need to do the below preliminary steps:

Clone this repo, and download the Optix SDK from https://developer.nvidia.com/designworks/optix/downloads/legacy. 
```
git clone https://github.com/NVIDIA/modulus-sym.git
cd modulus-sym/ && mkdir deps
```
Currently Modulus supports v7.0. Place the Optix file in the deps directory and make it executable. Also clone the pysdf library in the deps folder (NVIDIA Internal)
```
chmod +x deps/NVIDIA-OptiX-SDK-7.0.0-linux64.sh 
git clone <internal pysdf repo>
```

Then to build the image, insert next tag and run below:
```
docker build -t modulus-sym -f Dockerfile .. 
```

Alternatively, if you want to skip pysdf installation, you can run the following:
```
docker build -t modulus-sym -f Dockerfile --target no-pysdf ..
```
