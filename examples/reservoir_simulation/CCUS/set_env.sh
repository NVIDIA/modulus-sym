#!/bin/bash

echo "Installing opm-simulators program (including Flow). This may take a few minutes."
echo "This install happens only during container launch."

apt-get update > install.log 2>&1 
apt-get install -y apt-utils make wget vim software-properties-common >> install.log 2>&1
add-apt-repository -y ppa:opm/ppa >> install.log 2>&1
apt-get update >> install.log 2>&1
apt-get install -y libopm-simulators-bin >> install.log 2>&1

if [ $? -eq 0 ]; then
	echo "Setup complete, container ready for use."
else
	echo "Flow failed to install. Some features might not work."
fi

/opt/nvidia/nvidia_entrypoint.sh "$@"
