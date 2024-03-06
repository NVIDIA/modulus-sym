#!/bin/bash

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
