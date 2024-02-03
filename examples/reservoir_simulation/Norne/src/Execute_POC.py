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

import subprocess

# Define the scripts to run
scripts_to_run = [
    {
        "name": "3D PINO APPROACH",
        "script": "Forward_problem_PINO.py",
        "description": "NVIDIA PINO surrogate building of the pressure, water saturation and gas saturation fields",
    },
    {
        "name": "PEACEMANN MODELLING - CCR Approach",
        "script": "Learn_CCR.py",
        "description": "Peacemann model surrogate using a mixture of experts approach",
        "references": [
            "(1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,",
            "Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general",
            "method for learning discontinuous functions.Foundations of Data Science,",
            "1(2639-8001-2019-4-491):491, 2019.",
            "",
            "(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of",
            "Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.",
        ],
    },
    {
        "name": "Inverse Problem",
        "script": "Inverse_problem.py",
        "description": "adaptive Regularised Ensemble Kalman Inversion method for solution of the Inverse problem",
    },
]

total_scripts = len(scripts_to_run)
scripts_completed = 0

# Function to run a script and track progress
def run_script(script_info):
    global scripts_completed
    scripts_completed += 1
    print(f'Script {scripts_completed}/{total_scripts}: {script_info["name"]}')
    print("---------------------------------------------------------------")
    print("Description:")
    print(script_info["description"])
    print("---------------------------------------------------------------")
    if "references" in script_info:
        print("References:")
        for reference in script_info["references"]:
            print(reference)
    subprocess.run(["python", script_info["script"]])
    print("-------------------PROGRAM EXECUTED-----------------------------------")
    print("\n")


# Print the initial message
print(
    f"{total_scripts} scripts will be executed in this NVIDIA Reservoir characterisation POC"
)

# Loop through and run the scripts
for script_info in scripts_to_run:
    run_script(script_info)

print("Executed.")
