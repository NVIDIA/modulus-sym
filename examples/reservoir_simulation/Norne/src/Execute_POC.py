"""
//////////////////////////////////////////////////////////////////////////////
Copyright (C) NVIDIA Corporation.  All rights reserved.

NVIDIA Sample Code for the Norne field

Please refer to the NVIDIA end user license agreement (EULA) associated

with this source code for terms and conditions that govern your use of

this software. Any use, reproduction, disclosure, or distribution of

this software and related documentation outside the terms of the EULA

is strictly prohibited.

//////////////////////////////////////////////////////////////////////////////
@Author: Clement Etienam
"""
import subprocess

# Define the scripts to run
scripts_to_run = [
    {
        'name': '3D PINO APPROACH',
        'script': 'Forward_problem_PINO.py',
        'description': 'NVIDIA PINO surrogate building of the pressure, water saturation and gas saturation fields'
    },
    {
        'name': 'PEACEMANN MODELLING - CCR Approach',
        'script': 'Learn_CCR.py',
        'description': 'Peacemann model surrogate using a mixture of experts approach',
        'references': [
            '(1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,',
            'Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general',
            'method for learning discontinuous functions.Foundations of Data Science,',
            '1(2639-8001-2019-4-491):491, 2019.',
            '',
            '(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of',
            'Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.'
        ]
    },
    {
        'name': 'Inverse Problem',
        'script': 'Inverse_problem.py',
        'description': 'adaptive Regularised Ensemble Kalman Inversion method for solution of the Inverse problem'
    },
]

total_scripts = len(scripts_to_run)
scripts_completed = 0

# Function to run a script and track progress
def run_script(script_info):
    global scripts_completed
    scripts_completed += 1
    print(f'Script {scripts_completed}/{total_scripts}: {script_info["name"]}')
    print('---------------------------------------------------------------')
    print('Description:')
    print(script_info['description'])
    print('---------------------------------------------------------------')
    if 'references' in script_info:
        print('References:')
        for reference in script_info['references']:
            print(reference)
    subprocess.run(["python", script_info['script']])
    print('-------------------PROGRAM EXECUTED-----------------------------------')
    print('\n')

# Print the initial message
print(f'{total_scripts} scripts will be executed in this NVIDIA Reservoir characterisation POC')

# Loop through and run the scripts
for script_info in scripts_to_run:
    run_script(script_info)

print('Executed.')


