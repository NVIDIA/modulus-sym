#!/bin/bash
ext=`date +"%y_%m.tgz"`
root=.
mandatory="$root/Energy_SDK_License_Agreement.pdf $root/README.md"
common="$root/Dockerfile $root/src $root/Visuals $root/scripts $root/PACKETS $root/NORNE $root/Numerical_experiment $root/Norne_Initial_ensemble $root/Necessaryy"

#doc=$root/Documents
excl="--exclude=*/.* --exclude=*/.* --exclude=*/venv/* --exclude=*/__pycache__/* --exclude=*/outputs/* --exclude=*/venv-mtc-lab --exclude=*/.ipynb_checkpoints --exclude=*/*.pptx --exclude=*/.vscode/* --exclude=*/.gitattributes --exclude=*/.gitkeep"

tar cvfz "Nvidia_EnergySDK_ResSim_HistoryMatching_v"$ext $excl $mandatory $common 
