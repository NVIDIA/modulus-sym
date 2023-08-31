#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:33:56 2023

@author: clementetienam
"""
import os
from modulus.sym.hydra import to_absolute_path
from modulus.sym.key import Key
from NVRS import *
from modulus.sym.models.fno import *
from modulus.sym.models.afno.afno import *
import shutil
import pandas as pd
import scipy.io as sio
import torch
import yaml
from PIL import Image

oldfolder = os.getcwd()
os.chdir(oldfolder)

data = []
os.chdir('../COMPARE_RESULTS/FNO')
True_measurement = pd.read_csv('RSM_NUMERICAL.csv') 
True_measurement = True_measurement.values.astype(np.float32)[:,1:]

data.append(True_measurement)


FNO = pd.read_csv('RSM_MODULUS.csv') 
FNO = FNO.values.astype(np.float32)[:,1:]
data.append(FNO)

os.chdir(oldfolder)

os.chdir('../COMPARE_RESULTS/PINO')
PINO = pd.read_csv('RSM_MODULUS.csv') 
PINO = PINO.values.astype(np.float32)[:,1:]
data.append(PINO)
os.chdir(oldfolder)

os.chdir(oldfolder)


os.chdir('../COMPARE_RESULTS')
Plot_Models(data)
Plot_bar(data)
os.chdir(oldfolder)
