
"""
//////////////////////////////////////////////////////////////////////////////
Copyright (C) NVIDIA Corporation.  All rights reserved.

NVIDIA Sample Code

Please refer to the NVIDIA end user license agreement (EULA) associated

with this source code for terms and conditions that govern your use of

this software. Any use, reproduction, disclosure, or distribution of

this software and related documentation outside the terms of the EULA

is strictly prohibited.

//////////////////////////////////////////////////////////////////////////////


@author : clement etienam
"""
import os
from PIL import Image

print('')
print('Now - Creating GIF')
import glob

oldfolder = os.getcwd()
os.chdir(oldfolder)

choice = int(input('What Surrogate model did you train for?: \n\
1 = FNO\n\
2 = PINO\n\
3 = AFNOD\n\
4 = AFNOP\n\
'))


if choice ==1:
    os.chdir('outputs/Forward_problem_FNO/ResSim/validators')
elif choice ==2:
    os.chdir('outputs/Forward_problem_PINO/ResSim/validators')
elif choice ==3:
    os.chdir('outputs/Forward_problem_AFNOD/ResSim/validators')
else:
    os.chdir('outputs/Forward_problem_AFNOP/ResSim/validators')
    
frames=[]
imgs=sorted(glob.glob("*test_pressure_pressure_simulations*"), key=os.path.getmtime)
for i in imgs:
    new_frame=Image.open(i)
    frames.append(new_frame)

frames[0].save('pressure_test.gif',format='GIF',\
                append_images=frames[1:],save_all=True,duration=500,loop=0)
    
    
frames1=[]
imgs=sorted(glob.glob("*test_saturation_saturation_simulations*"), key=os.path.getmtime)
for i in imgs:
    new_frame=Image.open(i)
    frames1.append(new_frame)

frames1[0].save('saturation_test.gif',format='GIF',\
                append_images=frames1[1:],save_all=True,duration=500,loop=0) 
    
    
# frames=[]
# imgs=sorted(glob.glob("*val_pressure_pressure_simulations*"), key=os.path.getmtime)
# for i in imgs:
#     new_frame=Image.open(i)
#     frames.append(new_frame)

# frames[0].save('pressure_val.gif',format='GIF',\
#                 append_images=frames[1:],save_all=True,duration=500,loop=0)
    
    
# frames1=[]
# imgs=sorted(glob.glob("*val_saturation_saturation_simulations*"), key=os.path.getmtime)
# for i in imgs:
#     new_frame=Image.open(i)
#     frames1.append(new_frame)

# frames1[0].save('saturation_val.gif',format='GIF',\
#                 append_images=frames1[1:],save_all=True,duration=500,loop=0) 
    


from glob import glob
for f3 in glob("*test_saturation_saturation_simulations*"):
    os.remove(f3)
    
for f4 in glob("*test_pressure_pressure_simulations*"):
    os.remove(f4)   

# for f5 in glob("*val_pressure_pressure_simulations*"):
#     os.remove(f5) 
    
# for f6 in glob("*val_saturation_saturation_simulations*"):
#     os.remove(f6)

os.chdir(oldfolder)