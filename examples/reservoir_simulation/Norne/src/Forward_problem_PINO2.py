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
from NVRS import *
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)
import modulus
#import torch
import gc
from modulus.sym.hydra import  ModulusConfig
#from modulus.hydra import to_absolute_path
from modulus.sym.key import Key
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator,PointwiseValidator

from modulus.sym.domain.constraint.continuous import PointwiseConstraint
from modulus.sym.dataset import DictGridDataset
from modulus.sym.utils.io.plotter import GridValidatorPlotter
from modulus.sym.models.fully_connected import *
from modulus.sym.models.fno import *
from modulus.sym.node import Node
from skimage.transform import resize
#from sklearn.model_selection import train_test_split
import requests


#import cupy
from typing import Dict
import torch.nn.functional as F
from ops import dx, ddx,compute_differential,compute_second_differential
from modulus.sym.utils.io.plotter import ValidatorPlotter
from typing import Union, Tuple
from pathlib import Path


def to_absolute_path_and_create(*args: Union[str, Path]) -> Union[Path, str, Tuple[Union[Path, str]]]:
    """Converts file path to absolute path based on the current working directory and creates the subfolders."""
    
    out = ()
    base = Path(os.getcwd())
    for path in args:
        p = Path(path)
        
        if p.is_absolute():
            ret = p
        else:
            ret = base / p
        
        # Create the directory/subfolders
        ret.mkdir(parents=True, exist_ok=True)
        
        if isinstance(path, str):
            out = out + (str(ret),)
        else:
            out = out + (ret,)

    if len(args) == 1:
        out = out[0]
    return out

torch.set_default_dtype(torch.float32)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        #params = { 'id' : id, 'confirm' : 1 }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
def dx1(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                -0.5,
                0.0,
                0.5,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output


def ddx1(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute second order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                1.0,
                -2.0,
                1.0,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                1.0 / 90.0,
                -3.0 / 20.0,
                3.0 / 2.0,
                -49.0 / 18.0,
                3.0 / 2.0,
                -3.0 / 20.0,
                1.0 / 90.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx**2) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output

def StoneIIModel (params,device,Sg,Sw):
    #device = params["device"]
    k_rwmax = params["k_rwmax"].to(device)
    k_romax = params["k_romax"].to(device)
    k_rgmax = params["k_rgmax"].to(device)
    n = params["n"].to(device)
    p = params["p"].to(device)
    q = params["q"].to(device)
    m = params["m"].to(device)
    Swi = params["Swi"].to(device)
    Sor = params["Sor"].to(device)
    

    
    denominator = 1 - Swi - Sor

    krw = k_rwmax * ((Sw - Swi) / denominator).pow(n)
    kro = k_romax * (1 - (Sw - Swi) / denominator).pow(p) * (1 - Sg / denominator).pow(q)
    krg = k_rgmax * (Sg / denominator).pow(m)

    return krw, kro, krg

def Black_oil_peacemann(UO,BO,UW,BW,DZ,RE,in_var,out_var,device,max_inn_fcn,max_out_fcn,paramz,p_bub, p_atm,CFO):

    skin = 0
    rwell = 200
    spit = []
    N = in_var.shape[0]
    pwf_producer = 100
    spit = torch.zeros(N).to(device)
    for clement in range(N):

        inn = in_var[clement].T * max_inn_fcn
        outt = out_var[clement].T * max_out_fcn
        
        oil_rate = outt[:,:22]
        water_rate = outt[:,22:44]
        gas_rate = outt[:,44:66]
        
        permeability = inn[:,:22]
        pressure = inn[:,22:44]
        oil = inn[:,44:66]
        gas =inn[:,66:88]
        water =inn[:,88:110]
        
        # Compute Oil rate loss
        krw,kro,krg = StoneIIModel (paramz,device,gas,water)
        BO = calc_bo(p_bub, p_atm, CFO, pressure.mean())
        up = UO * BO
        down = 2 * torch.pi * permeability * kro * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = pressure.mean() - pwf_producer
        qoil = torch.abs(-(drawdown * J))
        loss_oil = torch.sum(torch.abs(qoil - oil_rate)/22)

        # Compute water rate loss
        up = UW * BW
        down = 2 * torch.pi * permeability * krw * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = pressure.mean() - pwf_producer
        qwater = torch.abs(-(drawdown * J))
        loss_water = torch.sum(torch.abs(qwater - water_rate)/22)    
        

        # Compute gas rate loss
        UG = calc_mu_g(pressure.mean())
        BG = calc_bg(p_bub, p_atm, pressure.mean())
        
        up = UG * BG
        down = 2 * torch.pi * permeability * krg * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = pressure.mean() - pwf_producer
        qgas = torch.abs(-(drawdown * J))
        loss_gas = torch.sum(torch.abs(qgas - gas_rate)/22)

        overall_loss = loss_oil + loss_water + loss_gas 
        spit[clement] = overall_loss
        
    loss = spit.mean()
    return loss        
    

# [pde-loss]
# define custom class for black oil model
def Black_oil(neededM,input_var,SWI,SWR,UW,BW,UO,BO,
             nx,ny,nz,SWOW,SWOG,target_min,target_max,minK,maxK,
             minP,maxP,p_bub,p_atm,CFO,maxQx,maxQwx,maxTx,maxQgx,Relperm,params,pde_method):


     u = input_var["pressure"]
     perm = input_var["perm"]
     fin = neededM["Q"]/maxQx
     fin = fin.repeat(u.shape[0], 1, 1, 1, 1)
     finwater = neededM["Qw"]/maxQwx
     finwater = finwater.repeat(u.shape[0], 1, 1, 1, 1)
     dt = neededM["Time"]
     pini = input_var["Pini"]
     poro = input_var["Phi"]
     sini = input_var["Swini"]
     sat = input_var["water_sat"]
     satg = input_var["gas_sat"]
     fault = input_var["fault"]
     fingas = neededM["Qg"]/maxQgx
     fingas = fingas.repeat(u.shape[0], 1, 1, 1, 1)
     actnum = neededM["actnum"]
     actnum = actnum.repeat(u.shape[0], 1, 1, 1, 1)
     sato = 1-(sat + satg)
     siniuse = sini[0,0,0,0,0]    
     dxf = 1.0 / u.shape[3]
     
     
     # fin = fin/maxQx
     # finwater = finwater/maxQwx
     # dt = dt/maxTx
     # fingas = fingas/maxQgx
     
     
     #Rescale back
 
     #pressure
     #u = u * maxP
     
     #Initial_pressure
     #pini = pini * maxP
     #Permeability
     a = perm #* maxK
     
     #Pressure equation Loss
     cuda = 0
     device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")         
 
      
     #print(pressurey.shape)            
     p_loss = torch.zeros_like(u).to(device,torch.float32)
     s_loss = torch.zeros_like(u).to(device,torch.float32)
 
     finusew = finwater
 
      
     prior_pressure = torch.zeros(sat.shape[0],sat.shape[1],\
                         nz,nx,ny).to(device,torch.float32)
     prior_pressure[:,0,:,:,:] = pini[:,0,:,:,:]
     prior_pressure[:,1:,:,:,:] = u[:,:-1,:,:,:]
     
     avg_p = prior_pressure.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)\
     .mean(dim=4, keepdim=True)
     UG = calc_mu_g(avg_p)
     RS = calc_rs(p_bub, avg_p)
     BG = calc_bg(p_bub, p_atm, avg_p)
     #BO = params1['BO']#calc_bo(p_bub, p_atm, CFO, avg_p)
     
  
     avg_p = torch.where(torch.isnan(avg_p), torch.tensor(0.0, device=avg_p.device), avg_p)
     avg_p = torch.where(torch.isinf(avg_p), torch.tensor(0.0, device=avg_p.device), avg_p) 
     
     UG = torch.where(torch.isnan(UG), torch.tensor(0.0, device=UG.device), UG)
     UG = torch.where(torch.isinf(UG), torch.tensor(0.0, device=UG.device), UG)
 
     BG = torch.where(torch.isnan(BG), torch.tensor(0.0, device=BG.device), BG)
     BG = torch.where(torch.isinf(BG), torch.tensor(0.0, device=BG.device), BG)
 
     RS = torch.where(torch.isnan(RS), torch.tensor(0.0, device=RS.device), RS)
     RS = torch.where(torch.isinf(RS), torch.tensor(0.0, device=RS.device), RS)        
 
     # BO = torch.where(torch.isnan(BO), torch.tensor(0.0, device=BO.device), BO)
     # BO = torch.where(torch.isinf(BO), torch.tensor(0.0, device=BO.device), BO)
 
 
     #dsp = u - prior_pressure  #dp
     
     prior_sat = torch.zeros(sat.shape[0],sat.shape[1],\
                         nz,nx,ny).to(device,torch.float32)
     prior_sat[:,0,:,:,:] = siniuse * \
     (torch.ones(sat.shape[0],nz,nx,ny).to(device,torch.float32)) 
     prior_sat[:,1:,:,:,:] = sat[:,:-1,:,:,:] 
     
     prior_gas = torch.zeros(sat.shape[0],sat.shape[1],\
                         nz,nx,ny).to(device,torch.float32)
     prior_gas[:,0,:,:,:] = (torch.zeros(sat.shape[0],nz,nx,ny).\
                         to(device,torch.float32)) 
     prior_gas[:,1:,:,:,:] = satg[:,:-1,:,:,:] 
     
     
     prior_time = torch.zeros(sat.shape[0],sat.shape[1],\
                         nz,nx,ny).to(device,torch.float32)
     prior_time[:,0,:,:,:] = (torch.zeros(sat.shape[0],nz,nx,ny).\
                             to(device,torch.float32)) 
     prior_time[:,1:,:,:,:] = dt[:,:-1,:,:,:]         
      
     dsw = sat - prior_sat #ds 
     #dsw = torch.clip(dsw,0.001,None)
     
    # dsg = satg - prior_gas #ds 
     #dsg = torch.clip(dsg,0.001,None)        
 
     dtime = dt - prior_time #ds 
     dtime[torch.isnan(dtime)] = 0
     dtime[torch.isinf(dtime)] = 0 
     
     
     #Pressure equation Loss  

     if Relperm == 1:                  
     
         #KRW, KRO, KRG = RelPerm(prior_sat,prior_gas, SWI, SWR, SWOW, SWOG)
         one_minus_swi_swr = 1 - (SWI + SWR)
     
     
         soa = torch.divide((1 - (prior_sat + prior_gas) - SWR),one_minus_swi_swr)  
         swa = torch.divide((prior_sat - SWI) ,one_minus_swi_swr)
         sga = torch.divide(prior_gas,one_minus_swi_swr)
         
     
     
         KROW = linear_interp(prior_sat, SWOW[:, 0], SWOW[:, 1])
         KRW = linear_interp(prior_sat, SWOW[:, 0], SWOW[:, 2])
         KROG = linear_interp(prior_gas, SWOG[:, 0], SWOG[:, 1])
         KRG = linear_interp(prior_gas, SWOG[:, 0], SWOG[:, 2])
     
         KRO = (torch.divide(KROW , (1 - swa)) * torch.divide(KROG , (1 - sga))) * soa
     else:
    	 KRW,KRO,KRG = StoneIIModel (params,device,prior_gas,prior_sat)               
     Mw = torch.divide(KRW,(UW * BW))
     Mo = torch.divide(KRO,(UO * BO))
     Mg = torch.divide(KRG,(UG * BG))
     
 
 
     Mg[torch.isnan(Mg)] = 0
     Mg[torch.isinf(Mg)] = 0 
 
 
     Mw = torch.where(torch.isnan(Mw), torch.tensor(0.0, device=Mw.device), Mw)
     Mw = torch.where(torch.isinf(Mw), torch.tensor(0.0, device=Mw.device), Mw)
 
     Mo = torch.where(torch.isnan(Mo), torch.tensor(0.0, device=Mo.device), Mo)
     Mo = torch.where(torch.isinf(Mo), torch.tensor(0.0, device=Mo.device), Mo)
 
     Mg = torch.where(torch.isnan(Mg), torch.tensor(0.0, device=Mg.device), Mg)
     Mg = torch.where(torch.isinf(Mg), torch.tensor(0.0, device=Mg.device), Mg)        
     
 
 
     Mt = torch.add(torch.add(torch.add(Mw,Mo),Mg),Mo*RS)
     
 
     a1 = Mt * a * fault # overall Effective permeability 
     a1water = Mw * a * fault # water Effective permeability 
     a1gas = Mg * a * fault # gas Effective permeability
     a1oil = Mo * a * fault # oil Effective permeability
     
     if pde_method ==1:
         #compute first dffrential for pressure     
         dudx_fdm,dudy_fdm,dudz_fdm = compute_differential(u, dxf)
     
         #Compute second diffrential for pressure         
         dduddx_fdm,dduddy_fdm,dduddz_fdm =  compute_second_differential(u, dxf)
     
         
         
         #compute first dffrential for effective overall permeability
         dcdx,dcdy,dcdz = compute_differential(a1.float(), dxf)        
         
        
     
     
         # Apply the function to all tensors that might contain NaNs
         actnum = replace_nan_with_zero(actnum)
         fin = replace_nan_with_zero(fin)
         dcdx = replace_nan_with_zero(dcdx)
         dudx_fdm = replace_nan_with_zero(dudx_fdm)
         a1 = replace_nan_with_zero(a1)
         dduddx_fdm = replace_nan_with_zero(dduddx_fdm)
         dcdy = replace_nan_with_zero(dcdy)
         dudy_fdm = replace_nan_with_zero(dudy_fdm)
         dduddy_fdm = replace_nan_with_zero(dduddy_fdm)
         
    
    
         right = ((dcdx * dudx_fdm)
         + (a1 * dduddx_fdm)
         + (dcdy * dudy_fdm)
         + (a1 * dduddy_fdm)
         + (dcdz * dudz_fdm)
         + (a1 * dduddz_fdm) )
         #print(right.shape)
         #print(fin.shape)
         right = torch.mul(actnum,right)
         fin = torch.mul(actnum,fin)
         darcy_pressure = torch.sum(torch.abs((fin.reshape(sat.shape[0],-1) - 
                                right.reshape(sat.shape[0],-1)))/sat.shape[0])
                      
    
         # Zero outer boundary
         #darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0) 
         darcy_pressure = dxf * darcy_pressure * 1e-30
         
         p_loss = (darcy_pressure)
    
     
     
         # Water Saturation equation loss
         dudx = dudx_fdm
         dudy = dudy_fdm
         dudz = dudz_fdm
         
         dduddx = dduddx_fdm
         dduddy = dduddy_fdm
         dduddz = dduddz_fdm         
     
      
         
         #compute first diffrential for effective water permeability
         dadx,dady,dadz = compute_differential(a1water.float(), dxf)        
         
     
     
         # Apply the function to all tensors that could possibly have NaNs
         actnum = replace_nan_with_zero(actnum)
         poro = replace_nan_with_zero(poro)
         dsw = replace_nan_with_zero(dsw)
         dtime = replace_nan_with_zero(dtime)
         dadx = replace_nan_with_zero(dadx)
         dudx = replace_nan_with_zero(dudx)
         a1water = replace_nan_with_zero(a1water)
         dduddx = replace_nan_with_zero(dduddx)
         dady = replace_nan_with_zero(dady)
         dudy = replace_nan_with_zero(dudy)
         dduddy = replace_nan_with_zero(dduddy)
         finusew = replace_nan_with_zero(finusew)
         
         # Now, compute darcy_saturation using the sanitized tensors:
    
         
        
         flux = (dadx * dudx) + (a1water * dduddx) + (dady * dudy) + (a1water * dduddy) + (dadz * dudz) + (a1water * dduddz)
         fifth = poro * (dsw/dtime)
         fifth = torch.mul(actnum,fifth)           
         toge = flux + finusew
         toge = torch.mul(actnum,toge)
         darcy_saturation = torch.sum(torch.abs(fifth.reshape(sat.shape[0],-1)- toge.reshape(sat.shape[0],-1))/sat.shape[0])
    
         darcy_saturation = dxf * darcy_saturation * 1e-30
     
         
         s_loss = (darcy_saturation)
    
    
         # Gas Saturation equation loss
         #dudx_fdm
         #dudy_fdm
         Ugx = a1gas * dudx_fdm
         Ugy = a1gas * dudy_fdm
         Ugz = a1gas * dudz_fdm
         
         Uox = a1oil * dudx_fdm *RS
         Uoy = a1oil * dudy_fdm *RS 
         Uoz = a1oil * dudz_fdm *RS 
     
         
         Ubigx = Ugx + Uox
         Ubigy = Ugy + Uoy
         Ubigz = Ugz + Uoz
         
     
         #compute first dffrential
         dubigxdx,dubigxdy,dubigxdz = compute_differential(Ubigx.float(), dxf)         
         dubigydx,dubigydy,dubigydz = compute_differential(Ubigy.float(), dxf)
         dubigzdx,dubigzdy,dubigzdz = compute_differential(Ubigz.float(), dxf)       
             
     
         
         # Using replace_nan_with_zero on tensors that might contain NaNs:
         actnum = replace_nan_with_zero(actnum)
         dubigxdx = replace_nan_with_zero(dubigxdx)
         fingas = replace_nan_with_zero(fingas)
         dubigxdy = replace_nan_with_zero(dubigxdy)
         dubigydx = replace_nan_with_zero(dubigydx)
         dubigydy = replace_nan_with_zero(dubigydy)
         poro = replace_nan_with_zero(poro)
         satg = replace_nan_with_zero(satg)
         BG = replace_nan_with_zero(BG)
         sato = replace_nan_with_zero(sato)
         BO = replace_nan_with_zero(BO)
         RS = replace_nan_with_zero(RS)
         dtime = replace_nan_with_zero(dtime)
         
         # Now compute darcy_saturationg using the sanitized tensors:
    
         darcy_saturationg = torch.mul(actnum,(((dubigxdx - fingas) + \
        (dubigxdy - fingas)+(dubigxdz - fingas)+ (dubigydx - fingas)+ (dubigydy - fingas) + (dubigydz - fingas)+\
        (dubigzdx - fingas)+ (dubigzdy - fingas) + (dubigzdz - fingas)) - \
        (-torch.divide(torch.mul(poro,(torch.add((torch.divide(satg,BG)),\
                        (torch.mul(torch.divide(sato,BO),RS))))),dtime))))
         
         
             
         darcy_saturationg = torch.sum(torch.abs(darcy_saturationg))/sat.shape[0]       
         sg_loss = dxf * darcy_saturationg * 1e-30

         #dxf = 1/1000
     else:
         # compute first dffrential
         gulpa = []
         gulp2a = []
         for m in range(sat.shape[0]):  # Batch
             inn_now = u[m, :, :, :, :]
             gulp = []
             gulp2 = []
             for i in range(nz):
                 now = inn_now[:, i, :, :][:, None, :, :]
                 dudx_fdma = dx1(
                     now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                 )
                 dudy_fdma = dx1(
                     now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                 )
                 gulp.append(dudx_fdma)
                 gulp2.append(dudy_fdma)
             check = torch.stack(gulp, 2)[:, 0, :, :, :]
             check2 = torch.stack(gulp2, 2)[:, 0, :, :]
             gulpa.append(check)
             gulp2a.append(check2)
         dudx_fdm = torch.stack(gulpa, 0)
         dudx_fdm = dudx_fdm.clamp(min=1e-10)  # ensures that all values are at least 1e-10

         dudy_fdm = torch.stack(gulp2a, 0)
         dudy_fdm = dudy_fdm.clamp(min=1e-10)
 
         # Compute second diffrential
         gulpa = []
         gulp2a = []
         for m in range(sat.shape[0]):  # Batch
             inn_now = u[m, :, :, :, :]
             gulp = []
             gulp2 = []
             for i in range(nz):
                 now = inn_now[:, i, :, :][:, None, :, :]
                 dudx_fdma = ddx1(
                     now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                 )
                 dudy_fdma = ddx1(
                     now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                 )
                 gulp.append(dudx_fdma)
                 gulp2.append(dudy_fdma)
             check = torch.stack(gulp, 2)[:, 0, :, :, :]
             check2 = torch.stack(gulp2, 2)[:, 0, :, :]
             gulpa.append(check)
             gulp2a.append(check2)
         dduddx_fdm = torch.stack(gulpa, 0)
         dduddx_fdm = dduddx_fdm.clamp(min=1e-10)
         dduddy_fdm = torch.stack(gulp2a, 0)
         dduddy_fdm = dduddy_fdm.clamp(min=1e-10)
 
         gulp = []
         gulp2 = []
         for i in range(nz):
             inn_now2 = a1.float()[:, :, i, :, :]
             dudx_fdma = dx1(
                 inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
             )
             dudy_fdma = dx1(
                 inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
             )
             gulp.append(dudx_fdma)
             gulp2.append(dudy_fdma)
         dcdx = torch.stack(gulp, 2)
         dcdx = dcdx.clamp(min=1e-10)
         dcdy = torch.stack(gulp2, 2)
         dcdy = dcdy.clamp(min=1e-10)
         
         actnum = replace_nan_with_zero(actnum)
         fin = replace_nan_with_zero(fin)
         dcdx = replace_nan_with_zero(dcdx)
         dudx_fdm = replace_nan_with_zero(dudx_fdm)
         a1 = replace_nan_with_zero(a1)
         dduddx_fdm = replace_nan_with_zero(dduddx_fdm)
         dcdy = replace_nan_with_zero(dcdy)
         dudy_fdm = replace_nan_with_zero(dudy_fdm)
         dduddy_fdm = replace_nan_with_zero(dduddy_fdm)
 
         # Expand dcdx
         # dss = dcdx
         dsout = torch.zeros((sat.shape[0], sat.shape[1], nz, nx, ny)).to(
             device, torch.float32
         )
         for k in range(dcdx.shape[0]):
             see = dcdx[k, :, :, :, :]
             gulp = []
             for i in range(sat.shape[1]):
                 gulp.append(see)
 
             checkken = torch.vstack(gulp)
             dsout[k, :, :, :, :] = checkken
 
         dcdx = dsout
 
         dsout = torch.zeros((sat.shape[0], sat.shape[1], nz, nx, ny)).to(
             device, torch.float32
         )
         for k in range(dcdx.shape[0]):
             see = dcdy[k, :, :, :, :]
             gulp = []
             for i in range(sat.shape[1]):
                 gulp.append(see)
 
             checkken = torch.vstack(gulp)
             dsout[k, :, :, :, :] = checkken
 
         dcdy = dsout
 
         darcy_pressure = (
             fin
             + (dcdx * dudx_fdm)
             + (a1 * dduddx_fdm)
             + (dcdy * dudy_fdm)
             + (a1 * dduddy_fdm)
         )
 
         # Zero outer boundary
         # darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
         #p_loss = dxf * torch.mul(actnum,darcy_pressure) * 1e-10 
         p_loss = torch.mul(actnum,darcy_pressure)
         p_loss = torch.sum(torch.abs(p_loss))/sat.shape[0]
        # p_loss = p_loss.reshape(1, 1)
         p_loss = dxf * (p_loss)  

         # Saruration equation loss
         dudx = dudx_fdm
         dudy = dudy_fdm
 
         dduddx = dduddx_fdm
         dduddy = dduddy_fdm
 
         gulp = []
         gulp2 = []
         for i in range(nz):
             inn_now2 = a1water.float()[:, :, i, :, :]
             dudx_fdma = dx1(
                 inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
             )
             dudy_fdma = dx1(
                 inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
             )
             gulp.append(dudx_fdma)
             gulp2.append(dudy_fdma)
         dadx = torch.stack(gulp, 2)
         dady = torch.stack(gulp2, 2)
         dadx = dadx.clamp(min=1e-10)
         dady = dady.clamp(min=1e-10)
 
         dsout = torch.zeros((sat.shape[0], sat.shape[1], nz, nx, ny)).to(
             device, torch.float32
         )
         for k in range(dadx.shape[0]):
             see = dadx[k, :, :, :, :]
             gulp = []
             for i in range(sat.shape[1]):
                 gulp.append(see)
 
             checkken = torch.vstack(gulp)
             dsout[k, :, :, :, :] = checkken
 
         dadx = dsout
 
         dsout = torch.zeros((sat.shape[0], sat.shape[1], nz, nx, ny)).to(
             device, torch.float32
         )
         for k in range(dady.shape[0]):
             see = dady[k, :, :, :, :]
             gulp = []
             for i in range(sat.shape[1]):
                 gulp.append(see)
 
             checkken = torch.vstack(gulp)
             dsout[k, :, :, :, :] = checkken
 
         dady = dsout
         
         actnum = replace_nan_with_zero(actnum)
         poro = replace_nan_with_zero(poro)
         dsw = replace_nan_with_zero(dsw)
         dtime = replace_nan_with_zero(dtime)
         dadx = replace_nan_with_zero(dadx)
         dudx = replace_nan_with_zero(dudx)
         a1water = replace_nan_with_zero(a1water)
         dduddx = replace_nan_with_zero(dduddx)
         dady = replace_nan_with_zero(dady)
         dudy = replace_nan_with_zero(dudy)
         dduddy = replace_nan_with_zero(dduddy)
         finusew = replace_nan_with_zero(finusew)
 
         flux = (dadx * dudx) + (a1water * dduddx) + (dady * dudy) + (a1water * dduddy)
         fifth = poro * (dsw / dtime)
         toge = flux + finusew
         darcy_saturation = fifth - toge
 
         # Zero outer boundary
         # darcy_saturation = F.pad(darcy_saturation[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
         s_loss = torch.mul(actnum,darcy_saturation)
         s_loss = torch.sum(torch.abs(s_loss))/sat.shape[0]
         #s_loss = s_loss.reshape(1, 1)
         s_loss = dxf * (s_loss)             
             
         
         #Gas Saturation
         Ugx = a1gas * dudx_fdm
         Ugy = a1gas * dudy_fdm

         
         Uox = a1oil * dudx_fdm *RS
         Uoy = a1oil * dudy_fdm *RS 

     
         
         Ubigx = Ugx + Uox
         Ubigy = Ugy + Uoy
         
         Ubigx = Ubigx.clamp(min=1e-10)
         Ubigy = Ubigy.clamp(min=1e-10)
         

         # compute first dffrential
         gulpa = []
         gulp2a = []
         for m in range(sat.shape[0]):  # Batch
             inn_now = Ubigx.float()[m, :, :, :, :]
             gulp = []
             gulp2 = []
             for i in range(nz):
                 now = inn_now[:, i, :, :][:, None, :, :]
                 dudx_fdma = dx1(
                     now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                 )
                 dudy_fdma = dx1(
                     now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                 )
                 gulp.append(dudx_fdma)
                 gulp2.append(dudy_fdma)
             check = torch.stack(gulp, 2)[:, 0, :, :, :]
             check2 = torch.stack(gulp2, 2)[:, 0, :, :]
             gulpa.append(check)
             gulp2a.append(check2)
         dubigxdx = torch.stack(gulpa, 0)
         dubigxdx = dubigxdx.clamp(min=1e-10)
         dubigxdy = torch.stack(gulp2a, 0) 
         dubigxdy = dubigxdy.clamp(min=1e-10)
     
     
         #compute first dffrential
         gulpa = []
         gulp2a = []
         for m in range(sat.shape[0]):  # Batch
             inn_now = Ubigy.float()[m, :, :, :, :]
             gulp = []
             gulp2 = []
             for i in range(nz):
                 now = inn_now[:, i, :, :][:, None, :, :]
                 dudx_fdma = dx1(
                     now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                 )
                 dudy_fdma = dx1(
                     now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                 )
                 gulp.append(dudx_fdma)
                 gulp2.append(dudy_fdma)
             check = torch.stack(gulp, 2)[:, 0, :, :, :]
             check2 = torch.stack(gulp2, 2)[:, 0, :, :]
             gulpa.append(check)
             gulp2a.append(check2)
         dubigydx = torch.stack(gulpa, 0)
         dubigydx = dubigydx.clamp(min=1e-10)
         dubigydy = torch.stack(gulp2a, 0)
         dubigydy = dubigydy.clamp(min=1e-10)

         actnum = replace_nan_with_zero(actnum)
         dubigxdx = replace_nan_with_zero(dubigxdx)
         fingas = replace_nan_with_zero(fingas)
         dubigxdy = replace_nan_with_zero(dubigxdy)
         dubigydx = replace_nan_with_zero(dubigydx)
         dubigydy = replace_nan_with_zero(dubigydy)
         poro = replace_nan_with_zero(poro)
         satg = replace_nan_with_zero(satg)
         BG = replace_nan_with_zero(BG)
         sato = replace_nan_with_zero(sato)
         BO = replace_nan_with_zero(BO)
         RS = replace_nan_with_zero(RS)
         dtime = replace_nan_with_zero(dtime)

          
       
         left = ((dubigxdx + dubigydx)-fingas) + ((dubigxdy + dubigydy)-fingas)
         right  = -(torch.divide(torch.mul(poro, (torch.divide(satg, BG) + torch.mul(torch.divide(sato, BO), RS))), dtime))
         sg_loss = left - right
         sg_loss = torch.mul(actnum, (left - right))
             
         sg_loss = torch.sum(torch.abs(sg_loss))/sat.shape[0]
         #sg_loss = sg_loss.reshape(1, 1)
         sg_loss = dxf * (sg_loss) 
     #sg_loss = (sg_loss)
     #sg_loss = ((sg_loss) ** 2).mean()
 
     
     p_loss = torch.where(torch.isnan(p_loss), torch.tensor(0, device=p_loss.device), p_loss)
     p_loss = torch.where(torch.isinf(p_loss), torch.tensor(0, device=p_loss.device), p_loss)
 
 
     s_loss = torch.where(torch.isnan(s_loss), torch.tensor(0, device=s_loss.device), s_loss)
     s_loss = torch.where(torch.isinf(s_loss), torch.tensor(0, device=s_loss.device), s_loss)
     

 
 
     sg_loss = torch.where(torch.isnan(sg_loss), torch.tensor(0, device=sg_loss.device), sg_loss)
     sg_loss = torch.where(torch.isinf(sg_loss), torch.tensor(0, device=sg_loss.device), sg_loss)

 
     
     output_var = {"pressured": p_loss,"saturationd": s_loss,"saturationdg": sg_loss}
     
     for key, tensor in output_var.items():
         tensor_clone = tensor.clone()
         replacement_tensor = torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
         output_var[key] = torch.where(torch.isnan(tensor_clone), replacement_tensor, tensor_clone)       
    
     return output_var["pressured"],output_var["saturationd"],output_var["saturationdg"]


class Labelledset():
    def __init__(self,datacc):
        self.data1 = torch.from_numpy(datacc['perm'])
        self.data5 = torch.from_numpy(datacc['Phi'])
        self.data7 = torch.from_numpy(datacc['Pini'])
        self.data8 = torch.from_numpy(datacc['Swini'])
        self.data9 = torch.from_numpy(datacc['pressure'])
        self.data10 =  torch.from_numpy(datacc['water_sat'])
        self.data11 =  torch.from_numpy(datacc['gas_sat'])
        self.data12 =  torch.from_numpy(datacc['fault'])

        
    def __getitem__(self, index):
        x1 = self.data1[index]
        x5 = self.data5[index]
        x7 = self.data7[index]
        x8 = self.data8[index]
        x9 = self.data9[index]
        x10 = self.data10[index]
        x11 = self.data11[index]
        x12 = self.data12[index]


        cuda = 0
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        
        return {'perm':x1.to(device,torch.float32),'Phi': x5.to(device,torch.float32),'Pini': x7.to(device,torch.float32),\
        'Swini': x8.to(device,torch.float32)\
            ,'pressure': x9.to(device,torch.float32),'water_sat': x10.to(device,torch.float32)\
        ,'gas_sat': x11.to(device,torch.float32),'fault': x12.to(device,torch.float32)}
    
    def __len__(self):
        return len(self.data1)
    

class LabelledsetP():
    def __init__(self,datacc):
        self.data1 = torch.from_numpy(datacc['X'])
        self.data2 = torch.from_numpy(datacc['Y'])


        
    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]

        cuda = 0
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        
        return {'X':x1.to(device,torch.float32),'Y': x2.to(device,torch.float32)}
    
    def __len__(self):
        return len(self.data1)
    
    
@modulus.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    text = """
                                                              dddddddd                                                         
    MMMMMMMM               MMMMMMMM                           d::::::d                lllllll                                  
    M:::::::M             M:::::::M                           d::::::d                l:::::l                                  
    M::::::::M           M::::::::M                           d::::::d                l:::::l                                  
    M:::::::::M         M:::::::::M                           d:::::d                 l:::::l                                  
    M::::::::::M       M::::::::::M  ooooooooooo      ddddddddd:::::duuuuuu    uuuuuu  l::::luuuuuu    uuuuuu     ssssssssss   
    M:::::::::::M     M:::::::::::Moo:::::::::::oo  dd::::::::::::::du::::u    u::::u  l::::lu::::u    u::::u   ss::::::::::s  
    M:::::::M::::M   M::::M:::::::o:::::::::::::::od::::::::::::::::du::::u    u::::u  l::::lu::::u    u::::u ss:::::::::::::s 
    M::::::M M::::M M::::M M::::::o:::::ooooo:::::d:::::::ddddd:::::du::::u    u::::u  l::::lu::::u    u::::u s::::::ssss:::::s
    M::::::M  M::::M::::M  M::::::o::::o     o::::d::::::d    d:::::du::::u    u::::u  l::::lu::::u    u::::u  s:::::s  ssssss 
    M::::::M   M:::::::M   M::::::o::::o     o::::d:::::d     d:::::du::::u    u::::u  l::::lu::::u    u::::u    s::::::s      
    M::::::M    M:::::M    M::::::o::::o     o::::d:::::d     d:::::du::::u    u::::u  l::::lu::::u    u::::u       s::::::s   
    M::::::M     MMMMM     M::::::o::::o     o::::d:::::d     d:::::du:::::uuuu:::::u  l::::lu:::::uuuu:::::u ssssss   s:::::s 
    M::::::M               M::::::o:::::ooooo:::::d::::::ddddd::::::du:::::::::::::::ul::::::u:::::::::::::::us:::::ssss::::::s
    M::::::M               M::::::o:::::::::::::::od:::::::::::::::::du:::::::::::::::l::::::lu:::::::::::::::s::::::::::::::s 
    M::::::M               M::::::Moo:::::::::::oo  d:::::::::ddd::::d uu::::::::uu:::l::::::l uu::::::::uu:::us:::::::::::ss  
    MMMMMMMM               MMMMMMMM  ooooooooooo     ddddddddd   ddddd   uuuuuuuu  uuullllllll   uuuuuuuu  uuuu sssssssssss   
    """   
    print(text)  
    print('')
    print('------------------------------------------------------------------')    
    print('')
    print ('\n')
    print ('|-----------------------------------------------------------------|')
    print ('|                 TRAIN THE MODEL USING A 3D PINO APPROACH:        |')
    print ('|-----------------------------------------------------------------|')
    print('')
    
    
    oldfolder = os.getcwd()
    os.chdir(oldfolder)


    
    interest = None
    while True:
        interest = int(input('Select 1 = Run the Flow simulation to generate samples | 2 = Use saved data \n'))
        if (interest>2) or (interest<1):
            #raise SyntaxError('please select value between 1-2')
            print('')
            print('please try again and select value between 1-2')
        else:
            
            break
        
    print('')
    Relperm = None
    while True:
        Relperm = int(input('Select 1 = Interpolation- Correy | 2 = Stone II \n'))
        if (Relperm>2) or (Relperm<1):
            #raise SyntaxError('please select value between 1-2')
            print('')
            print('please try again and select value between 1-2')
        else:
            
            break
        
    if Relperm == 2:  
        print('Selected Stone II method for Relative permeability computation')
        # Parameters for Stone11 method
        params = {
            'k_rwmax': torch.tensor(0.3),
            'k_romax': torch.tensor(0.9),
            'k_rgmax': torch.tensor(0.8),
            'n': torch.tensor(2.0),
            'p': torch.tensor(2.0),
            'q': torch.tensor(2.0),
            'm': torch.tensor(2.0),
            'Swi': torch.tensor(0.1),
            'Sor': torch.tensor(0.2),
        }
        

    
    print('')        
    pde_method = None
    while True:
        pde_method = int(input('Select 1 = approximate | 2 = Extensive \n'))
        if (pde_method>2) or (pde_method<1):
            #raise SyntaxError('please select value between 1-2')
            print('')
            print('please try again and select value between 1-2')
        else:
            
            break  
    
    if not os.path.exists(to_absolute_path('../PACKETS')):
        os.makedirs(to_absolute_path('../PACKETS'))
    else:
        pass
    
    if interest ==1:
        if not os.path.exists(to_absolute_path('../RUNS')):
            os.makedirs(to_absolute_path('../RUNS'))
        else:
            pass

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU found. Please run on a system with a GPU.")        

    # Varaibles needed for NVRS


    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz
 
    # training

    pini_alt = 600
    
    
    bb = os.path.isfile(to_absolute_path('../PACKETS/conversions.mat'))
    if (bb==True): 
        mat = sio.loadmat(to_absolute_path("../PACKETS/conversions.mat"))
        steppi = int(mat["steppi"])
        steppi_indices = mat["steppi_indices"].flatten()
        N_ens = int(mat["N_ens"])
        #print(N_ens)
        #print(steppi)
    else:
        steppi = None
        while True:
            steppi = int(input('input the time step for training between 1-246 (Hint: Use fewer time step)\n'))
            if (steppi>246) or (steppi<1):
                #raise SyntaxError('please select value between 1-2')
                print('')
                print('please try again and select value between 1-246')
            else:
                
                break
        steppi_indices = np.linspace(1, 246, steppi, dtype=int)

        N_ens = None
        while True:
            N_ens = int(input('Enter the ensemble size between 2-100\n'))
            if (N_ens>246) or (N_ens<1):
                #raise SyntaxError('please select value between 1-2')
                print('')
                print('please try again and select value between 2-100')
            else:
                
                break        
        
    print(steppi)    
    #steppi = 246 
    """
    input_channel = 5 #[Perm,Phi,initial_pressure, initial_water_sat,FTM] 
    output_channel = 3 #[Pressure, Sw,Sg]
    """
    

    #os.chdir(to_absolute_path('..'))
    #os.chdir(oldfolder)
    oldfolder2 = os.getcwd()
    effective = np.genfromtxt(to_absolute_path("../NORNE/actnum.out"), dtype='float')

    check = np.ones((nx,ny,nz),dtype=np.float32)
    #fname = 'NORNE/hilda.yaml'
    #plan = read_yaml(fname)
    
    SWOW = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOW),dtype=float)).to(device)
    
    SWOG = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOG),dtype=float)).to(device)
    
    injectors = cfg.custom.WELLSPECS.water_injector_wells
    producers = cfg.custom.WELLSPECS.producer_wells
    gass = cfg.custom.WELLSPECS.gas_injector_wells
    
    N_injw = len(cfg.custom.WELLSPECS.water_injector_wells)  # Number of water injectors
    N_pr = len(cfg.custom.WELLSPECS.producer_wells)  #Number of producers
    N_injg = len(cfg.custom.WELLSPECS.gas_injector_wells)  # Number of gas injectors
    
    
    BO = float(cfg.custom.PROPS.BO)
    BW = float(cfg.custom.PROPS.BW)
    UW = float(cfg.custom.PROPS.UW)
    UO = float(cfg.custom.PROPS.UO)
    SWI = cp.float32(cfg.custom.PROPS.SWI)
    SWR = cp.float32(cfg.custom.PROPS.SWR)
    CFO = cp.float32(cfg.custom.PROPS.CFO) 
    p_atm = cp.float32(float(cfg.custom.PROPS.PATM))
    p_bub = cp.float32(float(cfg.custom.PROPS.PB))

    params1 = {
        'BO': torch.tensor(BO),
        'UO': torch.tensor(UO),
        'BW': torch.tensor(BW),
        'UW': torch.tensor(UW),
    }    

    skin = torch.tensor(0).to(device)
    rwell = torch.tensor(200).to(device)
    pwf_producer = torch.tensor(100).to(device)
    RE = torch.tensor(0.2 * 100).to(device)
    #cgrid = np.genfromtxt("NORNE/clementgrid.out", dtype='float')
    
    Truee1 = np.genfromtxt(to_absolute_path("../NORNE/rossmary.GRDECL"), dtype='float')
    
    Trueea = np.reshape(Truee1.T,(nx,ny,nz),'F')
    Trueea = np.reshape(Trueea,(-1,1),'F')
    Trueea = Trueea * effective.reshape(-1,1)
    
    string_Jesus ='flow FULLNORNE.DATA --parsing-strictness=low'
    string_Jesus2 ='flow FULLNORNE2.DATA --parsing-strictness=low'
    
    #N_ens = 2
    njobs = 12
    #os.chdir('NORNE')

    
    source_dir = to_absolute_path('../Necessaryy')
    #dest_dir = 'path_to_folder_B'

    perm_ensemble = np.genfromtxt(to_absolute_path('../NORNE/sgsim.out'))
    poro_ensemble = np.genfromtxt(to_absolute_path('../NORNE/sgsimporo.out'))
    fault_ensemble = np.genfromtxt(to_absolute_path('../NORNE/faultensemble.dat'))
    
    perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
    poro_ensemble = clip_and_convert_to_float32(poro_ensemble)
    fault_ensemble = clip_and_convert_to_float32(fault_ensemble)
    effective = clip_and_convert_to_float32(effective)

    if interest == 1:    
        for kk in range(N_ens):
            path_out = to_absolute_path('../RUNS/Realisation' + str(kk))
            os.makedirs(path_out, exist_ok=True) 
                
        executor_copy = get_reusable_executor(max_workers=njobs)
        with executor_copy:
            Parallel(n_jobs=njobs,backend='multiprocessing', verbose=50)(delayed(
            copy_files)(source_dir,to_absolute_path('../RUNS/Realisation' + str(kk)))for kk in range(N_ens) )
            executor_copy.shutdown(wait=False)
            
        executor_save = get_reusable_executor(max_workers=njobs)
        with executor_save:
            Parallel(n_jobs=njobs,backend='multiprocessing', verbose=50)(delayed(
            save_files)(perm_ensemble[:,kk],poro_ensemble[:,kk],fault_ensemble[:,kk],\
                to_absolute_path('../RUNS/Realisation' + str(kk)),
            oldfolder)for kk in range(N_ens) )
            executor_save.shutdown(wait=False)    
        
        print('')
        print('---------------------------------------------------------------------')    
        print('')
        print ('\n')
        print ('|-----------------------------------------------------------------|')
        print ('|                 RUN FLOW SIMULATOR FOR ENSEMBLE                  |')
        print ('|-----------------------------------------------------------------|')
        print('')
               
        executor_run = get_reusable_executor(max_workers=njobs)
        with executor_run:
            Parallel(n_jobs=njobs,backend='multiprocessing', verbose=50)(delayed(
            Run_simulator)\
            (to_absolute_path('../RUNS/Realisation' + str(kk)),\
            oldfolder2,string_Jesus,string_Jesus2)for kk in range(N_ens) )
            executor_run.shutdown(wait=False) 
        
        print ('|-----------------------------------------------------------------|')
        print ('|                 EXECUTED RUN of  FLOW SIMULATION FOR ENSEMBLE   |')
        print ('|-----------------------------------------------------------------|')    
            
        
        print ('|-----------------------------------------------------------------|')
        print ('|                 DATA CURRATION IN PROCESS                       |')
        print ('|-----------------------------------------------------------------|')
        N = N_ens
        pressure =[]
        Sgas = []
        Swater = []
        Fault =[]
        Time = []
        
        permeability = np.zeros((N,1,nx,ny,nz))
        porosity = np.zeros((N,1,nx,ny,nz))
        actnumm = np.zeros((N,1,nx,ny,nz))
        for i in range(N):
            folder = to_absolute_path('../RUNS/Realisation' + str(i))
            Pr,sw,sg,tt,flt = Geta_all(folder,nx,ny,nz,effective,oldfolder,check,string_Jesus,steppi,steppi_indices)

            Pr = round_array_to_4dp(clip_and_convert_to_float32(Pr))
            sw = round_array_to_4dp(clip_and_convert_to_float32(sw))
            sg = round_array_to_4dp(clip_and_convert_to_float32(sg))
            tt = round_array_to_4dp(clip_and_convert_to_float32(tt))
            flt = round_array_to_4dp(clip_and_convert_to_float32(flt))
            
            pressure.append(Pr)
            Sgas.append(sg)
            Swater.append(sw)
            Fault.append(flt)
            Time.append(tt)
            
            permeability[i,0,:,:,:] = np.reshape(perm_ensemble[:,i],(nx,ny,nz),'F')
            porosity[i,0,:,:,:] =  np.reshape(poro_ensemble[:,i],(nx,ny,nz),'F') 
            actnumm[i,0,:,:,:] = np.reshape(effective,(nx,ny,nz),'F')
            
            del Pr
            gc.collect()
            del sw
            gc.collect()
            del sg
            gc.collect()
            del tt
            gc.collect()
            del flt
            gc.collect()
            
            
        pressure = np.stack(pressure, axis=0)
        Sgas = np.stack(Sgas, axis=0)
        Swater = np.stack(Swater, axis=0)
        Fault = np.stack(Fault, axis=0)[:,None,:,:,:]
        Time = np.stack(Time, axis=0)
        ini_pressure = pini_alt *np.ones((N,1,nx,ny,nz),dtype=np.float32) 
        ini_sat = 0.2 * np.ones((N,1,nx,ny,nz),dtype=np.float32)
        
        
        inn_fcn,out_fcn = Get_data_FFNN(oldfolder,N,pressure,Sgas,Swater,permeability,Time,steppi,steppi_indices)
        inn_fcn[np.isnan(inn_fcn)] = 0.
        out_fcn[np.isnan(out_fcn)] = 0.  
        
        #X_data2 = {"x": inn_fcn, "y":out_fcn}
        
        # Read the first and second sheets, skip the header
        data1 = pd.read_excel(to_absolute_path('../Necessaryy/Book1.xlsx'), sheet_name=0, header=None)  # Reads the first sheet
        data2 = pd.read_excel(to_absolute_path('../Necessaryy/Book1.xlsx'), sheet_name=1, header=None)  # Reads the second sheet
        waterz = np.nan_to_num(clip_and_convert_to_float32(data1.values[1:,]), nan=0)
        gasz = np.nan_to_num(clip_and_convert_to_float32(data2.values[1:,]), nan=0)
        
        Qw, Qg, Qo  = Get_source_sink(N,nx,ny,nz,waterz,gasz,steppi,steppi_indices)
        Q = Qw + Qg + Qo
        
        print ('|-----------------------------------------------------------------|')
        print ('|                 DATA CURRATED                                   |')
        print ('|-----------------------------------------------------------------|')    
    
    
    
        target_min = 0.01
        target_max = 1.
        
        permeability[np.isnan(permeability)] = 0. 
        Time[np.isnan(Time)] = 0.
        pressure[np.isnan(pressure)] = 0.
        Qw[np.isnan(Qw)] = 0.
        Qg[np.isnan(Qg)] = 0.
        Q[np.isnan(Q)] = 0.
        
        permeability[np.isinf(permeability)] = 0. 
        Time[np.isinf(Time)] = 0.
        pressure[np.isinf(pressure)] = 0.
        Qw[np.isinf(Qw)] = 0.
        Qg[np.isinf(Qg)] = 0.
        Q[np.isinf(Q)] = 0.        
                                  
        minK,maxK,permeabilityx = scale_clement(permeability,target_min,target_max) #Permeability
        minT,maxT,Timex = scale_clement(Time,target_min,target_max) # Time
        minP,maxP,pressurex = scale_clement(pressure,target_min,target_max) #pressure
        minQw,maxQw,Qwx = scale_clement(Qw,target_min,target_max)# Qw
        minQg,maxQg,Qgx = scale_clement(Qg,target_min,target_max) #Qg
        minQ,maxQ,Qx = scale_clement(Q,target_min,target_max) #Q
        
        permeabilityx[np.isnan(permeabilityx)] = 0. 
        Timex[np.isnan(Timex)] = 0.
        pressurex[np.isnan(pressurex)] = 0.
        Qwx[np.isnan(Qwx)] = 0.
        Qgx[np.isnan(Qgx)] = 0.
        Qx[np.isnan(Qx)] = 0.
        

        permeabilityx[np.isinf(permeabilityx)] = 0. 
        Timex[np.isinf(Timex)] = 0.
        pressurex[np.isinf(pressurex)] = 0.
        Qwx[np.isinf(Qwx)] = 0.
        Qgx[np.isinf(Qgx)] = 0.
        Qx[np.isinf(Qx)] = 0.        
        
        
        ini_pressure[np.isnan(ini_pressure)] = 0.    
        ini_pressurex = ini_pressure /maxP 
        
        ini_pressurex = clip_and_convert_to_float32(ini_pressurex)
         
        ini_pressurex[np.isnan(ini_pressurex)] = 0.
        porosity[np.isnan(porosity)] = 0.  
        Fault[np.isnan(Fault)] = 0. 
        Swater[np.isnan(Swater)] = 0.  
        Sgas[np.isnan(Sgas)] = 0. 
        actnumm[np.isnan(actnumm)] = 0.  
        ini_sat[np.isnan(ini_sat)] = 0. 
        
        
        ini_pressurex[np.isinf(ini_pressurex)] = 0.
        porosity[np.isinf(porosity)] = 0.  
        Fault[np.isinf(Fault)] = 0. 
        Swater[np.isinf(Swater)] = 0.  
        Sgas[np.isinf(Sgas)] = 0. 
        actnumm[np.isinf(actnumm)] = 0.  
        ini_sat[np.isinf(ini_sat)] = 0.         
          
        X_data1 = {'permeability':permeabilityx,\
                'porosity':porosity,'Pressure': pressurex,\
                'Fault': Fault,\
               'Water_saturation':Swater,\
                 'Time': Timex,\
                'Gas_saturation': Sgas,\
                'actnum':actnumm,\
                'Pini':ini_pressurex,\
                'Qw':Qwx,\
                'Qg':Qgx,\
                'Q':Qx,\
                'Sini':ini_sat}
            
        X_data1 = clean_dict_arrays(X_data1)
            
        # for key in X_data1.keys():
        #     X_data1[key][np.isnan(X_data1[key])] = 0           # Convert NaN to 0
        #     X_data1[key][np.isinf(X_data1[key])] = 0           # Convert infinity to 0
        #     #X_data1[key] = np.clip(X_data1[key], target_min, target_max)

                
        del permeabilityx
        gc.collect()
        del permeability
        gc.collect()
        del porosity
        gc.collect()
        del pressurex
        gc.collect()
        del Fault
        gc.collect()
        del Swater
        gc.collect()
        del Timex
        gc.collect()
        del Sgas
        gc.collect()
        del actnumm
        gc.collect()
        del ini_pressurex
        gc.collect()
        del Qwx
        gc.collect()
        del Qgx
        gc.collect()
        del Qx
        gc.collect()
        del ini_sat
        gc.collect()
        
        del pressure
        gc.collect()
        del Time
        gc.collect()
        del ini_pressure
        gc.collect()  
        
         
            
        # Save compressed dictionary
        with gzip.open(to_absolute_path('../PACKETS/data_train.pkl.gz'), 'wb') as f1:
            pickle.dump(X_data1, f1)
            
        with gzip.open(to_absolute_path('../PACKETS/data_test.pkl.gz'), 'wb') as f2:
            pickle.dump(X_data1, f2)
        
        min_inn_fcn,max_inn_fcn,inn_fcnx = scale_clement(inn_fcn,target_min,target_max)
        min_out_fcn,max_out_fcn,out_fcnx = scale_clement(out_fcn,target_min,target_max)
          
        inn_fcnx = clip_and_convert_to_float32(inn_fcnx)
        out_fcnx = clip_and_convert_to_float32(out_fcnx)
        
        X_data2 = {'X':inn_fcnx,'Y':out_fcnx}
        for key in X_data2.keys():
            X_data2[key][np.isnan(X_data2[key])] = 0.           # Convert NaN to 0
            X_data2[key][np.isinf(X_data2[key])] = 0.           # Convert infinity to 0
            #X_data2[key] = np.clip(X_data2[key], target_min, target_max)
            
            
        del inn_fcnx
        gc.collect()
        del inn_fcn
        gc.collect()
        del out_fcnx
        gc.collect()
        del out_fcn
        gc.collect()
        
        with gzip.open(to_absolute_path('../PACKETS/data_train_peaceman.pkl.gz'), 'wb') as f3:
            pickle.dump(X_data2, f3)
    
        with gzip.open(to_absolute_path('../PACKETS/data_test_peaceman.pkl.gz'), 'wb') as f4:
            pickle.dump(X_data2, f4)
        
        sio.savemat(to_absolute_path('../PACKETS/conversions.mat'), {'minK':minK,'maxK':maxK,'minT':minT,'maxT':maxT,\
    'minP':minP,'maxP':maxP,'minQW':minQw,'maxQW':maxQw,'minQg':minQg,'maxQg':maxQg,\
    'minQ':minQ,'maxQ':maxQ,'min_inn_fcn':min_inn_fcn,'max_inn_fcn':max_inn_fcn,\
    'min_out_fcn':min_out_fcn,'max_out_fcn':max_out_fcn,'steppi':steppi,\
    'steppi_indices':steppi_indices,'N_ens':N_ens},do_compression=True)

        print ('|-----------------------------------------------------------------|')
        print ('|                 DATA SAVED                                      |')
        print ('|-----------------------------------------------------------------|')    
        
        print ('|-----------------------------------------------------------------|')
        print ('|                 REMOVE FOLDERS USED FOR THE RUN                 |')
        print ('|-----------------------------------------------------------------|')

        for jj in range(N_ens):
            folderr= to_absolute_path('../RUNS/Realisation' + str(jj))
            rmtree(folderr)
        rmtree(to_absolute_path('../RUNS')) 
    else:
        pass
                
    cuda = 0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu") 
    
    SWI = torch.from_numpy(np.array(SWI)).to(device)
    SWR = torch.from_numpy(np.array(SWR)).to(device)
    UW = torch.from_numpy(np.array(UW)).to(device)
    BW = torch.from_numpy(np.array(BW)).to(device)
    UO = torch.from_numpy(np.array(UO)).to(device)
    BO = torch.from_numpy(np.array(BO)).to(device)
    
    # SWOW = torch.from_numpy(SWOW).to(device)
    # SWOG = torch.from_numpy(SWOG).to(device)
    p_bub = torch.from_numpy(np.array(p_bub)).to(device)
    p_atm = torch.from_numpy(np.array(p_atm)).to(device)
    CFO = torch.from_numpy(np.array(CFO)).to(device)    


    mat = sio.loadmat(to_absolute_path("../PACKETS/conversions.mat"))
    minK = mat['minK']
    maxK = mat['maxK']
    minT = mat['minT']
    maxT = mat['maxT']
    minP = mat['minP']
    maxP = mat['maxP']
    minQw =mat['minQW']
    maxQw = mat['maxQW']
    minQg = mat['minQg']
    maxQg = mat['maxQg']
    minQ =mat['minQ']
    maxQ = mat['maxQ']
    min_inn_fcn = mat['min_inn_fcn']
    max_inn_fcn = mat['max_inn_fcn']
    min_out_fcn = mat['min_out_fcn']
    max_out_fcn = mat['max_out_fcn']
    
    target_min = 0.01
    target_max = 1
    print("These are the values:")
    print("minK value is:", minK)
    print("maxK value is:", maxK)
    print("minT value is:", minT)
    print("maxT value is:", maxT)
    print("minP value is:", minP)
    print("maxP value is:", maxP)
    print("minQw value is:", minQw)
    print("maxQw value is:", maxQw)
    print("minQg value is:", minQg)
    print("maxQg value is:", maxQg)
    print("minQ value is:", minQ)
    print("maxQ value is:", maxQ)
    print("min_inn_fcn value is:", min_inn_fcn)
    print("max_inn_fcn value is:", max_inn_fcn)
    print("min_out_fcn value is:", min_out_fcn)
    print("max_out_fcn value is:", max_out_fcn)
    print("target_min value is:", target_min)
    print("target_max value is:", target_max)        
          
    minKx = torch.from_numpy(minK).to(device)
    maxKx = torch.from_numpy(maxK).to(device)
    minTx = torch.from_numpy(minT).to(device)
    maxTx = torch.from_numpy(maxT).to(device)
    minPx = torch.from_numpy(minP).to(device)
    maxPx = torch.from_numpy(maxP).to(device)
    minQx = torch.from_numpy(minQ).to(device)
    maxQx = torch.from_numpy(maxQ).to(device)
    minQgx = torch.from_numpy(minQg).to(device)
    maxQgx = torch.from_numpy(maxQg).to(device)
    minQwx = torch.from_numpy(minQw).to(device)
    maxQwx = torch.from_numpy(maxQw).to(device)
    min_inn_fcnx = torch.from_numpy(min_inn_fcn).to(device)
    max_inn_fcnx = torch.from_numpy(max_inn_fcn).to(device)
    min_out_fcnx = torch.from_numpy(min_out_fcn).to(device)
    max_out_fcnx = torch.from_numpy(max_out_fcn).to(device)       
 
    del mat
    gc.collect()       

    

    print('Load simulated labelled training data')
    with gzip.open(to_absolute_path('../PACKETS/data_train.pkl.gz'), 'rb') as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    for key, value in X_data1.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())

    del mat
    gc.collect()
    
    
    for key in  X_data1.keys():
        # Convert NaN and infinity values to 0
        X_data1[key][np.isnan( X_data1[key])] = 0.
        X_data1[key][np.isinf( X_data1[key])] = 0. 
        #X_data1[key] = np.clip(X_data1[key], target_min, target_max)
        X_data1[key] = clip_and_convert_to_float32(X_data1[key])

    
    cPerm = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) #Permeability
    cPhi = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32)#Porosity
    cPini = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) #Initial pressure
    cSini = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) # Initial water saturation
    cfault = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) # Fault
    cQ = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cQw = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cQg = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cTime = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cactnum = np.zeros((1,1,nz,nx,ny),dtype=np.float32) # Fault

    
    cPress = np.zeros((N_ens,steppi,nz,nx,ny),dtype=np.float32) # Pressure
    cSat = np.zeros((N_ens,steppi,nz,nx,ny),dtype=np.float32) # Water saturation
    cSatg = np.zeros((N_ens,steppi,nz,nx,ny),dtype=np.float32) # gas saturation
    
    #print(X_data1['Q'].shape)
    X_data1['Q'][X_data1['Q']<=0] = 0
    X_data1['Qw'][X_data1['Qw']<=0] = 0
    X_data1['Qg'][X_data1['Qg']<=0] = 0    

    for i in range(nz):
        X_data1['Q'][0,:,:,:,i] = np.where(X_data1['Q'][0,:,:,:,i] < 0, 0, X_data1['Q'][0,:,:,:,i])
        X_data1['Qw'][0,:,:,:,i] = np.where(X_data1['Qw'][0,:,:,:,i] < 0, 0, X_data1['Qw'][0,:,:,:,i])
        X_data1['Qg'][0,:,:,:,i] = np.where(X_data1['Qg'][0,:,:,:,i] < 0, 0, X_data1['Qg'][0,:,:,:,i])        
        cQ[0,:,i,:,:] = X_data1['Q'][0,:,:,:,i]
        cQw[0,:,i,:,:] = X_data1['Qw'][0,:,:,:,i] 
        cQg[0,:,i,:,:] = X_data1['Qg'][0,:,:,:,i]
        cactnum[0,0,i,:,:] = X_data1['actnum'][0,0,:,:,i]
        cTime[0,:,i,:,:] = X_data1['Time'][0,:,:,:,i] 
    
    neededM = {'Q': torch.from_numpy(cQ).to(device,torch.float32),\
    'Qw': torch.from_numpy(cQw).to(device,dtype=torch.float32),\
    'Qg': torch.from_numpy(cQg).to(device,dtype=torch.float32),\
     'actnum': torch.from_numpy(cactnum).to(device,dtype=torch.float32),\
     'Time': torch.from_numpy(cTime).to(device,dtype=torch.float32)} 
        
    for key in neededM:
        neededM[key] = replace_nans_and_infs(neededM[key])
    
    
    for kk in range(N_ens):
        
        #INPUTS        
        for i in range(nz):
            cPerm[kk,0,i,:,:] = clip_and_convert_to_float3(X_data1['permeability'][kk,0,:,:,i])
            cfault[kk,0,i,:,:] = clip_and_convert_to_float3(X_data1['Fault'][kk,0,:,:,i])            
            cPhi[kk,0,i,:,:] = clip_and_convert_to_float3(X_data1['porosity'][kk,0,:,:,i])
            cPini[kk,0,i,:,:] = clip_and_convert_to_float3(X_data1['Pini'][kk,0,:,:,i])/maxP
            cSini[kk,0,i,:,:] = clip_and_convert_to_float3(X_data1['Sini'][kk,0,:,:,i])
                

        #OUTPUTS
        for mum in range(steppi):
            for i in range(nz):
                cPress[kk,mum,i,:,:] = clip_and_convert_to_float3(X_data1['Pressure'][kk,mum,:,:,i])
                cSat[kk,mum,i,:,:] = clip_and_convert_to_float3(X_data1['Water_saturation'][kk,mum,:,:,i])
                cSatg[kk,mum,i,:,:] = clip_and_convert_to_float3(X_data1['Gas_saturation'][kk,mum,:,:,i])
                   
    
    del X_data1
    gc.collect()
    data = {'perm': cPerm,'Phi': cPhi,'Pini': cPini,'Swini': cSini,'pressure': cPress,'water_sat': cSat,'gas_sat': cSatg,'fault':cfault}
      
    del cPerm
    gc.collect()
    del cQ
    gc.collect()
    del cQw
    gc.collect()
    del cQg
    gc.collect()
    del cPhi
    gc.collect()
    del cTime
    gc.collect()
    del cPini
    gc.collect()
    del cSini
    gc.collect()
    del cPress
    gc.collect()
    del cSat
    gc.collect()
    del cSatg
    gc.collect()
    del cfault
    gc.collect()
    del cactnum
    gc.collect()
    

    print('Load simulated labelled test data from .gz file')
    with gzip.open(to_absolute_path('../PACKETS/data_test.pkl.gz'), 'rb') as f:
        mat = pickle.load(f)
    X_data1t = mat
    for key, value in X_data1t.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())

    del mat
    gc.collect()
        
    for key in  X_data1t.keys():
        # Convert NaN and infinity values to 0
        X_data1t[key][np.isnan( X_data1t[key])] = 0
        X_data1t[key][np.isinf( X_data1t[key])] = 0
        #X_data1t[key] = np.clip(X_data1t[key], target_min, target_max)
        X_data1t[key] = clip_and_convert_to_float32(X_data1t[key])

    
    cPerm = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) #Permeability
    cPhi = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32)#Porosity
    cPini = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) #Initial pressure
    cSini = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) # Initial water saturation
    cfault = np.zeros((N_ens,1,nz,nx,ny),dtype=np.float32) # Fault
    cQ = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cQw = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cQg = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cTime = np.zeros((1,steppi,nz,nx,ny),dtype=np.float32) # Fault
    cactnum = np.zeros((1,1,nz,nx,ny),dtype=np.float32) # Fault

    
    cPress = np.zeros((N_ens,steppi,nz,nx,ny),dtype=np.float32) # Pressure
    cSat = np.zeros((N_ens,steppi,nz,nx,ny),dtype=np.float32) # Water saturation
    cSatg = np.zeros((N_ens,steppi,nz,nx,ny),dtype=np.float32) # gas saturation
    
    for kk in range(N_ens):
        
        #INPUTS        
        for i in range(nz):
            cPerm[kk,0,i,:,:] = X_data1t['permeability'][kk,0,:,:,i]
            cfault[kk,0,i,:,:] = X_data1t['Fault'][kk,0,:,:,i]            
            cPhi[kk,0,i,:,:] = X_data1t['porosity'][kk,0,:,:,i]
            cPini[kk,0,i,:,:] = X_data1t['Pini'][kk,0,:,:,i]/maxP
            cSini[kk,0,i,:,:] = X_data1t['Sini'][kk,0,:,:,i]
                

        #OUTPUTS
        for mum in range(steppi):
            for i in range(nz):
                cPress[kk,mum,i,:,:] = X_data1t['Pressure'][kk,mum,:,:,i]
                cSat[kk,mum,i,:,:] = X_data1t['Water_saturation'][kk,mum,:,:,i]
                cSatg[kk,mum,i,:,:] = X_data1t['Gas_saturation'][kk,mum,:,:,i]
    del X_data1t
    data_test = {'perm': cPerm,'Phi': cPhi,'Pini': cPini,'Swini':\
    cSini,'pressure': cPress,'water_sat': cSat,'gas_sat': cSatg,'fault':cfault}

    
        
    del cPerm
    gc.collect()
    del cQ
    gc.collect()
    del cQw
    gc.collect()
    del cQg
    gc.collect()
    del cPhi
    gc.collect()
    del cTime
    gc.collect()
    del cPini
    gc.collect()
    del cSini
    gc.collect()
    del cPress
    gc.collect()
    del cSat
    gc.collect()
    del cSatg
    gc.collect()
    del cfault
    gc.collect()
    del cactnum
    gc.collect() 
    
      


    print('Load simulated labelled training data for peacemann')
    with gzip.open(to_absolute_path('../PACKETS/data_train_peaceman.pkl.gz'), 'rb') as f:
        mat = pickle.load(f)
    X_data2 = mat
    del mat
    gc.collect()                  

    data2 = X_data2
    data2n = {key: value.transpose(0, 2, 1) for key, value in data2.items()}
    for key in data2n:
        data2n[key][data2n[key]<=0]=0
   
        

    print('Load simulated labelled test data for peacemann modelling')
    with gzip.open(to_absolute_path('../PACKETS/data_train_peaceman.pkl.gz'), 'rb') as f:
        mat = pickle.load(f)
    X_data2t = mat
    del mat
    gc.collect()                  

    data2_test = X_data2t
    data2n_test = {key: value.transpose(0, 2, 1) for key, value in data2_test.items()}   
    for key in data2n_test:
        data2n_test[key][data2n_test[key]<=0]=0    



    # Define FNO model 
    #Pressure
    decoder1 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("pressure", size=steppi)])
    fno_pressure = FNOArch([Key("perm", size=1),Key("Phi", size=1), Key("Pini", size=1),\
                Key("Swini", size=1),Key("fault", size=1) ],\
    dimension=3, decoder_net=decoder1)
    fno_saturation = fno_pressure.to(device)
        
    decoder2 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("water_sat", size=steppi)])
    fno_water = FNOArch([Key("perm", size=1),Key("Phi", size=1), Key("Pini", size=1),\
                Key("Swini", size=1),Key("fault", size=1) ],\
    dimension=3, decoder_net=decoder2)
    fno_water = fno_water.to(device)
        
    decoder3 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("gas_sat", size=steppi)])
    fno_gas = FNOArch([Key("perm", size=1),Key("Phi", size=1), Key("Pini", size=1),\
                Key("Swini", size=1),Key("fault", size=1) ],\
    dimension=3, decoder_net=decoder3) 
    fno_gas = fno_gas.to(device)    


    decoder4 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("Y", size=60)])
    fno_peacemann = FNOArch([Key("X", size=90) ],\
    fno_modes = 13, dimension=1,padding=20,nr_fno_layers=5, decoder_net=decoder4)
    fno_peacemann = fno_peacemann.to(device)

    RE = 0.2 * 100
    RE = torch.tensor(RE).to(device)
    DZ = torch.tensor(100).to(device)
    
    learning_rate = cfg.optimizer.lr
    gamma = 0.5
    step_size = 100
    
    optimizer_pressure = torch.optim.AdamW(fno_pressure.parameters(), 
                lr=learning_rate, weight_decay=cfg.optimizer.weight_decay)
    scheduler_pressure = torch.optim.lr_scheduler.StepLR(optimizer_pressure, 
                                            step_size=step_size, gamma=gamma)
    
    optimizer_water = torch.optim.AdamW(fno_water.parameters(), 
                    lr=learning_rate, weight_decay=cfg.optimizer.weight_decay)
    scheduler_water = torch.optim.lr_scheduler.StepLR(optimizer_water, 
                                            step_size=step_size, gamma=gamma)

    optimizer_gas = torch.optim.AdamW(fno_gas.parameters(), lr=learning_rate, 
                                    weight_decay=cfg.optimizer.weight_decay)
    scheduler_gas = torch.optim.lr_scheduler.StepLR(optimizer_gas, 
                                            step_size=step_size, gamma=gamma)    
    
    optimizer_peacemann = torch.optim.AdamW(fno_peacemann.parameters(), 
                    lr=learning_rate, weight_decay=cfg.optimizer.weight_decay)
    scheduler_peacemann = torch.optim.lr_scheduler.StepLR(optimizer_peacemann,
                                            step_size=step_size, gamma=gamma)     


    ##############################################################################
    #         START THE TRAINING OF THE MODEL WITH UNLABELLED DATA - OPERATOR LEARNING
    ##############################################################################
    print ( '>>>>>>>>>>>>>>>>>>>>>>>>>>OPERATOR LEARNING>>>>>>>>>>>>>>>>>>>>>>>>')
    hista = []
    hist2a = []
    hist22a = []
    aha = 0
    ah2a = 0
    ah22a = 0
    costa=[]
    cost2a = []
    cost22a = []
    overall_best = 0
    epochs = cfg.custom.epochs #'number of epochs to train'    
    
    myloss = LpLoss(size_average = True)         
    dataset = Labelledset(data)
    labelled_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size.grid
    )
    
    imtest_use = 1
    index = np.random.choice(100, imtest_use, \
                             replace=False)
    inn_test = {'perm': torch.from_numpy(data_test['perm'][index,:,:,:,:]).to(device,torch.float32),\
    'Phi': torch.from_numpy(data_test['Phi'][index,:,:,:,:]).to(device,dtype=torch.float32),'fault': \
    torch.from_numpy(data_test['fault'][index,:,:,:,:]).to(device,dtype=torch.float32),
     'Pini': torch.from_numpy(data_test['Pini'][index,:,:,:,:]).to(device,dtype=torch.float32),'Swini': \
     torch.from_numpy(data_test['Swini'][index,:,:,:,:]).to(device,dtype=torch.float32)} 
        
    out_test = {'pressure': torch.from_numpy(data_test['pressure'][index,:,:,:,:]).to(device,torch.float32),\
    'water_sat': torch.from_numpy(data_test['water_sat'][index,:,:,:,:]).to(device,\
    dtype=torch.float32), 'gas_sat': torch.from_numpy(data_test['gas_sat'][index,:,:,:,:]).to(device,\
        dtype=torch.float32)}



    inn_test_p = {'X': torch.from_numpy(data2n_test['X'][index,:,:]).to(device,torch.float32)}         
    out_test_p = {'Y': torch.from_numpy(data2n_test['Y'][index,:,:]).to(device,torch.float32)}
    

    inn_train_p = {'X': torch.from_numpy(data2n['X']).to(device,torch.float32)}         
    out_train_p = {'Y': torch.from_numpy(data2n['Y']).to(device,torch.float32)}    

   

    print(' Training with ' + str(N_ens) +' labelled members ')    
    start_epoch =  1
    start_time = time.time()
    imtest_use = 1
    for epoch in range(start_epoch, epochs + 1):
        fno_pressure.train()
        fno_water.train()
        fno_gas.train()
        fno_peacemann.train()
        loss_train = 0.
        print('Epoch ' + str(epoch ) + ' | ' +  str (epochs))
        print('****************************************************************')
        for inputaa in labelled_loader:

            inputin = {'perm':inputaa['perm'],'Phi':inputaa['Phi'],\
'fault': inputaa['fault'],'Pini': inputaa['Pini'],'Swini':inputaa['Swini']}


            target = {'pressure':inputaa['pressure'],'water_sat': inputaa['water_sat'],'gas_sat': inputaa['gas_sat']}

            
            optimizer_pressure.zero_grad()
            optimizer_water.zero_grad()
            optimizer_gas.zero_grad()
            optimizer_peacemann.zero_grad()
            
            
            #with torch.no_grad(): 
            #TRAIN
            outputa_p = fno_pressure(inputin)["pressure"]
            outputa_s = fno_water(inputin)["water_sat"]
            outputa_g = fno_gas(inputin)["gas_sat"]            
            outputa_pe = fno_peacemann(inn_train_p)["Y"]
    

            dtrue_p = target["pressure"]
            dtrue_s = target["water_sat"]
            dtrue_g = target["gas_sat"]
            dtrue_pe = out_train_p["Y"]
            
  
            loss_test1a = MyLossClement((outputa_p).reshape(cfg.batch_size.labelled,-1), \
                                    (dtrue_p).reshape(cfg.batch_size.labelled,-1))            
                
            loss_test2a = MyLossClement((outputa_s).reshape(cfg.batch_size.labelled,-1),\
                                     (dtrue_s).reshape(cfg.batch_size.labelled,-1)) 
                
            loss_test3a = MyLossClement((outputa_g).reshape(cfg.batch_size.labelled,-1),\
                                     (dtrue_g).reshape(cfg.batch_size.labelled,-1))

            loss_test4a = MyLossClement((outputa_pe).reshape(N_ens,-1),\
                                     (dtrue_pe).reshape(N_ens,-1))                
                
                
            loss_testa = (loss_test1a*cfg.loss.weights.pressure) + \
            (loss_test2a*cfg.loss.weights.water_sat) + \
            (loss_test3a*cfg.loss.weights.gas_sat) + \
            loss_test4a*cfg.loss.weights.Y
            #loss_testa = torch.log10(loss_testa)
            
            
            #TEST

            input_varr = inputin
            input_varr['pressure'] = outputa_p
            input_varr['water_sat'] = outputa_s
            input_varr['gas_sat'] = outputa_g
            
            input_varr1 = inputin
            input_varr1['pressure'] = dtrue_p
            input_varr1['water_sat'] = dtrue_s
            input_varr1['gas_sat'] = dtrue_g            
            
            #print(outputa_p.shape)


            f_loss2, f_water2,f_gass2 = Black_oil(neededM,input_varr,SWI,SWR,UW,BW,UO,BO,
             nx,ny,nz,SWOW,SWOG,target_min,target_max,minKx,maxKx,
             minPx,maxPx,p_bub,p_atm,CFO,maxQx,maxQwx,maxTx,maxQgx,Relperm,params,pde_method) 
            
            f_loss21, f_water21,f_gass21 = Black_oil(neededM,input_varr1,SWI,SWR,UW,BW,UO,BO,
             nx,ny,nz,SWOW,SWOG,target_min,target_max,minKx,maxKx,
             minPx,maxPx,p_bub,p_atm,CFO,maxQx,maxQwx,maxTx,maxQgx,Relperm,params,pde_method)             
            
                
            loss_pde2  = f_loss2   + f_water2  + f_gass2 + f_loss21 + f_water21 + f_gass21
            
            loss_pdea = Black_oil_peacemann(UO,BO,UW,BW,DZ,RE,inn_train_p["X"],
            out_train_p["Y"],device,max_inn_fcnx,max_out_fcnx,params,p_bub, p_atm,CFO)
            loss_pdeb = Black_oil_peacemann(UO,BO,UW,BW,DZ,RE,inn_train_p["X"],
            outputa_pe,device,max_inn_fcnx,max_out_fcnx,params,p_bub, p_atm,CFO)
            
            loss_pde = loss_pdea + loss_pdeb
            
            #loss_pe = Black_oil_peacemann(params,(inn_train_p)["X"],dtrue_pe,device,max_inn_fcn,max_out_fcn,params)

            loss2 = (((loss_pde2)*1e-7)) + \
            (((loss_testa)*cfg.custom.data_weighting)*1e5) + (loss_pde * 1e-7)
            
            model_pressure = fno_pressure
            model_saturation = fno_water
            model_gas = fno_gas
            model_peacemann = fno_peacemann
            loss2.backward()              
    
            optimizer_pressure.step()  
            optimizer_water.step()
            optimizer_gas.step()
            optimizer_peacemann.step()                
    
            loss_train += loss2.item()
            
            with torch.no_grad():
                outputa_ptest = fno_pressure(inn_test)["pressure"]
                outputa_stest = fno_water(inn_test)["water_sat"]
                outputa_gtest = fno_gas(inn_test)["gas_sat"]            
                outputa_petest = fno_peacemann(inn_test_p)["Y"]
        
    
                dtest_p = out_test["pressure"]
                dtest_s = out_test["water_sat"]
                dtest_g = out_test["gas_sat"]
                dtest_pe = out_test_p["Y"]
            
  
                loss_test1aa = MyLossClement((outputa_ptest).reshape(imtest_use,-1), \
                                        (dtest_p).reshape(imtest_use,-1))            
                    
                loss_test2aa = MyLossClement((outputa_stest).reshape(imtest_use,-1),\
                                         (dtest_s).reshape(imtest_use,-1)) 
                    
                loss_test3aa = MyLossClement((outputa_gtest).reshape(imtest_use,-1),\
                                         (dtest_g).reshape(imtest_use,-1))
    
                loss_test4aa = MyLossClement((outputa_petest).reshape(imtest_use,-1),\
                                         (dtest_pe).reshape(imtest_use,-1))                
                    
                    
                loss_testaa = loss_test1aa + loss_test2aa + loss_test3aa + loss_test4aa              
            
            
        scheduler_pressure.step()
        scheduler_water.step()
        scheduler_gas.step()
        scheduler_peacemann.step()
        
        ahnewa = loss2.detach().cpu().numpy()
        hista.append(ahnewa)
        
        ahnew2a = loss_testa.detach().cpu().numpy()
        hist2a.append(ahnew2a) 
    
        ahnew22a = loss_testaa.detach().cpu().numpy()
        hist22a.append(ahnew22a) 
        

           
        print(   'TRAINING')        
        if aha<ahnewa:
            print('   FORWARD PROBLEM COMMENT (Overall loss)) : Loss increased by ' + \
                  str(abs(aha-ahnewa)))
  
        elif aha>ahnewa:
            print('   FORWARD PROBLEM COMMENT (Overall loss)) : Loss decreased by ' + \
                  str(abs(aha-ahnewa)) )

                
                
        else:
            print('   FORWARD PROBLEM COMMENT (Overall loss) : No change in Loss ')          
    

    
        print('   training loss = ' + str(ahnewa))
        print('   pde loss dynamic = ' + str(loss_pde2.detach().cpu().numpy()))
        print('   pde loss peacemann = ' + str(loss_pde.detach().cpu().numpy()))
        print('   data loss = ' + str(loss_testa.detach().cpu().numpy()))
        print('    ******************************   ' )
        print('   Labelled pressure equation loss = ' + str(f_loss2.detach().cpu().numpy()))
        print('   Labelled saturation equation loss = ' + str(f_water2.detach().cpu().numpy()))
        print('   Labelled gas equation loss = ' + str(f_gass2.detach().cpu().numpy()))
        print('    ******************************   ' )

        print('   unlabelled pressure equation loss = ' + str(f_loss21.detach().cpu().numpy()))
        print('   unlabelled saturation equation loss = ' + str(f_water21.detach().cpu().numpy()))
        print('   unlabelled gas equation loss = ' + str(f_gass21.detach().cpu().numpy()))
        print('    ******************************   ' )         
        
        if ah2a<ahnew2a:
            print('   TRAINING COMMENT : Loss increased by ' + str(abs(ah2a-ahnew2a)))
            
        elif ah2a>ahnew2a:
            print('   TRAINING COMMENT : Loss decreased by ' + str(abs(ah2a-ahnew2a)) )
            

        else:
            print('   TRAINING COMMENT : No change in Loss ')     
        print('   Train loss = ' +str(ahnew2a)) 
        print('   Train: Pressure loss = ' +str(loss_test1a.detach().cpu().numpy())) 
        print('   Train: saturation loss = ' +str(loss_test2a.detach().cpu().numpy()))
        print('   Train: gas loss = ' +str(loss_test3a.detach().cpu().numpy()))
        print('   Train: peacemann loss = ' +str(loss_test4a.detach().cpu().numpy()))
        
        print('    ******************************   ' ) 
        
        
        if ah22a<ahnew22a:
            print('   TEST COMMENT : Loss increased by ' + str(abs(ah22a-ahnew22a)))
            
        elif ah22a>ahnew22a:
            print('   Test COMMENT : Loss decreased by ' + str(abs(ah22a-ahnew22a)) )
            

        else:
            print('   TEST COMMENT : No change in Loss ')     
        print('   Test loss = ' +str(ahnew22a)) 
        print('   Test: Pressure loss = ' +str(loss_test1aa.detach().cpu().numpy())) 
        print('   Test: saturation loss = ' +str(loss_test2aa.detach().cpu().numpy()))
        print('   Test: gas loss = ' +str(loss_test3aa.detach().cpu().numpy()))
        print('   Test: peacemann loss = ' +str(loss_test4aa.detach().cpu().numpy()))
        
        print('    ******************************   ' )         
        

        aha = ahnewa
        ah2a = ahnew2a
        ah22a = ahnew22a


        if epoch == 1:
            best_cost = aha
        else:
            pass  
        if best_cost > ahnewa:
            print('    ******************************   ' )
            print('   Forward models saved')
            print('   Current best cost = ' + str(best_cost))
            print('   Current epoch cost = ' + str(ahnewa))
            torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            torch.save(model_saturation.state_dict(), oldfolder + '/water_model.pth')
            torch.save(model_gas.state_dict(), oldfolder + '/gas_model.pth')
            torch.save(model_peacemann.state_dict(), oldfolder + '/peacemann_model.pth')
            best_cost = ahnewa
        else:
            print('    ******************************   ' )
            print('   Forward models NOT saved')
            print('   Current best cost = ' + str(best_cost))
            print('   Current epoch cost = ' + str(ahnewa))
        
        costa.append(ahnewa)
        cost2a.append(ahnew2a)
        cost22a.append(ahnew22a)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>Finished training >>>>>>>>>>>>>>>>> ')     
    elapsed_time_secs = time.time() - start_time
    msg = "PDE learning Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)  

    print('')     
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.semilogy(range(len(hista)), hista,'k-')
    plt.title('Forward problem -Overall loss', fontsize = 13)
    plt.ylabel('$\\phi_{n_{epoch}}$',fontsize = 13)
    plt.xlabel('$n_{epoch}$',fontsize = 13)
    
    
    
    plt.subplot(2, 2, 2)
    plt.semilogy(range(len(hist2a)), hist2a,'k-')
    plt.title('Training loss', fontsize = 13)
    plt.ylabel('$\\phi_{n_{epoch}}$',fontsize = 13)
    plt.xlabel('$n_{epoch}$',fontsize = 13)
    
    
    plt.subplot(2, 2, 3)
    plt.semilogy(range(len(hist22a)), hist22a,'k-')
    plt.title('Test loss', fontsize = 13)
    plt.ylabel('$\\phi_{n_{epoch}}$',fontsize = 13)
    plt.xlabel('$n_{epoch}$',fontsize = 13)
    
        
    plt.tight_layout(rect = [0,0,1,0.95])
    plt.suptitle('MODEL LEARNING',\
    fontsize = 16) 
    plt.savefig(oldfolder + '/cost_PyTorch.png')
    plt.close()
    plt.clf()   


if __name__ == "__main__":
    run()

