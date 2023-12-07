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

from typing import Dict
import torch.nn.functional as F
from ops import dx, ddx,compute_gradient_3d,compute_second_order_gradient_3d
from modulus.sym.utils.io.plotter import ValidatorPlotter
from typing import Union, Tuple
from pathlib import Path


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

# [pde-loss]
# define custom class for black oil model

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

def compute_peacemannoil(UO, BO, UW, BW, DZ, RE, device, max_inn_fcn, max_out_fcn, paramz,
                      p_bub, p_atm, steppi, CFO, sgas, swater, pressure, permeability):
    qoil = torch.zeros_like(sgas).to(device)
    skin = 0
    rwell = 200
    pwf_producer = 100

    def process_location(i, j, k, l):
        pre1 = pressure[i, j, :, :, :]
        sg1 = sgas[i, j, :, k, l]
        sw1 = swater[i, j, :, k, l]
        krw, kro, krg = StoneIIModel(paramz, device, sg1, sw1)
        BO_val = calc_bo(p_bub, p_atm, CFO, pre1.mean())
        up = UO * BO_val
        perm1 = permeability[i, 0, :, k, l]
        down = 2 * torch.pi * perm1 * kro * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        drawdown = pre1.mean() - pwf_producer
        qoil1 = torch.abs(-(drawdown * J))
        return -qoil1

    locations = [(14, 30), (9, 31), (13, 33), (8, 36), (8, 45), (9, 28), (9, 23),
                 (21, 21), (13, 27), (18, 37), (18, 53), (15, 65), (24, 36),
                (18, 53),(11, 71),(17, 67),(12, 66),(37, 97),(6, 63),(14, 75)
                ,(12, 66),(10, 27)]

    for m in range(sgas.shape[0]):
        for step in range(sgas.shape[1]):
            for location in locations:
                qoil[m, step, :, location[0], location[1]] = process_location(m, step, *location)

    return qoil

 

class Black_oil_peacemann(torch.nn.Module):
    def __init__(self,UO,BO,UW,BW,DZ,RE,device,max_inn_fcn,max_out_fcn,paramz,p_bub, p_atm,steppi,CFO):
        super().__init__()
        self.UW = UW
        self.BW = BW
        self.UO = UO
        self.BO = BO
        self.DZ = DZ
        self.device = device
        self.RE = RE
        self.p_bub = p_bub
        self.p_atm = p_atm
        self.paramz = paramz
        self.max_inn_fcn = max_inn_fcn 
        self.max_out_fcn = max_out_fcn
        self.steppi = steppi
        self.CFO = CFO
        
    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:    
    
        in_var = input_var["X"]
        out_var = input_var["Y"]


        skin = 0
        rwell = 200
        spit = []
        N = in_var.shape[0]
        pwf_producer = 100
        spit = torch.zeros(N,66,self.steppi).to(self.device)
        #loss_rest = torch.zeros(self.steppi,13).to(self.device)
        for clement in range(N):
    
            inn = in_var[clement,:,:].T * self.max_inn_fcn
            outt = out_var[clement,:,:].T * self.max_out_fcn
            
            oil_rate = outt[:,:22]
            water_rate = outt[:,22:44]
            gas_rate = outt[:,44:]
            
            permeability = inn[:,:22]
            pressure = inn[:,22]
            #oil = inn[:,44:66]
            gas =inn[:,45:67]
            water =inn[:,67:89]
            
            #Compute relative permeability
            krw,kro,krg = StoneIIModel (self.paramz,self.device,gas,water)
            
            # Compute Oil rate loss
            #krw,kro,krg = StoneIIModel (paramz,device,gas,water)
            BO = calc_bo(self.p_bub, self.p_atm, self.CFO, pressure.mean())
            up = self.UO *  BO
            down = 2 * torch.pi * permeability * kro * self.DZ
            right = torch.log(self.RE / rwell) + skin
            J = down / (up * right)
            # drawdown = p_prod - pwf_producer
            drawdown = pressure.mean() - pwf_producer
            qoil = torch.abs(-(drawdown * J))
            loss_oil = qoil - oil_rate
            loss_oil = ((loss_oil))/N
    
            # Compute water rate loss
            up = self.UW * self.BW
            down = 2 * torch.pi * permeability * krw * self.DZ
            right = torch.log(self.RE / rwell) + skin
            J = down / (up * right)
            # drawdown = p_prod - pwf_producer
            drawdown = pressure.mean() - pwf_producer
            qwater = torch.abs(-(drawdown * J))
            loss_water = qwater - water_rate 
            loss_water = ((loss_water))/N
            
    
            # Compute gas rate loss
            UG = calc_mu_g(pressure.mean())
            BG = calc_bg(self.p_bub, self.p_atm, pressure.mean())
            
            up = UG * BG
            down = 2 * torch.pi * permeability * krg * self.DZ
            right = torch.log(self.RE / rwell) + skin
            J = down / (up * right)
            # drawdown = p_prod - pwf_producer
            drawdown = pressure.mean() - pwf_producer
            qgas = torch.abs(-(drawdown * J))
            loss_gas = qgas - gas_rate
            loss_gas = ((loss_gas))/N
    
            overall_loss = torch.cat((loss_oil, loss_water, loss_gas), dim=1)
            overall_loss = overall_loss.T
            
            #print(overall_loss.shape)
            
            spit[clement,:,:] = overall_loss * 1e-10
        
        
        output_var = {"peacemanned": spit}

        return output_var  

    
class Black_oil(torch.nn.Module):
    "Custom Black oil PDE definition for PINO"

    def __init__(self,neededM,SWI,SWR,UW,BW,UO,BO,
                 nx,ny,nz,SWOW,SWOG,target_min,target_max,minK,maxK,
                 minP,maxP,p_bub,p_atm,CFO,Relperm,params,pde_method,RE,max_inn_fcn,max_out_fcn,DZ):
        super().__init__()
        self.neededM = neededM
        self.SWI = SWI
        self.SWR = SWR
        self.UW = UW
        self.BW = BW
        self.UO = UO
        self.BO = BO
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.SWOW = SWOW
        self.SWOG = SWOG
        self.target_min= target_min
        self.target_max = target_max
        self.minK = minK
        self.maxK = maxK
        self.minP = minP
        self.maxP = maxP
        self.p_bub = p_bub
        self.p_atm = p_atm
        self.CFO = CFO
        self.Relperm = Relperm
        self.params = params
        self.pde_method = pde_method
        self.RE = RE
        self.max_inn_fcn = max_inn_fcn
        self.max_out_fcn = max_out_fcn
        self.DZ = DZ
    
    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:    
    
        u = input_var["pressure"]
        perm = input_var["perm"]
        fin = self.neededM["Q"]
        fin = fin.repeat(u.shape[0], 1, 1, 1, 1)
        fin = fin.clamp(min=0)
        finwater = self.neededM["Qw"]
        finwater = finwater.repeat(u.shape[0], 1, 1, 1, 1)
        finwater = finwater.clamp(min=0)
        dt = self.neededM["Time"]
        pini = input_var["Pini"]
        poro = input_var["Phi"]
        sini = input_var["Swini"]
        sat = input_var["water_sat"]
        satg = input_var["gas_sat"]
        fault = input_var["fault"]
        fingas = self.neededM["Qg"]
        fingas = fingas.repeat(u.shape[0], 1, 1, 1, 1)
        fingas = fingas.clamp(min=0)
        actnum = self.neededM["actnum"]
        actnum = actnum.repeat(u.shape[0], 1, 1, 1, 1)
        sato = 1-(sat + satg)
        siniuse = sini[0,0,0,0,0]    
        dxf = 1e-2

     
        
        #Rescale back
    
        #pressure
        u = u * self.maxP
        
        #Initial_pressure
        pini = pini * self.maxP
        #Permeability
        a = perm * self.maxK
        permyes = a
        
        #Pressure equation Loss
        cuda = 0
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")         
    
         
        #print(pressurey.shape)            
        p_loss = torch.zeros_like(u).to(device,torch.float32)
        s_loss = torch.zeros_like(u).to(device,torch.float32)
    
        finusew = finwater
    
         
        prior_pressure = torch.zeros(sat.shape[0],sat.shape[1],\
                            self.nz,self.nx,self.ny).to(device,torch.float32)
        prior_pressure[:,0,:,:,:] = pini[:,0,:,:,:]
        prior_pressure[:,1:,:,:,:] = u[:,:-1,:,:,:]
        
        avg_p = prior_pressure.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)\
        .mean(dim=4, keepdim=True)
        UG = calc_mu_g(avg_p)
        RS = calc_rs(self.p_bub, avg_p)
        BG = calc_bg(self.p_bub, self.p_atm, avg_p)
        BO = calc_bo(self.p_bub, self.p_atm, self.CFO, avg_p)
        
     
        avg_p = torch.where(torch.isnan(avg_p), torch.tensor(0.0, device=avg_p.device), avg_p)
        avg_p = torch.where(torch.isinf(avg_p), torch.tensor(0.0, device=avg_p.device), avg_p) 
        
        UG = torch.where(torch.isnan(UG), torch.tensor(0.0, device=UG.device), UG)
        UG = torch.where(torch.isinf(UG), torch.tensor(0.0, device=UG.device), UG)
    
        BG = torch.where(torch.isnan(BG), torch.tensor(0.0, device=BG.device), BG)
        BG = torch.where(torch.isinf(BG), torch.tensor(0.0, device=BG.device), BG)
    
        RS = torch.where(torch.isnan(RS), torch.tensor(0.0, device=RS.device), RS)
        RS = torch.where(torch.isinf(RS), torch.tensor(0.0, device=RS.device), RS)        
    
        BO = torch.where(torch.isnan(BO), torch.tensor(0.0, device=BO.device), BO)
        BO = torch.where(torch.isinf(BO), torch.tensor(0.0, device=BO.device), BO)
    
    
        #dsp = u - prior_pressure  #dp
        
        prior_sat = torch.zeros(sat.shape[0],sat.shape[1],\
                            self.nz,self.nx,self.ny).to(device,torch.float32)
        prior_sat[:,0,:,:,:] = siniuse * \
        (torch.ones(sat.shape[0],self.nz,self.nx,self.ny).to(device,torch.float32)) 
        prior_sat[:,1:,:,:,:] = sat[:,:-1,:,:,:] 
        
        prior_gas = torch.zeros(sat.shape[0],sat.shape[1],\
                            self.nz,self.nx,self.ny).to(device,torch.float32)
        prior_gas[:,0,:,:,:] = (torch.zeros(sat.shape[0],self.nz,self.nx,self.ny).\
                            to(device,torch.float32)) 
        prior_gas[:,1:,:,:,:] = satg[:,:-1,:,:,:] 
        
        
        prior_time = torch.zeros(sat.shape[0],sat.shape[1],\
                            self.nz,self.nx,self.ny).to(device,torch.float32)
        prior_time[:,0,:,:,:] = (torch.zeros(sat.shape[0],self.nz,self.nx,self.ny).\
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
        
        #KRW, KRO, KRG = RelPerm(prior_sat,prior_gas, SWI, SWR, SWOW, SWOG)
        if self.Relperm ==1:
            one_minus_swi_swr = 1 - (self.SWI + self.SWR)
        
        
            soa = torch.divide((1 - (prior_sat + prior_gas) - self.SWR),one_minus_swi_swr)  
            swa = torch.divide((prior_sat - self.SWI) ,one_minus_swi_swr)
            sga = torch.divide(prior_gas,one_minus_swi_swr)
            
        
        
            KROW = linear_interp(prior_sat, self.SWOW[:, 0], self.SWOW[:, 1])
            KRW = linear_interp(prior_sat, self.SWOW[:, 0], self.SWOW[:, 2])
            KROG = linear_interp(prior_gas, self.SWOG[:, 0], self.SWOG[:, 1])
            KRG = linear_interp(prior_gas, self.SWOG[:, 0], self.SWOG[:, 2])
        
            KRO = (torch.divide(KROW , (1 - swa)) * torch.divide(KROG , (1 - sga))) * soa
        else:
            KRW,KRO,KRG = StoneIIModel (self.params,device,prior_gas,prior_sat)
	                
        Mw = torch.divide(KRW,(self.UW * self.BW))
        Mo = torch.divide(KRO,(self.UO * BO))
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
        
        if self.pde_method == 1:
            #compute first dffrential for pressure
            dudx_fdm = compute_gradient_3d(u, dx=dxf, dim=0, order=1, padding="replication")
            dudy_fdm = compute_gradient_3d(u, dx=dxf, dim=1, order=1, padding="replication")
            dudz_fdm = compute_gradient_3d(u, dx=dxf, dim=2, order=1, padding="replication")
        
            #Compute second diffrential for pressure    
            dduddx_fdm = compute_second_order_gradient_3d(u, dx=dxf, dim=0, padding="replication")
            dduddy_fdm = compute_second_order_gradient_3d(u, dx=dxf, dim=1, padding="replication")
            dduddz_fdm = compute_second_order_gradient_3d(u, dx=dxf, dim=2, padding="replication")
        
            
            
            #compute first dffrential for effective overall permeability
            dcdx = compute_gradient_3d(a1.float(), dx=dxf, dim=0, order=1, padding="replication")
            dcdy = compute_gradient_3d(a1.float(), dx=dxf, dim=1, order=1, padding="replication")
            dcdz = compute_gradient_3d(a1.float(), dx=dxf, dim=2, order=1, padding="replication")        
            
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
            
            # Compute darcy_pressure using the sanitized tensors
            finoil = compute_peacemannoil(self.UO, BO, self.UW, self.BW, self.DZ, self.RE, device, self.max_inn_fcn, \
                                self.max_out_fcn, self.params,
                                  self.p_bub, self.p_atm, prior_sat.shape[1], self.CFO, \
                            prior_gas, prior_sat, prior_pressure, permyes)
                                  
            fin = finoil + fingas + finwater 
                     
            darcy_pressure = torch.mul(actnum, (fin + dcdx * dudx_fdm + a1 * dduddx_fdm\
            + dcdy * dudy_fdm + a1 * dduddy_fdm + dcdz * dudz_fdm + a1 * dduddz_fdm))
                         
        
           
            # Zero outer boundary
            #darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0) 
            darcy_pressure = dxf * darcy_pressure * 1e-10
            
            p_loss = darcy_pressure
        
        
            # Water Saturation equation loss
            dudx = dudx_fdm
            dudy = dudy_fdm
            dudz = dudz_fdm
            
            dduddx = dduddx_fdm
            dduddy = dduddy_fdm
            dduddz = dduddz_fdm         
        
         
            
            #compute first diffrential for effective water permeability
            dadx = compute_gradient_3d(a1water.float(), dx=dxf, dim=0, order=1, padding="replication")
            dady = compute_gradient_3d(a1water.float(), dx=dxf, dim=1, order=1, padding="replication")
            dadz = compute_gradient_3d(a1water.float(), dx=dxf, dim=2, order=1, padding="replication")        
            
        
        
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
            inner_diff = dadx * dudx + a1water * dduddx + dady * dudy + a1water * dduddy \
            + dadz * dudz + a1water * dduddz + finusew
            
            darcy_saturation = torch.mul(actnum, (poro * torch.divide(dsw, dtime) - inner_diff))
        
        
            # Zero outer boundary
            #darcy_saturation = F.pad(darcy_saturation[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0) 
            darcy_saturation = dxf * darcy_saturation * 1e-10
        
            
            s_loss = darcy_saturation
        
            
            
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
        
            dubigxdx = compute_gradient_3d(Ubigx.float(), dx=dxf, dim=0, order=1, padding="replication")
            dubigxdy = compute_gradient_3d(Ubigx.float(), dx=dxf, dim=1, order=1, padding="replication")
            dubigxdz = compute_gradient_3d(Ubigx.float(), dx=dxf, dim=2, order=1, padding="replication")         
        
            #compute first dffrential
            dubigydx = compute_gradient_3d(Ubigy.float(), dx=dxf, dim=0, order=1, padding="replication")
            dubigydy = compute_gradient_3d(Ubigy.float(), dx=dxf, dim=1, order=1, padding="replication")
            dubigydz = compute_gradient_3d(Ubigy.float(), dx=dxf, dim=2, order=1, padding="replication") 
        
            dubigzdx = compute_gradient_3d(Ubigz.float(), dx=dxf, dim=0, order=1, padding="replication")
            dubigzdy = compute_gradient_3d(Ubigz.float(), dx=dxf, dim=1, order=1, padding="replication")
            dubigzdz = compute_gradient_3d(Ubigz.float(), dx=dxf, dim=2, order=1, padding="replication")         
                
        
            
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
            inner_sum = dubigxdx + dubigxdy + dubigxdz + dubigydx + dubigydy\
            + dubigydz + dubigzdx + dubigzdy + dubigzdz - 9*fingas
            
            div_term = torch.divide(torch.mul(poro, (torch.divide(satg, BG) + torch.mul(torch.divide(sato, BO), RS))), dtime)
            
            darcy_saturationg = torch.mul(actnum, (inner_sum + div_term))
                
                   
            sg_loss = dxf * darcy_saturationg * 1e-10
        

        else:
            #dxf = 1/1000
            
            # compute first dffrential
            gulpa = []
            gulp2a = []
            for m in range(sat.shape[0]):  # Batch
                inn_now = u[m, :, :, :, :]
                gulp = []
                gulp2 = []
                for i in range(self.nz):
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
                for i in range(self.nz):
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
            for i in range(self.nz):
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
            dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
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
    
            dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
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
            
            finoil = compute_peacemannoil(self.UO, BO, self.UW, self.BW, self.DZ, self.RE, device, self.max_inn_fcn, \
                                self.max_out_fcn, self.params,
                                  self.p_bub, self.p_atm, prior_sat.shape[1], self.CFO, \
                            prior_gas, prior_sat, prior_pressure, permyes)
                                  
            fin = finoil + fingas + finwater             
    
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
            p_loss = (torch.abs(p_loss))/sat.shape[0]
           # p_loss = p_loss.reshape(1, 1)
            p_loss = dxf * p_loss * 1e-30 

            # Saruration equation loss
            dudx = dudx_fdm
            dudy = dudy_fdm
    
            dduddx = dduddx_fdm
            dduddy = dduddy_fdm
    
            gulp = []
            gulp2 = []
            for i in range(self.nz):
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
    
            dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
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
    
            dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
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
            s_loss = (torch.abs(s_loss))/sat.shape[0]
            #s_loss = s_loss.reshape(1, 1)
            s_loss = dxf * s_loss * 1e-30            
                
            
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
                for i in range(self.nz):
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
                for i in range(self.nz):
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
                
            sg_loss = (torch.abs(sg_loss))/sat.shape[0]
            #sg_loss = sg_loss.reshape(1, 1)
            sg_loss = dxf * sg_loss * 1e-30
            
        #p_loss = torch.mul(actnum,p_loss)    
        p_loss = torch.where(torch.isnan(p_loss), torch.tensor(0, device=p_loss.device), p_loss)
        p_loss = torch.where(torch.isinf(p_loss), torch.tensor(0, device=p_loss.device), p_loss)
    
    
        s_loss = torch.where(torch.isnan(s_loss), torch.tensor(0, device=s_loss.device), s_loss)
        s_loss = torch.where(torch.isinf(s_loss), torch.tensor(0, device=s_loss.device), s_loss)

    
    
        sg_loss = torch.where(torch.isnan(sg_loss), torch.tensor(0, device=sg_loss.device), sg_loss)
        sg_loss = torch.where(torch.isinf(sg_loss), torch.tensor(0, device=sg_loss.device), sg_loss)        
        output_var = {"pressured": p_loss,"saturationd": s_loss,"saturationdg": sg_loss}
        
        # for key, tensor in output_var.items():
        #     tensor_clone = tensor.clone()
        #     replacement_tensor = torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
        #     output_var[key] = torch.where(torch.isnan(tensor_clone), replacement_tensor, tensor_clone)
        #     output_var[key] = torch.where(torch.isinf(tensor_clone), replacement_tensor, tensor_clone)



        # Calculate the square root of torch.float32 max value
        max_float32 = torch.tensor(torch.finfo(torch.float32).max)
        sqrt_max_float32 = torch.sqrt(max_float32)
        
        # Define the range for clipping
        clip_range = sqrt_max_float32.item()  # Convert to a Python float
        #clip_range = clip_range * 1e-4
        
        # Adjust this value as needed
        small_value = 1e-6
        
        for key, tensor in output_var.items():
            tensor_clone = tensor.clone()
            replacement_tensor = torch.full_like(tensor_clone, small_value)
            clipped_tensor = torch.where(
                torch.isnan(tensor_clone) | torch.isinf(tensor_clone) | (tensor_clone == 0),
                replacement_tensor,
                tensor_clone
            )
            clipped_tensor = torch.clamp(clipped_tensor, -clip_range, clip_range)
            clipped_tensor = torch.sqrt(torch.abs(clipped_tensor))
            output_var[key] = clipped_tensor
            


        for key, tensor in output_var.items():
            max_value = torch.max(tensor).item()
            min_value = torch.min(tensor).item()
        
            print(f"Key: {key}")
            print(f"Maximum value (before): {max_value}")
            print(f"Minimum value (before): {min_value}")
            print()
        
        threshold = 1e-6  # Define the minimum threshold value
        max_threshold = 1e6  # Define the maximum threshold value
        
        for key, tensor in output_var.items():
            # Check if the minimum value is less than the threshold
            if torch.min(tensor) < threshold:
                # Replace values below the threshold with the threshold
                tensor = torch.clamp(tensor, min=threshold)
        
            # Check if the maximum value is greater than the max_threshold
            if torch.max(tensor) > max_threshold:
                # Replace values above max_threshold with max_threshold
                tensor = torch.clamp(tensor, max=max_threshold)
        
            # Update the value in the output_var dictionary
            output_var[key] = tensor
        
            print(f"Key: {key}")
            print(f"Maximum value (after): {torch.max(tensor).item()}")
            print(f"Minimum value (after): {torch.min(tensor).item()}")
            print()



        return output_var

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

    DEFAULT = None
    while True:
        DEFAULT=int(input('Use best default options:\n1=Yes\n2=No\n'))
        if (DEFAULT>2) or (DEFAULT<1):
            #raise SyntaxError('please select value between 1-2')
            print('')
            print('please try again and select value between 1-2')
        else:
            
            break
    print('')
    if DEFAULT == 1:
        print('Default configuration selected, sit back and relax.....')
    else:
        pass
    
    if DEFAULT ==1:
        interest = 2
    else:
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
    if DEFAULT ==1:
        Relperm = 2
    else:
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
    else:
        pass
    
    print('')
    if DEFAULT ==1:
        pde_method = 2
    else:        
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
        
    #print(steppi)    
    #steppi = 246 
    """
    input_channel = 5 #[Perm,Phi,initial_pressure, initial_water_sat,FTM] 
    output_channel = 3 #[Pressure, Sw,Sg]
    """
    

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
    DZ = torch.tensor(100).to(device)
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
        
    with gzip.open(to_absolute_path('../PACKETS/simulations_train.pkl.gz'), 'wb') as f4:
	    pickle.dump(data, f4)        

    preprocess_FNO_mat2(to_absolute_path("../PACKETS/simulations_train.pkl.gz")) 
    os.remove(to_absolute_path("../PACKETS/simulations_train.pkl.gz"))    
    del data
    gc.collect()        
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
    gc.collect()
    data_test = {'perm': cPerm,'Phi': cPhi,'Pini': cPini,'Swini':\
    cSini,'pressure': cPress,'water_sat': cSat,'gas_sat': cSatg,'fault':cfault}

    
    with gzip.open(to_absolute_path('../PACKETS/simulations_test.pkl.gz'), 'wb') as f4:
	    pickle.dump(data_test, f4)        

    preprocess_FNO_mat2(to_absolute_path("../PACKETS/simulations_test.pkl.gz"))
    os.remove(to_absolute_path("../PACKETS/simulations_test.pkl.gz"))     
        
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
    
    
    del data_test
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
 
    del X_data2
    gc.collect()
    
    sio.savemat(to_absolute_path("../PACKETS/peacemann_train.mat"),data2n,do_compression=True)
    preprocess_FNO_mat(to_absolute_path("../PACKETS/peacemann_train.mat")) 
    del data2
    del data2n
    gc.collect()
    

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
    
    sio.savemat(to_absolute_path("../PACKETS/peacemann_test.mat"),data2n_test,do_compression=True)
    preprocess_FNO_mat(to_absolute_path("../PACKETS/peacemann_test.mat"))    
    del X_data2t
    gc.collect()
    del data2_test
    del data2n_test
    gc.collect()
    
    #threshold = 1000000


    # load training/ test data for Numerical simulation model
    input_keys = [
        Key("perm"),
        Key("Phi"),
        Key("Pini"),
        Key("Swini"),
        Key("fault"),        
    ]

    output_keys_pressure = [
        Key("pressure"),       
    ]
    
    output_keys_water = [
        Key("water_sat"),       
    ]
    
    output_keys_gas = [
        Key("gas_sat"),        
    ]
    


    invar_train, outvar_train1,outvar_train2,outvar_train3 = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulations_train.pk.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_water],
        [k.name for k in output_keys_gas],
        n_examples = N_ens,
    )
    os.remove(to_absolute_path("../PACKETS/simulations_train.pk.hdf5"))
    #os.remove(to_absolute_path("../PACKETS/simulations_train.mat"))
    
    for key in invar_train.keys():
        invar_train[key][np.isnan(invar_train[key])] = 0           # Convert NaN to 0
        invar_train[key][np.isinf(invar_train[key])] = 0           # Convert infinity to 0
        invar_train[key] = clip_and_convert_to_float32(invar_train[key])
        
    for key, value in invar_train.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)

        

    for key in outvar_train1.keys():
        outvar_train1[key][np.isnan(outvar_train1[key])] = 0           # Convert NaN to 0
        outvar_train1[key][np.isinf(outvar_train1[key])] = 0           # Convert infinity to 0 
        outvar_train1[key] = clip_and_convert_to_float32(outvar_train1[key])
    for key, value in outvar_train1.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)
        

    for key in outvar_train2.keys():
        outvar_train2[key][np.isnan(outvar_train2[key])] = 0           # Convert NaN to 0
        outvar_train2[key][np.isinf(outvar_train2[key])] = 0           # Convert infinity to 0 
        outvar_train2[key] = clip_and_convert_to_float32(outvar_train2[key])
    for key, value in outvar_train2.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)
        
    for key in outvar_train3.keys():
        outvar_train3[key][np.isnan(outvar_train3[key])] = 0           # Convert NaN to 0
        outvar_train3[key][np.isinf(outvar_train3[key])] = 0           # Convert infinity to 0 
        outvar_train3[key] = clip_and_convert_to_float32(outvar_train3[key])
    for key, value in outvar_train3.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)        
    

    # outvar_train1["pressured"] = np.zeros_like(outvar_train1["pressure"])
    # outvar_train2["saturationd"] = np.zeros_like(outvar_train2["water_sat"])
    # outvar_train3["saturationdg"] = np.zeros_like(outvar_train3["gas_sat"])
       
        

    
    invar_test, outvar_test1,outvar_test2,outvar_test3 = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulations_test.pk.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_water],
        [k.name for k in output_keys_gas],        
        n_examples=cfg.custom.ntest,
    )
    os.remove(to_absolute_path("../PACKETS/simulations_test.pk.hdf5"))
    #os.remove(to_absolute_path("../PACKETS/simulations_test.mat"))
    

    for key in invar_test.keys():
        invar_test[key][np.isnan(invar_test[key])] = 0           # Convert NaN to 0
        invar_test[key][np.isinf(invar_test[key])] = 0           # Convert infinity to 0
        invar_test[key] = clip_and_convert_to_float32(invar_test[key])
    for key, value in invar_test.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)        

    for key in outvar_test1.keys():
        outvar_test1[key][np.isnan(outvar_test1[key])] = 0           # Convert NaN to 0
        outvar_test1[key][np.isinf(outvar_test1[key])] = 0           # Convert infinity to 0 
        outvar_test1[key] = clip_and_convert_to_float32(outvar_test1[key])
    for key, value in outvar_test1.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize of = :", value.shape)

    for key in outvar_test2.keys():
        outvar_test2[key][np.isnan(outvar_test2[key])] = 0           # Convert NaN to 0
        outvar_test2[key][np.isinf(outvar_test2[key])] = 0           # Convert infinity to 0 
        outvar_test2[key] = clip_and_convert_to_float32(outvar_test2[key])
    for key, value in outvar_test2.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize of = :", value.shape)
        
        
    for key in outvar_test3.keys():
        outvar_test3[key][np.isnan(outvar_test3[key])] = 0           # Convert NaN to 0
        outvar_test3[key][np.isinf(outvar_test3[key])] = 0           # Convert infinity to 0 
        outvar_test3[key] = clip_and_convert_to_float32(outvar_test3[key])
    for key, value in outvar_test3.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize of = :", value.shape)        



    # load training/ test data for peaceman model
    input_keysp = [
        Key("X"),        
    ]
    output_keysp = [
        Key("Y")
    ]
    
    # parse data

    invar_trainp, outvar_trainp = load_FNO_dataset2a(
        to_absolute_path("../PACKETS/peacemann_train.hdf5"),
        [k.name for k in input_keysp],
        [k.name for k in output_keysp],
        n_examples = N_ens,
    )
    invar_testp, outvar_testp = load_FNO_dataset2a(
        to_absolute_path("../PACKETS/peacemann_test.hdf5"),
        [k.name for k in input_keysp],
        [k.name for k in output_keysp],        
        n_examples=cfg.custom.ntest,
    )
    
    for key, value in invar_trainp.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)
    
    for key, value in invar_testp.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)
    
    for key, value in outvar_trainp.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)   

    for key, value in outvar_testp.items():
    	print(f"For key '{key}':")
    	print("\tContains inf:", np.isinf(value).any())
    	print("\tContains -inf:", np.isinf(-value).any())
    	print("\tContains NaN:", np.isnan(value).any())
    	print("\tSize = :", value.shape)


    os.remove(to_absolute_path("../PACKETS/peacemann_train.hdf5"))  
    os.remove(to_absolute_path("../PACKETS/peacemann_train.mat"))
    os.remove(to_absolute_path("../PACKETS/peacemann_test.hdf5"))
    os.remove(to_absolute_path("../PACKETS/peacemann_test.mat")) 

    outvar_trainp["peacemanned"] = np.zeros_like(outvar_trainp["Y"])       


    train_dataset_pressure = DictGridDataset(invar_train, outvar_train1)
    train_dataset_water = DictGridDataset(invar_train, outvar_train2)
    train_dataset_gas = DictGridDataset(invar_train, outvar_train3)
    train_dataset_p = DictGridDataset(invar_trainp, outvar_trainp)
    

    test_dataset_pressure = DictGridDataset(invar_test, outvar_test1)
    test_dataset_water = DictGridDataset(invar_test, outvar_test2)
    test_dataset_gas = DictGridDataset(invar_test, outvar_test3)
    test_dataset_p = DictGridDataset(invar_testp, outvar_testp)
    


    # [init-node]
    # Make custom Darcy residual node for PINO

    # Define FNO model 
    #Pressure
    decoder1 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("pressure", size=steppi)])
    fno_pressure = FNOArch([Key("perm", size=1),Key("Phi", size=1), Key("Pini", size=1),\
                Key("Swini", size=1),Key("fault", size=1) ],\
    dimension=3, decoder_net=decoder1) 
        
    decoder2 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("water_sat", size=steppi)])
    fno_water = FNOArch([Key("perm", size=1),Key("Phi", size=1), Key("Pini", size=1),\
                Key("Swini", size=1),Key("fault", size=1) ],\
    dimension=3, decoder_net=decoder2) 
        
    decoder3 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("gas_sat", size=steppi)])
    fno_gas = FNOArch([Key("perm", size=1),Key("Phi", size=1), Key("Pini", size=1),\
                Key("Swini", size=1),Key("fault", size=1) ],\
    dimension=3, decoder_net=decoder3) 
        


    decoder4 = ConvFullyConnectedArch([Key("z", size=32)], \
                [Key("Y", size=66)])
    fno_peacemann = FNOArch([Key("X", size=90) ],\
    fno_modes = 13, dimension=1,padding=20,nr_fno_layers=5, decoder_net=decoder4)
    
    inputs = [
        "perm",
        "Phi",        
        "Pini",
        "Swini",
        "pressure", 
        "water_sat",
        "gas_sat",
        "fault",         
    ]
    
    # darcyy = Node(
    #     inputs = inputs,        
    #     outputs = [
    #         "pressured",
    #         "saturationd",
    #         "saturationdg",
    #     ],
        
    #     evaluate = Black_oil(neededM,SWI,SWR,UW,BW,UO,BO,
    #                   nx,ny,nz,SWOW,SWOG,target_min,target_max,minKx,maxKx,
    #                   minPx,maxPx,p_bub,p_atm,CFO,Relperm,params,pde_method,RE,max_inn_fcnx,max_out_fcnx,DZ),
    #     name="Darcy node",
    # )
    
    inputs = [
        "X",
        "Y",                
    ]
    
    peacemannp = Node(
        inputs = inputs,        
        outputs = [
            "peacemanned",
        ],
        
        evaluate =  Black_oil_peacemann(UO,BO,UW,BW,DZ,RE,device,max_inn_fcnx,
                                    max_out_fcnx,params,p_bub, p_atm,steppi,CFO),
        name="Peacemann node",
    )



    nodes =[fno_pressure.make_node('fno_forward_model_pressure')] +\
    [fno_peacemann.make_node('fno_forward_model_peacemann')] +\
    [fno_water.make_node('fno_forward_model_water')]+\
    [fno_gas.make_node('fno_forward_model_gas')] + [peacemannp]#+ [darcyy]
    
    #nodes = [fno_pressure.make_node('fno_forward_model_presssure')] 
    # [constraint]
    # make domain
    domain = Domain()


    # add constraints to domain
    supervised_dynamic_pressure = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_pressure,
        batch_size=1,
    )
    domain.add_constraint(supervised_dynamic_pressure, "supervised_dynamic_pressure")
    
    supervised_dynamic_water = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_water,
        batch_size=1,
    )
    domain.add_constraint(supervised_dynamic_water, "supervised_dynamic_water")


    supervised_dynamic_gas = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_gas,
        batch_size=1,
    )
    domain.add_constraint(supervised_dynamic_gas, "supervised_dynamic_gas")

    
    supervised_peacemann = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_p,
        batch_size=1,
    )

    domain.add_constraint(supervised_peacemann, "supervised_peacemann")
    

    # [constraint]    
    # add validator

    test_dynamic_pressure = GridValidator(
        nodes,
        dataset=test_dataset_pressure,
        batch_size=cfg.batch_size.test,
        requires_grad=False,
    )   
    domain.add_validator(test_dynamic_pressure, "test_dynamic_pressure")
    

    test_dynamic_water = GridValidator(
        nodes,
        dataset=test_dataset_water,
        batch_size=cfg.batch_size.test,
        requires_grad=False,
    )   
    domain.add_validator(test_dynamic_water, "test_dynamic_water")
    
    
    test_dynamic_gas = GridValidator(
        nodes,
        dataset=test_dataset_gas,
        batch_size=cfg.batch_size.test,
        requires_grad=False,
    )   
    domain.add_validator(test_dynamic_gas, "test_dynamic_gas")
    

    test_peacemann = GridValidator(
        nodes,
        dataset=test_dataset_p,
        batch_size=cfg.batch_size.test,
        requires_grad=False,
    )
    domain.add_validator(test_peacemann, "test_peaceman")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()


