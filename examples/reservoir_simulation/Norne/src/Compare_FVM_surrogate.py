# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import modulus
from modulus.sym.hydra import to_absolute_path
from modulus.sym.key import Key
from NVRS import *
from modulus.sym.models.fno import *
import pandas as pd
from multiprocessing import Lock, Value
from PIL import Image
import requests
import concurrent.futures


def read_yaml(fname):
    """Read Yaml file into a dict of parameters"""
    print(f"Read simulation plan from {fname}...")
    with open(fname, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            # print(data)
        except yaml.YAMLError as exc:
            print(exc)
        return data


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "confirm": 1}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


def process_chunk(chunk):
    chunk_results = []
    for kk in chunk:
        result = process_step(kk)
        chunk_results.append(result)

    with lock:
        Runs = len(chunks)
        processed_chunks.value += 1
        completion_percentage = (processed_chunks.value / len(chunks)) * 100
        remaining_percentage = 100 - completion_percentage
        # print(f"Processed chunk {processed_chunks.value}. {completion_percentage:.2f}% completed. {remaining_percentage:.2f}% remaining.")
        progressBar = "\rPlotting Progress: " + ProgressBar(
            Runs, processed_chunks.value, Runs
        )
        ShowBar(progressBar)
        # time.sleep(1)

    return chunk_results


def process_step(kk):

    current_time = dt[kk]

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = ((pressure[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]

    lookf = ((pressure_true[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    # lookf = lookf * pini_alt
    diff1 = (abs(look - lookf) * effectiveuse)[:, :, ::-1]

    ax1 = f_3.add_subplot(4, 3, 1, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 2, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "pressure Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 3, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "pressure diff",
        injectors,
        producers,
        gass,
    )
    R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())

    look = ((Swater[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    lookf = ((Swater_true[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * effectiveuse)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 4, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 5, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "water Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 6, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "water diff",
        injectors,
        producers,
        gass,
    )
    R2w, L2w = compute_metrics(look.ravel(), lookf.ravel())

    look = Soil[0, kk, :, :, :]
    look = (look * effectiveuse)[:, :, ::-1]
    lookf = Soil_true[0, kk, :, :, :]
    lookf = (lookf * effectiveuse)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * effectiveuse)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 7, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "oil Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 8, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "oil Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 9, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "oil diff",
        injectors,
        producers,
        gass,
    )
    R2o, L2o = compute_metrics(look.ravel(), lookf.ravel())

    look = (((Sgas[0, kk, :, :, :])) * effectiveuse)[:, :, ::-1]
    lookf = (((Sgas_true[0, kk, :, :, :])) * effectiveuse)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * effectiveuse)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 10, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 11, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "gas Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 12, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "gas diff",
        injectors,
        producers,
        gass,
    )
    R2g, L2g = compute_metrics(look.ravel(), lookf.ravel())

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()

    return current_time, (R2p, L2p), (R2w, L2w), (R2o, L2o), (R2g, L2g)


oldfolder = os.getcwd()
os.chdir(oldfolder)


if not os.path.exists("../COMPARE_RESULTS"):
    os.makedirs("../COMPARE_RESULTS")

print("")
surrogate = 1

print("")
Trainmoe = 2
print("-----------------------------------------------------------------")
print(
    "Using Cluster Classify Regress (CCR) for peacemann model -Hard Prediction          "
)
print("")
print("References for CCR include: ")
print(
    " (1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\n\
Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\n\
method for learning discontinuous functions.Foundations of Data Science,\n\
1(2639-8001-2019-4-491):491, 2019.\n"
)
print("")
print(
    "(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\n\
Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.\n"
)
print("-----------------------------------------------------------------------")
# pred_type=int(input('Choose: 1=Hard Prediction, 2= Soft Prediction: '))
pred_type = 1

folderr = "../COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION"
if not os.path.exists("../COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION"):
    os.makedirs("../COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION")
else:
    shutil.rmtree("../COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION")
    os.makedirs("../COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION")

degg = 3
num_cores = 6
print("")


fname = "conf/config_PINO.yaml"


mat = sio.loadmat("../PACKETS/conversions.mat")
minK = mat["minK"]
maxK = mat["maxK"]
minT = mat["minT"]
maxT = mat["maxT"]
minP = mat["minP"]
maxP = mat["maxP"]
minQw = mat["minQW"]
maxQw = mat["maxQW"]
minQg = mat["minQg"]
maxQg = mat["maxQg"]
minQ = mat["minQ"]
maxQ = mat["maxQ"]
min_inn_fcn = mat["min_inn_fcn"]
max_inn_fcn = mat["max_inn_fcn"]
min_out_fcn = mat["min_out_fcn"]
max_out_fcn = mat["max_out_fcn"]
steppi = int(mat["steppi"])
# print(steppi)
steppi_indices = mat["steppi_indices"].flatten()

target_min = 0.01
target_max = 1


plan = read_yaml(fname)
nx = plan["custom"]["PROPS"]["nx"]
ny = plan["custom"]["PROPS"]["ny"]
nz = plan["custom"]["PROPS"]["nz"]
Ne = 1
perm_ensemble = np.genfromtxt("../NORNE/sgsim.out")
poro_ensemble = np.genfromtxt("../NORNE/sgsimporo.out")
fault_ensemble = np.genfromtxt("../NORNE/faultensemble.dat")
# index = np.random.choice(perm_ensemble.shape[1], Ne, \
#                          replace=False)
index = 20
perm_use = perm_ensemble[:, index].reshape(-1, 1)
poro_use = poro_ensemble[:, index].reshape(-1, 1)
fault_use = fault_ensemble[:, index].reshape(-1, 1)
effective = np.genfromtxt("../NORNE/actnum.out", dtype="float")

effectiveuse = np.reshape(effective, (nx, ny, nz), "F")

injectors = plan["custom"]["WELLSPECS"]["water_injector_wells"]
producers = plan["custom"]["WELLSPECS"]["producer_wells"]
gass = plan["custom"]["WELLSPECS"]["gas_injector_wells"]

N_injw = len(
    plan["custom"]["WELLSPECS"]["water_injector_wells"]
)  # Number of water injectors
N_pr = len(plan["custom"]["WELLSPECS"]["producer_wells"])  # Number of producers
N_injg = len(
    plan["custom"]["WELLSPECS"]["gas_injector_wells"]
)  # Number of gas injectors
string_Jesus = "flow FULLNORNE.DATA --parsing-strictness=low"
string_Jesus2 = "flow FULLNORNE2.DATA --parsing-strictness=low"
oldfolder2 = os.getcwd()
path_out = "../True_Flow"
os.makedirs(path_out, exist_ok=True)
copy_files("../Necessaryy", path_out)
save_files(perm_use, poro_use, fault_use, path_out, oldfolder2)

print("")
print("---------------------------------------------------------------------")
print("")
print("\n")
print("|-----------------------------------------------------------------|")
print("|                 RUN FLOW SIMULATOR                              |")
print("|-----------------------------------------------------------------|")
print("")
start_time_plots1 = time.time()
Run_simulator(path_out, oldfolder2, string_Jesus, string_Jesus2)
elapsed_time_secs = (time.time() - start_time_plots1) / 2
msg = "Reservoir simulation with FLOW  took: %s secs (Wall clock time)" % timedelta(
    seconds=round(elapsed_time_secs)
)
print(msg)
print("")
print("Finished FLOW NUMERICAL simulations")

print("|-----------------------------------------------------------------|")
print("|                 DATA CURRATION IN PROCESS                       |")
print("|-----------------------------------------------------------------|")
N = Ne
# steppi = 246
check = np.ones((nx, ny, nz), dtype=np.float16)
pressure = []
Sgas = []
Swater = []
Time = []

permeability = np.zeros((N, 1, nx, ny, nz))
porosity = np.zeros((N, 1, nx, ny, nz))
actnumm = np.zeros((N, 1, nx, ny, nz))


folder = path_out
Pr, sw, sg, tt, _ = Geta_all(
    folder,
    nx,
    ny,
    nz,
    effective,
    oldfolder2,
    check,
    string_Jesus,
    steppi,
    steppi_indices,
)
pressure.append(Pr)
Sgas.append(sg)
Swater.append(sw)
Time.append(tt)

permeability[0, 0, :, :, :] = np.reshape(perm_ensemble[:, index], (nx, ny, nz), "F")
porosity[0, 0, :, :, :] = np.reshape(poro_ensemble[:, index], (nx, ny, nz), "F")
actnumm[0, 0, :, :, :] = np.reshape(effective, (nx, ny, nz), "F")


pressure_true = np.stack(pressure, axis=0)
Sgas_true = np.stack(Sgas, axis=0)
Swater_true = np.stack(Swater, axis=0)
Soil_true = 1 - (Sgas_true + Swater_true)
Time = np.stack(Time, axis=0)


# os.chdir('../NORNE')
# Time = Get_Time(nx,ny,nz,steppi,steppi_indices,Ne)
# os.chdir(oldfolder)


_, out_fcn_true = Get_data_FFNN1(
    folder,
    oldfolder2,
    N,
    pressure_true,
    Sgas_true,
    Swater_true,
    permeability,
    Time,
    steppi,
    steppi_indices,
)

folderrr = "../True_Flow"
rmtree(folderrr)

print("")
print("---------------------------------------------------------------------")
print("")
print("\n")
print("|-----------------------------------------------------------------|")
print("|          RUN  NVIDIA MODULUS RESERVOIR SIMULATION SURROGATE     |")
print("|-----------------------------------------------------------------|")
print("")


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No GPU found. Please run on a system with a GPU.")
inn = ensemble_pytorch(
    perm_use,
    poro_use,
    fault_use,
    nx,
    ny,
    nz,
    Ne,
    effectiveuse,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
)


print("")
print("Finished constructing Pytorch inputs")


print("*******************Load the trained Forward models*******************")

decoder1 = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
fno_pressure = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
        Key("fault", size=1),
    ],
    dimension=3,
    decoder_net=decoder1,
)

decoder2 = ConvFullyConnectedArch([Key("z", size=32)], [Key("water_sat", size=steppi)])
fno_water = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
        Key("fault", size=1),
    ],
    dimension=3,
    decoder_net=decoder2,
)

decoder3 = ConvFullyConnectedArch([Key("z", size=32)], [Key("gas_sat", size=steppi)])
fno_gas = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
        Key("fault", size=1),
    ],
    dimension=3,
    decoder_net=decoder3,
)


decoder4 = ConvFullyConnectedArch([Key("z", size=32)], [Key("Y", size=66)])
fno_peacemann = FNOArch(
    [Key("X", size=90)],
    fno_modes=13,
    dimension=1,
    padding=20,
    nr_fno_layers=5,
    decoder_net=decoder4,
)


print("-----------------Surrogate Model learned with PINO----------------")
if not os.path.exists(("outputs/Forward_problem_PINO/ResSim/")):
    os.makedirs(("outputs/Forward_problem_PINO/ResSim/"))
else:
    pass

bb = os.path.isfile(
    "outputs/Forward_problem_PINO/ResSim/fno_forward_model_pressure.0.pth"
)
if bb == False:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1lyojCxW4aHKVm5XM66zKWdW6W8aYbO2F",
        "outputs/Forward_problem_PINO/ResSim/fno_forward_model_pressure.0.pth",
    )
    print("...Downlaod completed.......")

    os.chdir("outputs/Forward_problem_PINO/ResSim")

    fno_pressure.load_state_dict(torch.load("fno_forward_model_pressure.0.pth"))
    fno_pressure = fno_pressure.to(device)
    fno_pressure.eval()
    os.chdir(oldfolder)
else:

    os.chdir("outputs/Forward_problem_PINO/ResSim")
    print(" Surrogate model learned with PINO for dynamic properties pressure model")
    fno_pressure.load_state_dict(torch.load("fno_forward_model_pressure.0.pth"))
    fno_pressure = fno_pressure.to(device)
    fno_pressure.eval()
    os.chdir(oldfolder)

bb = os.path.isfile("outputs/Forward_problem_PINO/ResSim/fno_forward_model_water.0.pth")
if bb == False:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1QH4QnkwSwfoMgAqp_MxrVVt25ylbzUqv",
        "outputs/Forward_problem_PINO/ResSim/fno_forward_model_water.0.pth",
    )
    print("...Downlaod completed.......")

    os.chdir("outputs/Forward_problem_PINO/ResSim")

    fno_water.load_state_dict(torch.load("fno_forward_model_water.0.pth"))
    fno_water = fno_water.to(device)
    fno_water.eval()
    os.chdir(oldfolder)
else:

    os.chdir("outputs/Forward_problem_PINO/ResSim")
    print(" Surrogate model learned with PINO for dynamic properties- water model")
    fno_water.load_state_dict(torch.load("fno_forward_model_water.0.pth"))
    fno_water = fno_water.to(device)
    fno_water.eval()
    os.chdir(oldfolder)

bb = os.path.isfile("outputs/Forward_problem_PINO/ResSim/fno_forward_model_gas.0.pth")
if bb == False:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1QvnH4kcRSu-Q0WgzY7LPWbWkohPOvSIT",
        "outputs/Forward_problem_FNO/ResSim/fno_forward_model_gas.0.pth",
    )
    print("...Downlaod completed.......")

    os.chdir("outputs/Forward_problem_PINO/ResSim")

    fno_gas.load_state_dict(torch.load("fno_forward_model_gas.0.pth"))
    fno_gas = fno_gas.to(device)
    fno_gas.eval()
    os.chdir(oldfolder)
else:

    os.chdir("outputs/Forward_problem_PINO/ResSim")
    print(" Surrogate model learned with PINO for dynamic properties - Gas model")
    fno_gas.load_state_dict(torch.load("fno_forward_model_gas.0.pth"))
    fno_gas = fno_gas.to(device)
    fno_gas.eval()
    os.chdir(oldfolder)


bba = os.path.isfile(
    "outputs/Forward_problem_PINO/ResSim/fno_forward_model_peacemann.0.pth"
)
if bba == False:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1kyST2aMmqTAdfv-C6MI4CKd3Vi-4Ja4F",
        "outputs/Forward_problem_PINO/ResSim/fno_forward_model_peacemann.0.pth",
    )
    print("...Downlaod completed.......")
    os.chdir("outputs/Forward_problem_PINO/ResSim")

    fno_peacemann.load_state_dict(torch.load("fno_forward_model_peacemann.0.pth"))
    fno_peacemann = fno_peacemann.to(device)
    fno_peacemann.eval()
    os.chdir(oldfolder)
else:
    os.chdir("outputs/Forward_problem_PINO/ResSim")
    print(" Surrogate model learned with PINO for peacemann well model")
    fno_peacemann.load_state_dict(torch.load("fno_forward_model_peacemann.0.pth"))
    fno_peacemann = fno_peacemann.to(device)
    fno_peacemann.eval()
    os.chdir(oldfolder)

print("********************Model Loaded*************************************")
start_time_plots2 = time.time()
_, ouut_peacemann, pressure, Swater, Sgas, Soil = Forward_model_ensemble(
    Ne,
    inn,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    fno_pressure,
    fno_water,
    fno_gas,
    device,
    fno_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
)
elapsed_time_secs2 = time.time() - start_time_plots2
msg = (
    "Reservoir simulation with NVidia Modulus (CCR - Hard prediction)  took: %s secs (Wall clock time)"
    % timedelta(seconds=round(elapsed_time_secs2))
)
print(msg)
print("")


modulus_time = elapsed_time_secs2
flow_time = elapsed_time_secs


if modulus_time < flow_time:
    slower_time = modulus_time
    faster_time = flow_time
    slower = "Nvidia modulus Surrogate"
    faster = "flow Reservoir simulator"
    speedup = math.ceil(flow_time / modulus_time)
    os.chdir(folderr)
    # Data
    tasks = ["Flow", "modulus"]
    times = [faster_time, slower_time]

    # Colors
    colors = ["green", "red"]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tasks, times, color=colors)
    plt.ylabel("Time (seconds)", fontweight="bold")
    plt.title("Execution Time Comparison for moduls vs. Flow", fontweight="bold")

    # Annotate the bars with their values
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            round(yval, 2),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Indicate speedup on the chart
    plt.text(
        0.5,
        550,
        f"Speedup: {speedup}x",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="blue",
    )
    namez = "Compare_time.png"
    plt.savefig(namez)
    plt.clf()
    plt.close()
    os.chdir(oldfolder)

else:
    slower_time = flow_time
    faster_time = modulus_time
    slower = "flow Reservoir simulator"
    faster = "Nvidia modulus Surrogate"
    speedup = math.ceil(modulus_time / flow_time)
    os.chdir(folderr)
    # Data
    tasks = ["Flow", "modulus"]
    times = [slower_time, faster_time]

    # Colors
    colors = ["green", "red"]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tasks, times, color=colors)
    plt.ylabel("Time (seconds)", fontweight="bold")
    plt.title("Execution Time Comparison for moduls vs. Flow", fontweight="bold")

    # Annotate the bars with their values
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            round(yval, 2),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Indicate speedup on the chart
    plt.text(
        0.5,
        550,
        f"Speedup: {speedup}x",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="blue",
    )
    namez = "Compare_time.png"
    plt.savefig(namez)
    plt.clf()
    plt.close()
    os.chdir(oldfolder)


message = (
    f"{slower} execution took: {slower_time} seconds\n"
    f"{faster} execution took: {faster_time} seconds\n"
    f"Speedup =  {speedup}X  "
)

print(message)


os.chdir("../NORNE")
Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)

Time_unie = np.zeros((steppi))
for i in range(steppi):
    Time_unie[i] = Time[0, i, 0, 0, 0]
# shape = [batch, steppi, nz,nx,ny]
os.chdir(oldfolder)

dt = Time_unie
print("")
print("Plotting outputs")
os.chdir(folderr)

Runs = steppi
ty = np.arange(1, Runs + 1)
Time_vector = np.zeros((steppi))
Accuracy_presure = np.zeros((steppi, 2))
Accuracy_oil = np.zeros((steppi, 2))
Accuracy_water = np.zeros((steppi, 2))
Accuracy_gas = np.zeros((steppi, 2))


# lock = Lock()
# processed_chunks = Value('i', 0)

for kk in range(steppi):
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    current_time = dt[kk]
    Time_vector[kk] = current_time

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = ((pressure[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]

    lookf = ((pressure_true[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    # lookf = lookf * pini_alt
    diff1 = (abs(look - lookf) * effectiveuse)[:, :, ::-1]

    ax1 = f_3.add_subplot(4, 3, 1, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 2, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "pressure Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 3, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "pressure diff",
        injectors,
        producers,
        gass,
    )
    R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_presure[kk, 0] = R2p
    Accuracy_presure[kk, 1] = L2p

    look = ((Swater[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    lookf = ((Swater_true[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * effectiveuse)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 4, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 5, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "water Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 6, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "water diff",
        injectors,
        producers,
        gass,
    )
    R2w, L2w = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_water[kk, 0] = R2w
    Accuracy_water[kk, 1] = L2w

    look = Soil[0, kk, :, :, :]
    look = (look * effectiveuse)[:, :, ::-1]
    lookf = Soil_true[0, kk, :, :, :]
    lookf = (lookf * effectiveuse)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * effectiveuse)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 7, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "oil Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 8, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "oil Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 9, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "oil diff",
        injectors,
        producers,
        gass,
    )
    R2o, L2o = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_oil[kk, 0] = R2o
    Accuracy_oil[kk, 1] = L2o

    look = (((Sgas[0, kk, :, :, :])) * effectiveuse)[:, :, ::-1]
    lookf = (((Sgas_true[0, kk, :, :, :])) * effectiveuse)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * effectiveuse)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 10, projection="3d")
    Plot_Modulus(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas Modulus",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 11, projection="3d")
    Plot_Modulus(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "gas Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 12, projection="3d")
    Plot_Modulus(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "gas diff",
        injectors,
        producers,
        gass,
    )
    R2g, L2g = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_gas[kk, 0] = R2g
    Accuracy_gas[kk, 1] = L2g

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()

progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
ShowBar(progressBar)
time.sleep(1)


# NUM_CORES = 6  # specify the number of cores you want to use

# # Split the range of steps into chunks
# if steppi < NUM_CORES:
#     num_cores = steppi
# else:
#     num_cores = NUM_CORES
# chunks = [
# list(range(i, min(i + steppi // num_cores, steppi)))
# for i in range(0, steppi, steppi // num_cores)
# ]


# with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#     chunked_results = list(executor.map(process_chunk, chunks))


# # Flatten the chunked results to get the ordered results
# results = [result for sublist in chunked_results for result in sublist]


# for kk, (current_time, acc_pressure, acc_oil, acc_water,acc_gas) in enumerate(results):
#     Time_vector[kk] = current_time
#     Accuracy_presure[kk] = acc_pressure
#     Accuracy_oil[kk] = acc_oil
#     Accuracy_water[kk] = acc_water
#     Accuracy_water[kk] = acc_gas


fig4 = plt.figure(figsize=(20, 20), dpi=100)
font = FontProperties()
font.set_family("Helvetica")
font.set_weight("bold")

fig4.text(
    0.5,
    0.98,
    "R2(%) Accuracy - Modulus/Numerical(GPU)",
    ha="center",
    va="center",
    fontproperties=font,
    fontsize=11,
)
fig4.text(
    0.5,
    0.49,
    "L2(%) Accuracy - Modulus/Numerical(GPU)",
    ha="center",
    va="center",
    fontproperties=font,
    fontsize=11,
)

# Plot R2 accuracies
plt.subplot(2, 4, 1)
plt.plot(
    Time_vector,
    Accuracy_presure[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("Pressure", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

plt.subplot(2, 4, 2)
plt.plot(
    Time_vector,
    Accuracy_water[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("water saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

plt.subplot(2, 4, 3)
plt.plot(
    Time_vector,
    Accuracy_oil[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("oil saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

plt.subplot(2, 4, 4)
plt.plot(
    Time_vector,
    Accuracy_gas[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("gas saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

# Plot L2 accuracies
plt.subplot(2, 4, 5)
plt.plot(
    Time_vector,
    Accuracy_presure[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("Pressure", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)

plt.subplot(2, 4, 6)
plt.plot(
    Time_vector,
    Accuracy_water[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("water saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)

plt.subplot(2, 4, 7)
plt.plot(
    Time_vector,
    Accuracy_oil[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("oil saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)

plt.subplot(2, 4, 8)
plt.plot(
    Time_vector,
    Accuracy_gas[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("gas saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)


plt.tight_layout(rect=[0, 0.05, 1, 0.93])
namez = "R2L2.png"
plt.savefig(namez)
plt.clf()
plt.close()


print("")
print("Now - Creating GIF")
import glob

import re

frames = []
imgs = sorted(glob.glob("*Dynamic*"), key=sort_key)

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "Evolution.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)


from glob import glob

for f3 in glob("*Dynamic*"):
    os.remove(f3)

# os.chdir(oldfolder)


print("")
print("Saving prediction in CSV file")
write_RSM(ouut_peacemann[0, :, :66], Time_vector, "Modulus")
write_RSM(out_fcn_true[0, :, :66], Time_vector, "Flow")

CCRhard = ouut_peacemann[0, :, :66]
Truedata = out_fcn_true[0, :, :66]

print("")
print("Plotting well responses and accuracies")
# Plot_R2(Accuracy_presure,Accuracy_water,Accuracy_oil,Accuracy_gas,steppi,Time_vector)
Plot_RSM_percentile(ouut_peacemann[0, :, :66], out_fcn_true[0, :, :66], Time_vector)
os.chdir(oldfolder)


print("")
Trainmoe = 1
print("----------------------------------------------------------------------")
print("Using FNO for peacemann model           ")
print("")

pred_type = 1

folderr = "../COMPARE_RESULTS/PINO/PEACEMANN_FNO"
if not os.path.exists("../COMPARE_RESULTS/PINO/PEACEMANN_FNO"):
    os.makedirs("../COMPARE_RESULTS/PINO/PEACEMANN_FNO")
else:
    shutil.rmtree("../COMPARE_RESULTS/PINO/PEACEMANN_FNO")
    os.makedirs("../COMPARE_RESULTS/PINO/PEACEMANN_FNO")

# Specify the filename you want to copy
source_directory = "../COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION"
destination_directory = "../COMPARE_RESULTS/PINO/PEACEMANN_FNO"
filename = "Evolution.gif"

# Construct the full paths for source and destination
source_path = os.path.join(source_directory, filename)
destination_path = os.path.join(destination_directory, filename)

# Perform the copy operation
shutil.copy(source_path, destination_path)

# Construct the full paths for source and destination
filename = "R2L2.png"
source_path = os.path.join(source_directory, filename)
destination_path = os.path.join(destination_directory, filename)

# Perform the copy operation
shutil.copy(source_path, destination_path)


start_time_plots2 = time.time()
_, ouut_peacemann, pressure, Swater, Sgas, Soil = Forward_model_ensemble(
    Ne,
    inn,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    fno_pressure,
    fno_water,
    fno_gas,
    device,
    fno_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
)
elapsed_time_secs2 = time.time() - start_time_plots2
msg = (
    "Reservoir simulation with NVIDIA Modulus (FNO)  took: %s secs (Wall clock time)"
    % timedelta(seconds=round(elapsed_time_secs2))
)
print(msg)
print("")


modulus_time = elapsed_time_secs2
flow_time = elapsed_time_secs

if modulus_time < flow_time:
    slower_time = modulus_time
    faster_time = flow_time
    slower = "Nvidia modulus Surrogate"
    faster = "flow Reservoir simulator"
    speedup = math.ceil(flow_time / modulus_time)
    os.chdir(folderr)
    # Data
    tasks = ["Flow", "modulus"]
    times = [faster_time, slower_time]

    # Colors
    colors = ["green", "red"]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tasks, times, color=colors)
    plt.ylabel("Time (seconds)", fontweight="bold")
    plt.title("Execution Time Comparison for moduls vs. Flow", fontweight="bold")

    # Annotate the bars with their values
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            round(yval, 2),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Indicate speedup on the chart
    plt.text(
        0.5,
        550,
        f"Speedup: {speedup}x",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="blue",
    )
    namez = "Compare_time.png"
    plt.savefig(namez)
    plt.clf()
    plt.close()
    os.chdir(oldfolder)

else:
    slower_time = flow_time
    faster_time = modulus_time
    slower = "flow Reservoir simulator"
    faster = "Nvidia modulus Surrogate"
    speedup = math.ceil(modulus_time / flow_time)
    os.chdir(folderr)
    # Data
    tasks = ["Flow", "modulus"]
    times = [slower_time, faster_time]

    # Colors
    colors = ["green", "red"]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tasks, times, color=colors)
    plt.ylabel("Time (seconds)", fontweight="bold")
    plt.title("Execution Time Comparison for moduls vs. Flow", fontweight="bold")

    # Annotate the bars with their values
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            round(yval, 2),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Indicate speedup on the chart
    plt.text(
        0.5,
        550,
        f"Speedup: {speedup}x",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="blue",
    )
    namez = "Compare_time.png"
    plt.savefig(namez)
    plt.clf()
    plt.close()
    os.chdir(oldfolder)


message = (
    f"{slower} execution took: {slower_time} seconds\n"
    f"{faster} execution took: {faster_time} seconds\n"
    f"Speedup =  {speedup}X  "
)

print(message)


os.chdir("../NORNE")
Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)

Time_unie = np.zeros((steppi))
for i in range(steppi):
    Time_unie[i] = Time[0, i, 0, 0, 0]
# shape = [batch, steppi, nz,nx,ny]
os.chdir(oldfolder)

dt = Time_unie
print("")
print("Plotting outputs")
os.chdir(folderr)

Runs = steppi
ty = np.arange(1, Runs + 1)
Time_vector = np.zeros((steppi))


# lock = Lock()
# processed_chunks = Value('i', 0)

for kk in range(steppi):
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    current_time = dt[kk]
    Time_vector[kk] = current_time


progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
ShowBar(progressBar)
time.sleep(1)


print("")
print("Saving prediction in CSV file")
write_RSM(ouut_peacemann[0, :, :66], Time_vector, "Modulus")
write_RSM(out_fcn_true[0, :, :66], Time_vector, "Flow")

print("")
print("Plotting well responses and accuracies")
# Plot_R2(Accuracy_presure,Accuracy_water,Accuracy_oil,Accuracy_gas,steppi,Time_vector)
Plot_RSM_percentile(ouut_peacemann[0, :, :66], out_fcn_true[0, :, :66], Time_vector)
FNOpred = ouut_peacemann[0, :, :66]
os.chdir(oldfolder)

os.chdir("../COMPARE_RESULTS")
columns = [
    "B_1BH",
    "B_1H",
    "B_2H",
    "B_3H",
    "B_4BH",
    "B_4DH",
    "B_4H",
    "D_1CH",
    "D_1H",
    "D_2H",
    "D_3AH",
    "D_3BH",
    "D_4AH",
    "D_4H",
    "E_1H",
    "E_2AH",
    "E_2H",
    "E_3AH",
    "E_3CH",
    "E_3H",
    "E_4AH",
    "K_3H",
]


# timezz = True_mat[:,0].reshape(-1,1)

P10 = CCRhard
P90 = FNOpred


True_mat = Truedata
timezz = Time_vector

plt.figure(figsize=(40, 40))

for k in range(22):
    plt.subplot(5, 5, int(k + 1))
    plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="Flow")
    plt.plot(timezz, P10[:, k], color="blue", lw="2", label="PINO -CCR(hard)")
    plt.plot(timezz, P90[:, k], color="orange", lw="2", label="PINO - FNO")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title(columns[k], fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

# os.chdir('RESULTS')
plt.savefig(
    "Oil.png"
)  # save as png                                  # preventing the figures from showing
# os.chdir(oldfolder)
plt.clf()
plt.close()


plt.figure(figsize=(40, 40))

for k in range(22):
    plt.subplot(5, 5, int(k + 1))
    plt.plot(timezz, True_mat[:, k + 22], color="red", lw="2", label="Flow")
    plt.plot(timezz, P10[:, k + 22], color="blue", lw="2", label="PINO -CCR(hard)")
    plt.plot(timezz, P90[:, k + 22], color="orange", lw="2", label="PINO - FNO")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title(columns[k], fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

# os.chdir('RESULTS')
plt.savefig(
    "Water.png"
)  # save as png                                  # preventing the figures from showing
# os.chdir(oldfolder)
plt.clf()
plt.close()


plt.figure(figsize=(40, 40))

for k in range(22):
    plt.subplot(5, 5, int(k + 1))
    plt.plot(timezz, True_mat[:, k + 44], color="red", lw="2", label="Flow")
    plt.plot(timezz, P10[:, k + 44], color="blue", lw="2", label="PINO -CCR(hard)")
    plt.plot(timezz, P90[:, k + 44], color="orange", lw="2", label="PINO - FNO")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title(columns[k], fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

# os.chdir('RESULTS')
plt.savefig(
    "Gas.png"
)  # save as png                                  # preventing the figures from showing
# os.chdir(oldfolder)
plt.clf()
plt.close()


# Define your P10
True_data = np.reshape(Truedata, (-1, 1), "F")

CCRhard = np.reshape(CCRhard, (-1, 1), "F")
cc1 = ((np.sum((((CCRhard) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
print("RMSE of PINO - CCR (hard prediction) reservoir model  =  " + str(cc1))


FNO = np.reshape(FNOpred, (-1, 1), "F")
cc3 = ((np.sum((((FNO) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
print("RMSE of  PINO - FNO reservoir model  =  " + str(cc3))


plt.figure(figsize=(10, 10))
values = [cc1, cc3]
model_names = ["PINO CCR-hard", "PINO - FNO"]
colors = ["b", "orange"]


# Find the index of the minimum RMSE value
min_rmse_index = np.argmin(values)

# Get the minimum RMSE value
min_rmse = values[min_rmse_index]

# Get the corresponding model name
best_model = model_names[min_rmse_index]

# Print the minimum RMSE value and its corresponding model name
print(f"The minimum RMSE value = {min_rmse}")
print(f"Recommended reservoir forward model workflow = {best_model} reservoir model.")

# Create a histogram
plt.bar(model_names, values, color=colors)
plt.xlabel("Reservoir Models")
plt.ylabel("RMSE")
plt.title("Histogram of RMSE Values for Different Reservoir Surrogate Model workflow")
plt.legend(model_names)
plt.savefig(
    "Histogram.png"
)  # save as png                                  # preventing the figures from showing
plt.clf()
plt.close()
os.chdir(oldfolder)
print("")
print("-------------------PROGRAM EXECUTED-----------------------------------")
