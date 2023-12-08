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

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.stats import rankdata, norm
from scipy import interpolate
import matplotlib.pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import datetime
import multiprocessing
import os
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
import numpy
import gzip
from copy import copy
from joblib import Parallel, delayed
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import os
from kneed import KneeLocator
import scipy.io as sio
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# import gen
# from gen import indeces_K_cut, GenMat, test_points_gen
import warnings

warnings.filterwarnings("ignore")

oldfolder = os.getcwd()
cores = multiprocessing.cpu_count()
print(" ")
print(" This computer has %d cores, which will all be utilised in parallel " % cores)
# print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
print(" ")

start = datetime.datetime.now()
print(str(start))

print("-------------------LOAD FUNCTIONS-------------------------------------")


def interpolatebetween(xtrain, cdftrain, xnew):
    numrows1 = len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2 = np.zeros((numrows1, numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:, i]), cdftrain[:, i], kind="linear")
        cdftest = f(xnew[:, i])
        norm_cdftest2[:, i] = np.ravel(cdftest)
    return norm_cdftest2


# Combinatorials funcs


def gaussianizeit(input1):
    numrows1 = len(input1)
    numcols = len(input1[0])
    newbig = np.zeros((numrows1, numcols))
    for i in range(numcols):
        input11 = input1[:, i]
        newX = norm.ppf(rankdata(input11) / (len(input11) + 1))
        newbig[:, i] = newX.T
    return newbig


def getoptimumk(X, i, training_master, oldfolder):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("Elbow Method showing the optimal n_clusters for machine %d" % (i))
    os.chdir(training_master)
    plt.savefig("machine_%d.jpg" % (i + 1))
    os.chdir(oldfolder)
    # plt.show()
    plt.close()
    plt.clf()
    return kuse


def getoptimumkcost(X, i, training_master, oldfolder):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("Elbow Method showing the optimal n_clusters for machine %d" % (i))
    os.chdir(training_master)
    plt.savefig("machine_Energy__%d.jpg" % (i + 1))
    os.chdir(oldfolder)
    plt.show()
    return kuse


def best_fit(X, Y):

    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)  # or len(Y)
    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar

    print("best fit line:\ny = {:.2f} + {:.2f}x".format(a, b))
    return a, b


def Performance_plot_cost(CCR, Trued, stringg, training_master, oldfolder):

    CoDview = np.zeros((1, Trued.shape[1]))
    R2view = np.zeros((1, Trued.shape[1]))

    plt.figure(figsize=(40, 40))

    for jj in range(Trued.shape[1]):
        print(" Compute L2 and R2 for the machine _" + str(jj + 1))

        clementanswer2 = np.reshape(CCR[:, jj], (-1, 1))
        outputtest2 = np.reshape(Trued[:, jj], (-1, 1))
        numrowstest = len(outputtest2)
        outputtest2 = np.reshape(outputtest2, (-1, 1))
        Lerrorsparse = (
            LA.norm(outputtest2 - clementanswer2) / LA.norm(outputtest2)
        ) ** 0.5
        L_22 = 1 - (Lerrorsparse**2)
        # Coefficient of determination
        outputreq = np.zeros((numrowstest, 1))
        for i in range(numrowstest):
            outputreq[i, :] = outputtest2[i, :] - np.mean(outputtest2)
        CoDspa = 1 - (LA.norm(outputtest2 - clementanswer2) / LA.norm(outputreq))
        CoD2 = 1 - (1 - CoDspa) ** 2
        print("")

        CoDview[:, jj] = CoD2
        R2view[:, jj] = L_22

        jk = jj + 1
        plt.subplot(9, 9, jk)
        palette = copy(plt.get_cmap("inferno_r"))
        palette.set_under("white")  # 1.0 represents not transparent
        palette.set_over("black")  # 1.0 represents not transparent
        vmin = min(np.ravel(outputtest2))
        vmax = max(np.ravel(outputtest2))
        sc = plt.scatter(
            np.ravel(clementanswer2),
            np.ravel(outputtest2),
            c=np.ravel(outputtest2),
            vmin=vmin,
            vmax=vmax,
            s=35,
            cmap=palette,
        )
        plt.colorbar(sc)
        plt.title("Energy_" + str(jj), fontsize=9)
        plt.ylabel("Machine", fontsize=9)
        plt.xlabel("True data", fontsize=9)
        a, b = best_fit(
            np.ravel(clementanswer2),
            np.ravel(outputtest2),
        )
        yfit = [a + b * xi for xi in np.ravel(clementanswer2)]
        plt.plot(np.ravel(clementanswer2), yfit, color="r")
        plt.annotate(
            "R2= %.3f" % CoD2,
            (0.8, 0.2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=9,
        )

    CoDoverall = (np.sum(CoDview, axis=1)) / Trued.shape[1]
    R2overall = (np.sum(R2view, axis=1)) / Trued.shape[1]
    os.chdir(training_master)
    plt.savefig("%s.jpg" % stringg)
    os.chdir(oldfolder)
    return CoDoverall, R2overall, CoDview, R2view


def run_model(model, inn, ouut, i, training_master, oldfolder):
    model.fit(inn, ouut)
    filename = "Classifier_%d.bin" % i
    os.chdir(training_master)
    model.save_model(filename)
    os.chdir(oldfolder)
    return model


def run_modelcost(model, inn, ouut, i, training_master, oldfolder):
    model.fit(inn, ouut)
    filename = "Classifiercost_%d.bin" % i
    os.chdir(training_master)
    model.save_model(filename)
    os.chdir(oldfolder)
    return model


def startit(i, outpuut2, inpuut2, training_master, oldfolder, degg):
    print("")
    print("Starting CCR training machine %d" % (i + 1))
    useeo = outpuut2[:, i]
    useeo = np.reshape(useeo, (-1, 1), "F")

    usein = inpuut2
    usein = np.reshape(usein, (-1, 90), "F")  # 9+4

    clust = CCR_Machine(usein, useeo, i, training_master, oldfolder, degg)

    bigs = clust
    return bigs
    print("")
    print("Finished training machine %d" % (i + 1))


def endit(i, testt, training_master, oldfolder, pred_type, degg):
    print("")
    print("Starting prediction from machine %d" % (i + 1))

    numcols = len(testt[0])
    clemzz = PREDICTION_CCR__MACHINE(
        i, big, testt, numcols, training_master, oldfolder, pred_type, degg
    )

    print("")
    print("Finished Prediction from machine %d" % (i + 1))
    return clemzz


def fit_machine(a0, b0, deg):

    # dim = a0.shape[1]
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly = poly_features.fit_transform(a0)

    # Fit a linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, b0)
    return lin_reg, poly_features


def predict_machine(a0, deg, model, model2):
    x_new_poly = model2.transform(a0)
    y_new = model.predict(x_new_poly)
    return y_new


def CCR_Machine(inpuutj, outputtj, ii, training_master, oldfolder, degg):
    # print('Starting CCR')
    model = xgb.XGBClassifier(n_estimators=1000)
    # import numpy as np
    # import pickle
    X = inpuutj
    y = outputtj
    numruth = len(X[0])

    y_traind = y
    scaler1a = MinMaxScaler(feature_range=(0, 1))
    (scaler1a.fit(X))
    X = scaler1a.transform(X)
    scaler2a = MinMaxScaler(feature_range=(0, 1))
    (scaler2a.fit(y))
    y = scaler2a.transform(y)
    yruth = y
    os.chdir(training_master)
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    pickle.dump(scaler1a, open(filenamex, "wb"))
    pickle.dump(scaler2a, open(filenamey, "wb"))
    os.chdir(oldfolder)
    y_traind = numruth * 10 * y
    matrix = np.concatenate((X, y_traind), axis=1)
    k = getoptimumk(matrix, ii, training_master, oldfolder)
    nclusters = k
    # nclusters=3
    print("Optimal k is: ", nclusters)
    kmeans = MiniBatchKMeans(n_clusters=nclusters, max_iter=2000).fit(matrix)
    filename = "Clustering_%d.asv" % ii
    os.chdir(training_master)
    pickle.dump(kmeans, open(filename, "wb"))
    os.chdir(oldfolder)
    dd = kmeans.labels_
    dd = dd.T
    dd = np.reshape(dd, (-1, 1))
    # -------------------#---------------------------------#
    inputtrainclass = X
    outputtrainclass = np.reshape(dd, (-1, 1))
    run_model(model, inputtrainclass, outputtrainclass, ii, training_master, oldfolder)
    # print('Split for classifier problem')
    X_train = X
    y_train = dd
    # -------------------Regression----------------#
    # print('Learn regression of the clusters with different labels from k-means ' )

    for i in range(nclusters):
        print("-- Learning cluster: " + str(i + 1) + " | " + str(nclusters))
        label0 = (np.asarray(np.where(y_train == i))).T
        # model0=xgb.XGBRegressor(n_estimators=2000)
        # model0=np.empty([1,2],dtype=object)

        a0 = X_train[label0[:, 0], :]
        a0 = np.reshape(a0, (-1, numruth), "F")
        b0 = yruth[label0[:, 0], :]
        b0 = np.reshape(b0, (-1, 1), "F")
        if a0.shape[0] != 0 and b0.shape[0] != 0:
            # model0.fit(a0, b0,verbose=False)
            theta, con1 = fit_machine(a0, b0, degg)

        filename = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
        filename2 = "Regressor_Features_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
        os.chdir(training_master)
        # sio.savemat(filename, {'model0':model0})

        with open(filename, "wb") as model_file:
            pickle.dump(theta, model_file)

        # Save the transformer
        with open(filename2, "wb") as transformer_file:
            pickle.dump(con1, transformer_file)

        os.chdir(oldfolder)
    return nclusters
    print("Finished CCR")


def PREDICTION_CCR__MACHINE(
    ii, nclusters, inputtest, numcols, training_master, oldfolder, pred_type, deg
):
    # import numpy as np
    # ii=0
    # nclusters=2
    # inputtest=X_test2
    print("Starting Prediction")
    filename1 = "Classifier_%d.bin" % ii
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    os.chdir(training_master)
    loaded_model = xgb.Booster({"nthread": 4})  # init model
    clfx = pickle.load(open(filenamex, "rb"))
    clfy = pickle.load(open(filenamey, "rb"))
    loaded_model.load_model(filename1)  # load data
    os.chdir(oldfolder)
    inputtest = clfx.transform(inputtest)
    labelDA = loaded_model.predict(xgb.DMatrix(inputtest))
    if nclusters == 2:
        labelDAX = 1 - labelDA
        labelDA = np.reshape(labelDA, (-1, 1))
        labelDAX = np.reshape(labelDAX, (-1, 1))
        labelDA = np.concatenate((labelDAX, labelDA), axis=1)

    numrowstest = len(inputtest)
    clementanswer = np.zeros((numrowstest, 1))
    # numcols=13
    if pred_type == 1:  # Hard prediction
        labelDA = np.argmax(labelDA, axis=-1)
        labelDA = np.reshape(labelDA, (-1, 1), "F")
        for i in range(nclusters):
            print("-- Predicting cluster: " + str(i) + " | " + str(nclusters))
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2a = "Regressor_Features_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)
            # Load the model
            with open(filename2, "rb") as model_file:
                model0 = pickle.load(model_file)

            # Load the transformer
            with open(filename2a, "rb") as transformer_file:
                modell0 = pickle.load(transformer_file)

            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                clementanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine(a00, deg, model0, modell0), (-1, 1)
                )

        clementanswer = clfy.inverse_transform(clementanswer)
    else:  # soft prediction
        # deg=4
        big_out = np.zeros((numrowstest, nclusters))
        for i in range(nclusters):
            print("-- predicting cluster: " + str(i + 1) + " | " + str(nclusters))
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2a = "Regressor_Features_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)
            with open(filename2, "rb") as model_file:
                model0 = pickle.load(model_file)

            # Load the transformer
            with open(filename2a, "rb") as transformer_file:
                modell0 = pickle.load(transformer_file)
            os.chdir(oldfolder)
            aa = np.reshape(predict_machine(inputtest, deg, model0, modell0), (-1, 1))
            aanew = np.multiply(aa, np.reshape(labelDA[:, i], (-1, 1)))
            big_out[:, i] = np.ravel(aanew)
        clementanswer = np.reshape(np.sum(big_out, axis=1), (-1, 1), "F")
        # clementanswer=clfy.inverse_transform(clementanswer)
    return clementanswer
    print("Finished prediction")


# ------------------Begin Code-------------------------------------------------------------------#
print("")
print("-------------------LOAD INPUT DATA-------------------------------------")
mat = sio.loadmat(("../PACKETS/conversions.mat"))
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

with gzip.open(("../PACKETS/data_train_peaceman.pkl.gz"), "rb") as f:
    mat = pickle.load(f)
X_data2 = mat


data2 = X_data2

X = np.vstack(data2["X"])
Y = np.vstack(data2["Y"])
Y = Y[:, :66]
Y[Y <= 0] = 0
datafind = os.path.join(oldfolder, "Data")
# degg=int(input('select degree polynomial: 4-10: '))
degg = 3


Machinetrue = "../ML_MACHINE"

if not os.path.exists(("../ML_MACHINE")):
    os.makedirs(("../ML_MACHINE"))
else:
    pass


np.random.seed(5)
trainingmaster = os.path.join(oldfolder, Machinetrue)


inpuutx, outpuutx = X, Y
os.chdir(oldfolder)
inpuutx = inpuutx.astype("float32")
outpuutx = outpuutx.astype("float32")

intee_raw = inpuutx
# pred_type=int(input('Choose: 1=Hard Prediction, 2= Soft Prediction: '))
pred_type = 1
# pred_type=2
print("-------------MODEL FITTING FOR PEACEMANN WELL MODEL-----------")
print("Using CCR for peacemann model fitting")
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
outpuutx2 = outpuutx
inpuutx2 = inpuutx
iniguess = inpuutx2

# inpuutx2=(scaler2a.transform(inpuutx2))
inpuut2, X_test2, outpuut2, y_test2 = train_test_split(
    inpuutx2, outpuutx2, test_size=0.01
)  # train_size=2000)


print("--------------------- Learn the Forward model with CCR----------------")

inputsz = range(Y.shape[1])
num_cores = 12  # multiprocessing.cpu_count()

from loky import get_reusable_executor

executor_run = get_reusable_executor(max_workers=num_cores)
with executor_run:
    bigs = Parallel(n_jobs=num_cores, backend="multiprocessing", verbose=50)(
        delayed(startit)(ib, outpuut2, inpuut2, trainingmaster, oldfolder, degg)
        for ib in inputsz
    )
    executor_run.shutdown(wait=False)

big = np.vstack(bigs)

os.chdir(trainingmaster)
for i in range(Y.shape[1]):
    a = open("clustersizescost.dat", "a+")
    a.write("%d \n" % (big[i, :]))
    a.close()
os.chdir(oldfolder)


print(" -------------------------Predict For Energy Machine-----------------")
os.chdir(trainingmaster)
cluster_all = np.genfromtxt("clustersizescost.dat", dtype="float")
cluster_all = np.reshape(cluster_all, (-1, 1), "F")
os.chdir(oldfolder)

executor_run1 = get_reusable_executor(max_workers=num_cores)
with executor_run1:
    clemes = Parallel(n_jobs=num_cores, backend="multiprocessing", verbose=50)(
        delayed(PREDICTION_CCR__MACHINE)(
            ib,
            int(cluster_all[ib, :]),
            X_test2,
            X.shape[1],
            trainingmaster,
            oldfolder,
            pred_type,
            degg,
        )
        for ib in inputsz
    )
    executor_run1.shutdown(wait=False)
outputpredenergy = np.hstack(clemes)

print(" ")


outputpredenergy = outputpredenergy * max_out_fcn
outputpredenergy[outputpredenergy <= 0] = 0
y_test2 = y_test2 * max_out_fcn

CoDoveralle, L_2overalle, CoDviewe, L_2viewe = Performance_plot_cost(
    outputpredenergy, y_test2, "Machine_Energy_perform", trainingmaster, oldfolder
)
print("R2 of fit using the Energy machine for model is :", CoDoveralle)
print("L2 of fit using the Energy machine for model is :", L_2overalle)
print("-------------------PROGRAM EXECUTED-------------------------------------")
