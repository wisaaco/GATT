
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

df = pd.read_csv("fits.csv")# ,names=["generation","Wmax","Wmin","Pmax","Pmin","Fmax","Fmin","Wmean","Pmean","Fmean","fitValue"])
df["Wmax"]

x = 0

df[x:].plot(y=["Wmax","Wmin","Wmean"])
df[x:].plot(y=["Pmax","Pmin","Pmean"])
df[x:].plot(y=["Fmax","Fmin","Fmean"])

df[:].plot(y=["Max_CVM","Min_CVM","Mean_CVM"])

# df.plot(y="fitValue")

df[x:].plot(y=["fitValue","Fit_maxPareto","Fit_meanPareto"])


#
#df[x:].plot(y=["NWmax","NWmin","NWmean"])
#df[x:].plot(y=["NPmax","NPmin","NPmean"])
#df[x:].plot(y=["NFmax","NFmin","NFmean"])



#df = pd.read_csv("fitUPMI.csv",names=["pm1","pm2","pm3"])


#
#df.plot(y=["HWmax","HWmin"])
#df.plot(y=["HPmax","HPmin"])
#df.plot(y=["HFmax","HFmin"])
#
#
#df[10:].plot(y="fitValue")

a, m  = 1.5,4
sizeFiles = []
for ix in range(11886):
    sizeFiles.append(int(np.random.pareto(a)*m)+1)
np.sum(sizeFiles)

Bloques = 100000

mu, sigma = 3.5, 1.43# mean and standard deviation
tExecution = np.random.lognormal(mu, sigma,11886)
np.mean(tExecution)

Segtundos 96