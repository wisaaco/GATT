
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

df = pd.read_csv("fits.csv")# ,names=["generation","Wmax","Wmin","Pmax","Pmin","Fmax","Fmin","Wmean","Pmean","Fmean","fitValue"])
df["Wmax"]

df.plot(y=["Wmax","Wmin","Wmean"])
df.plot(y=["Pmax","Pmin","Pmean"])
df.plot(y=["Fmax","Fmin","Fmean"])

df.plot(y=["Max_CVM","Min_CVM","Mean_CVM"])

df.plot(y="fitValue")

#
#df.plot(y=["HWmax","HWmin"])
#df.plot(y=["HPmax","HPmin"])
#df.plot(y=["HFmax","HFmin"])
#
#
#df[10:].plot(y="fitValue")

