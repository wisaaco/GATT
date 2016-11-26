# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 19:50:47 2016

@author: isaac
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np



dp = pd.read_csv("prob.csv")
#dp.as_matrix(columns=dp.columns[1:]).shape
len(dp.columns)-1


X = np.arange(0, len(dp), 1)
Y = np.arange(0, len(dp.columns)-1, 1)
X, Y = np.meshgrid(X, Y)
Z = dp.as_matrix(columns=dp.columns[1:]).T

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Generation')
ax.set_ylabel('Citizen')
ax.set_zlabel('Probability acc. fitness')
