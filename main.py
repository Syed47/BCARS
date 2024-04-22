# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:21:24 2024

@author: 19719431
"""

import os
os.chdir("C:/Users/19719431/OneDrive - Maynooth University/Desktop/BCARS/preprocessing")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pmma = pd.read_csv("data/PMMA flow.csv", header=None, delimiter=" ")

intensity = pmma.iloc[:, 1]
wl = pmma.iloc[::-1,0]
wn = [1e7*(1/w-1/770.9) for w in wl]

plt.plot(wn, intensity)
# Labeling the plot
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')

# Display the plot
plt.show()

#%%

dat = pd.read_csv("data/Flow_1.asc", header=None, delimiter=" ")
dat = dat.to_numpy().T[:1001,:]

wl = dat[0, :]
wn = [1e7*(1/w-1/770.9) for w in wl]

# plt.plot(wn, dat[1:, :])

for i in range(1,1001):

    plt.plot(wn, dat[i,:])
    # Labeling the plot
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    
#%%

index = np.mean(dat[:,:], axis=1)
print(index.shape)
appended_array = np.hstack((dat, index[:,np.newaxis]))

array_p = appended_array[1:,:]
print(array_p)
ordered = np.sort(array_p, axis=0)[::-1, :-1]

print(ordered.shape)


#%%
for i in range(995,1000):
    plt.plot(wn, ordered[i,:])
    # Labeling the plot
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
for i in range(0,5):
    plt.plot(wn, ordered[i,:]/5)
    # Labeling the plot
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
plt.show()




