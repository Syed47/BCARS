# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:05:15 2023

@author: ryanm
"""
import os
os.chdir("C:/Users/19719431/OneDrive - Maynooth University/Desktop/BCARS/preprocessing")
import pandas as pd
# from KK_image import *
import numpy as np
import matplotlib.pyplot as plt
import time as time
from drawnow import drawnow
from sklearn.decomposition import PCA
from numpy.linalg import svd as svd
from anscombe_transform import *
import anscombe_transform
import h5py
from scipy.linalg import diagsvd as diagsvd
from scipy.signal import savgol_filter
from crikit.cri.kk import KramersKronig
from crikit.cri.error_correction import PhaseErrCorrectALS as PEC
from crikit.cri.error_correction import ScaleErrCorrectSG as SEC
import matplotlib as mpl
from brokenaxes import brokenaxes
import matplotlib.gridspec as gridspec
mpl.rcParams.update(mpl.rcParamsDefault)
#

# Import h5 file
f = h5py.File(r"C:\Users\19719431\OneDrive - Maynooth University\Desktop\BCARS\h5files\13-Mar-2024_1839_10.047.h5", 'r')

# extract hyperspectral image data
df= f['Image'][:]
# transposing the array for standard configuration
df = np.transpose(df, (1,2,0))

# Our spectrometer has 1024 pixels
spec_len = 1024

df = np.asarray(df) 

# df = df[0:150, 0:150, :]

# Rotate image as we record sideways
df = np.rot90(np.flip(df,2))
df = (np.flip(df,0))

# Spatial length in pixels
side_len1 = (np.size(df,0))
side_len2 = (np.size(df,1))

# Plot high-contrast average spectral intensity
mean_im = np.mean(df[:,:,-20:-1],2)
sum_im = np.sum(df,2)
im = df[:,:,0]
image = sum_im/mean_im
plt.imshow(image,cmap='gray')
plt.show()

#%%

newdf = np.reshape(df, (400 * 400, 1024))

from sklearn.decomposition import PCA
pca_out = PCA(n_components=2)
pca_covariance = pca_out.fit_transform(newdf)
# print(dir(pca))
pca_img = np.reshape(pca_covariance, (400, 400, 2))
plt.imshow(pca_img[:,:,1])
plt.show()

#%% Singular Value Decomposition - Economy SVD

# We calculate the image mean and standard deviation in a background region - required for SVD reconstruction
mean = np.mean(np.mean(np.mean(df[350:400,0:50,450:500],2),0),0)
std = np.mean(np.mean(np.std(df[350:400,0:50,450:500],2),0),0) 

# df_ans = Anscombe transformed data for removing heteroscedasticity
df_ans = np.zeros((side_len1,side_len2,1024))
for i in range(np.size(df,0)):
    for k in range(np.size(df,1)):
        ansc = generalized_anscombe(df[i,k,:], mu=mean, sigma=std)      
        df_ans[i,k,:] = ansc

df_ans = np.reshape(df_ans,(np.size(df,0)*np.size(df,1),1024))

# Calculate singular value decomposition of array
U,S,V = svd(df_ans, full_matrices=False)
N = np.size(df,0);
sigma= std;
cutoff = (4/np.sqrt(3))*np.sqrt(N)*sigma;
#r = np.sum((S>cutoff))

#% Display highest singular values eigen-spectra
fig, ax= plt.subplots(nrows = 5, ncols = 5,figsize = (10,8))
ax = ax.flatten()
for i in range(25):
    r = i
    denoised = V[r,:]  
    label = r'$SV_{{{}}}$'.format((str(i+1)))
    ax[i].plot(denoised,label=label)
    ax[i].set_title(label)
fig.tight_layout()
plt.show()
#%% Show eigen-maps of singular vectors
fig, ax= plt.subplots(nrows = 4, ncols = 5,figsize = (11,11))
ax = ax.flatten()
for i in range(20):
    r = i
    U1 = U[:,r]
    denoised = np.reshape(U1,(side_len1,side_len2))
    label = r'$SV_{{{}}}$'.format((str(i+1)))
    z1= ax[i].imshow(denoised,label=label,cmap = 'hsv')
    ax[i].set_title(label)
    ax[i].set_aspect('equal')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()
#%% Calculate compact SVD reconstruction based on custom singular values

# These are the singular values which will be used for low-rank reconstruction (must use 0-20 always)
#r = [0,1,2,3,4,5,8,10,13,14,15,18]
r=range(20)

s_select = np.zeros(S.size)
s_select[r] = S[r]

M = U.shape[-1]
N = V.shape[0]
Sr = diagsvd(s_select,M,N)
denoised = U @ (Sr) @ V
df_ans = np.reshape(denoised,(side_len1,side_len2,spec_len))
df_denoised = np.zeros((side_len1,side_len2,spec_len))

# Calculating the inverse Anscombe transform on the SVD reconstructed data
for i in range(np.size(df,0)):
    for k in range(np.size(df,1)):
        inv_ans = inverse_generalized_anscombe(df_ans[i,k,:],mu=mean, sigma=std)     
        df_denoised[i,k,:] = inv_ans
#%% This is for checking the quality of denoising - set x and y to a signal pixel based off the image
x = 50
y = 50

plt.plot(df[x,y,:],label = 'before SVD',c = 'k',linewidth = 1)
plt.plot(df_denoised[x,y,:]+100,label = 'After SVD',c = 'r',linewidth = 1)
plt.legend()
plt.show()

#%% This is for optimizing the Kramers-Kronig NRB removal, we first perform it on a single spectrum and evaluate results

# First we crop the spectral length as sometimes we have laser present on the blue side of the spectrum
right_cutoff = 1010

# Import pre-calibrated wavelength spectrum (this needs updating when plotting for a publication)
wl = np.squeeze(np.asarray(pd.read_csv(r'C:\\Users\\19719431\\OneDrive - Maynooth University\\Desktop\\preprocessing\\wl.csv',header=None)))
wl=wl[10:right_cutoff]

# Calculating the Raman shift based off an estimate of the laser wavelength
wn = 1e7*(1/wl-1/770.9)

# Setting NRB as an average of a 'background' pixel area
nrb_x = 375
nrb_y = 25

nrb = (np.mean(np.mean(df_denoised[nrb_x :nrb_x +5,nrb_y:nrb_y+5 ,:],0),0))

# Cropping signals
signal = df_denoised[:,:,10:right_cutoff]
nrb = nrb[10:right_cutoff]

nrb_min = nrb
nrb_signal = nrb_min
spec_len = np.size(signal,2)
new_signal = np.ones((side_len1,side_len2,spec_len))

# can be used for pixel normalisation (usually ignore)
for i in range(side_len1):
    for k in range(side_len2):
        new_signal[i,k,:] = signal[i,k,:]#-signal[i,k,:].min()
        new_signal[i,k,:] = new_signal[i,k,:]#/new_signal[i,k,:].max()
        
cars_spectrum = new_signal[x,y,:]       
  

fix_rng = np.hstack((np.arange(4,6))) #370-600
fix_rng=fix_rng.astype(int)

rng = np.arange(0,np.size(cars_spectrum))
#idx = np.s_[0:0]
#rng[idx]=0
rng=rng.astype(int)

asym_param = 1e-2*np.ones(1000)
asym_param[380:800] = 1e-2
asym_param[0:380] = 1e-3

kk = KramersKronig(conjugate=1, norm_to_nrb=True,pad_factor=1)

retrieved = kk.calculate(cars_spectrum, nrb_signal)

pec = PEC(wavenumber_increasing=False, smoothness_param=1e2, asym_param=asym_param, redux=1, 
          fix_end_points=False, rng=rng, fix_rng=fix_rng, fix_const=0)

sec = SEC()
phase_corrected = pec.calculate(retrieved)

result = savgol_filter(sec.calculate(phase_corrected).imag,5,3)

fig, ax= plt.subplots(nrows = 2, ncols = 1,figsize = (9,8))
ax = ax.flatten()

ax[0].plot(np.angle(retrieved),label = 'raw phase')
#plt.plot(np.angle(pec.calculate(phase_corrected)),label = 'phase error')
ax[0].plot(np.angle(retrieved)-np.angle(pec.calculate(phase_corrected)),label = ' error')
ax[0].legend()
ax[1].plot(wn,result,label = 'Retrieved Im[$\chi^{(3)}$]')
ax[1].legend()
plt.show()

#%% Kramers-Kronig Retrieval on whole dataset using above single-pixel parameters
KK_signal = kk.calculate(new_signal, nrb_signal)    
phase_corrected = pec.calculate(KK_signal)
result = savgol_filter(sec.calculate(phase_corrected).imag,5,3)
#%% Plot Signal and NRB example
fig = plt.figure(figsize=(10,6),constrained_layout=False)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

bax1 = brokenaxes(xlims=((500, 1800),(2700, 3300)), subplot_spec=spec[0], wspace = 0.1)
bax1.plot(wn,result[x,y,:],label = 'signal',c='#1874CD',linewidth=1)
bax1.plot(wn,result[0,-1,:],label = 'glass',c='tab:green',linewidth=1)
#bax1.axs[0].set_ylim(-0.01,0.2)
#bax1.axs[1].set_ylim(-0.01,0.2)

#bax1.set_ylim(-0.0,0.1)
bax1.set_xlabel('Raman Shift ($\mathrm{cm}^{-1}$)',labelpad = 20)
bax1.set_ylabel('Intensity (A.U.)',labelpad = 30)
#bax1.legend(loc='upper left')
bax1.set_title('Retrieved BCARS spectra')

plt.show()
bax1.legend(loc=2)
plt.savefig('retrieved.png',dpi=600)

#%% This is for choosing the wavelengths to color
x, y = 128, 11
plt.plot(result[103, 188,:],label = 'signal',linewidth = 1)
plt.plot(result[129, 260,:],label = 'signal',linewidth = 1)
plt.plot(result[150, 150,:],label = 'glass',linewidth = 1)
# plt.plot(result[92, 33,:],label = 'signal',linewidth = 1)

plt.legend()
plt.show()
# 313, 261


#%% Produce and display RGB image with three intensity bands
from matplotlib.colors import LogNorm
rgb = np.zeros((side_len1,side_len2,3))

spec_line1 = 840
spec_line2 = 897
spec_line3 = 962

loc = result[:,:,338]

loc = loc > 0.00

cropped = result*loc[:,:,np.newaxis]

t1 = time.perf_counter()

for i in range(result.shape[0]):
  for j in range(result.shape[1]):
    rgb[i,j,0] = 1*cropped[i,j,spec_line1]/(cropped[:,:,spec_line1].max()).max() if cropped[i,j,spec_line1] > 0.00 else 0 # R
    rgb[i,j,1] = 1*cropped[i,j,spec_line2]/(cropped[:,:,spec_line2].max()).max() if cropped[i,j,spec_line2] > 0.0026 else 0 # R
    rgb[i,j,2] = 0*cropped[i,j,spec_line3]/(cropped[:,:,spec_line3].max()).max() if cropped[i,j,spec_line3] > 0.00 else 0 # R

fig,ax = plt.subplots(1,1)
ax.imshow(rgb)
plt.show()
#%%

newcropped = np.reshape(cropped, (150 * 150, 1000))

from sklearn.decomposition import PCA
pca_out = PCA(n_components=2)
pca_covariance = pca_out.fit_transform(newcropped)
# print(dir(pca))
pca_img = np.reshape(pca_covariance, (150, 150, 2))
plt.imshow(pca_img[:,:,1])
plt.show()

# method
#  data 1 half
#%%
cropped.shape

c1 = np.mean(cropped, axis=2)
plt.imshow(c1)
plt.plot(90, 50, marker="3", color="orange")
plt.plot(108, 145, marker="3", color="orange")
plt.plot(95, 90, marker="3", color="orange")
plt.show()

#%%
plt.plot(cropped[90,50,:])
plt.show()
plt.plot(cropped[108, 145, :])

plt.show()
plt.plot(cropped[90,95, :])

plt.show()
#%%

plt.imshow(c1[25:60, 75:125])
plt.plot(15, 28, color="orange", marker="3")
plt.show()

#%%
plt.plot(cropped[25+28,75+15,:])
plt.show()

#%%
plt.imshow(c1[ 125:150,100:125])
plt.plot(9,20, marker="3", color="orange")
plt.show()
#%%
plt.plot(cropped[125+9, 100+20, :])
plt.show()
#%%

cnew = cropped[50:150, 75:126,:]
cvis = np.mean(cnew, axis=2)
plt.imshow(cvis, cmap="gray")
plt.show()