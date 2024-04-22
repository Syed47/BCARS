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
plt.title("BCARS Image of PS and PMMA")
plt.xticks([])
plt.yticks([])
plt.show()

#%%

newdf = np.reshape(df, (400 * 400, 1024))

from sklearn.decomposition import PCA
pca_out = PCA(n_components=2)
pca_covariance = pca_out.fit_transform(newdf)
# print(dir(pca))
pca_img = np.reshape(pca_covariance, (400, 400, 2))
# plt.imshow(pca_img[:,:,0])
plt.imshow(pca_img[:,:,0])
plt.show()


#%%

# First we crop the spectral length as sometimes we have laser present on the blue side of the spectrum
# right_cutoff = 1010

# # Import pre-calibrated wavelength spectrum (this needs updating when plotting for a publication)
# wl = np.squeeze(np.asarray(pd.read_csv(r'C:\\Users\\19719431\\OneDrive - Maynooth University\\Desktop\\preprocessing\\wl.csv',header=None)))
# wl=wl[10:right_cutoff]

# # Calculating the Raman shift based off an estimate of the laser wavelength
# wn = 1e7*(1/wl-1/770.9)

# fig = plt.figure(figsize=(10,6),constrained_layout=False)
# spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
# x = 390
# y = 235

# bax1 = brokenaxes(xlims=((500, 1800),(2700, 3300)), subplot_spec=spec[0], wspace = 0.1)

# bax1.set_yticklabels([])
# bax1.plot(wn,df[x,y,:1000],label = 'signal',c='#1874CD',linewidth=1)
# bax1.plot(wn,df[0,-1,:1000],label = 'glass',c='tab:green',linewidth=1)

# bax1.plot(wn, df[203, 232,:1000], color='r', label = 'PMMA',linewidth = 1)
# # plt.plot(result[129, 260,:],label = 'signal',linewidth = 1)
# # bax1.plot(wn, df[387, 235,:1000],label = 'PS',linewidth = 1)
# # plt.plot(result[5, 382,:],label = 'signal',linewidth = 1)
# # plt.plot(result[150, 150,:],label = 'glass',linewidth = 1)
# plt.legend()
# plt.show()

# x is wavelength
# ignore y labels
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

#%% Display highest singular values eigen-spectra
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


# First we crop the spectral length as sometimes we have laser present on the blue side of the spectrum
right_cutoff = 1010

# Import pre-calibrated wavelength spectrum (this needs updating when plotting for a publication)
wl = np.squeeze(np.asarray(pd.read_csv(r'C:\\Users\\19719431\\OneDrive - Maynooth University\\Desktop\\preprocessing\\wl.csv',header=None)))
wl=wl[10:right_cutoff]

# Calculating the Raman shift based off an estimate of the laser wavelength

fig = plt.figure(figsize=(10,6),constrained_layout=False)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)


bax1 = brokenaxes(xlims=((500, 1900),(2700, 3200)), subplot_spec=spec[0], wspace = 0.1)
bax1.set_yticklabels([])

wl = list(filter(lambda x: x > 500, wl))

wn = [1e7*(1/w-1/770.9) for w in wl]

wnl = len(wn) # 1000
x = 203
y = 232

bax1.plot(wn, df[x,y,:wnl],label = 'PS before SVD',c = 'y',linewidth = 1)
bax1.plot(wn, df_denoised[x,y,:wnl]+40,label = 'PS after SVD',c = 'g',linewidth = 1)

x = 387
y = 235
bax1.plot(wn, df[x,y,:wnl]+100,label = 'PMMA before SVD',c = 'orange',linewidth = 1)
bax1.plot(wn, df_denoised[x,y,:wnl]+140,label = 'PMMA after SVD',c = 'r',linewidth = 1)

bax1.set_xlabel('Raman Shift ($\mathrm{cm}^{-1}$)',labelpad = 20)
bax1.set_ylabel('Intensity (A.U.)',labelpad = 30)
bax1.legend(loc='upper right')
bax1.set_title('BCARS spectra')

plt.show()


# x = 94
# y = 220
# plt.plot(wn, df[x,y,:1000] + 100,label = 'before SVD',c = 'k',linewidth = 1)
# plt.plot(wn, df_denoised[x,y,:1000]+150,label = 'After SVD',c = 'g',linewidth = 1)
# plt.legend()
# plt.show()

# x = 390
# y = 235
# plt.plot(wn, df[x,y,:1000],label = 'before SVD',c = 'k',linewidth = 1)
# plt.plot(wn, df_denoised[x,y,:1000]+50,label = 'After SVD',c = 'r',linewidth = 1)
# plt.legend()
# plt.show()




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
result = savgol_filter(sec.calculate(phase_corrected).imag,5,2)
#%% Plot Signal and NRB example
fig = plt.figure(figsize=(10,6),constrained_layout=False)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

bax1 = brokenaxes(xlims=((500, 1800),(2700, 3300)), subplot_spec=spec[0], wspace = 0.1)
bax1.set_yticklabels([])


bax1.plot(wn, result[203, 232,:],label = 'PS',c = 'r',linewidth = 1)

bax1.plot(wn, result[387, 235,:],label = 'PMMA',c = 'g',linewidth = 1)



# bax1.plot(wn,result[x,y,:],label = 'signal',c='#1874CD',linewidth=1)
# bax1.plot(wn,result[0,-1,:],label = 'glass',c='tab:green',linewidth=1)
#bax1.axs[0].set_ylim(-0.01,0.2)
#bax1.axs[1].set_ylim(-0.01,0.2)

#bax1.set_ylim(-0.0,0.1)
bax1.set_xlabel('Raman Shift ($\mathrm{cm}^{-1}$)',labelpad = 20)
bax1.set_ylabel('Intensity (A.U.)',labelpad = 30)
#bax1.legend(loc='upper left')
bax1.set_title('Processed BCARS spectra - NRB Removed')

plt.show()
bax1.legend(loc=2)
plt.savefig('retrieved.png',dpi=600)

#%% This is for choosing the wavelengths to color

# x = 390
# y = 235

# x = 94
# y = 220

plt.plot(wn, result[203, 232,:],label = 'PS',c='g',linewidth = 1)
# plt.plot(result[129, 260,:],label = 'signal',linewidth = 1)
plt.plot(wn, result[387, 235,:],label = 'PMMA',c='r',linewidth = 1)
# plt.plot(result[5, 382,:],label = 'signal',linewidth = 1)
# plt.plot(result[150, 150,:],label = 'glass',linewidth = 1)
plt.legend()
plt.xticks([])
plt.yticks([])
plt.show()
# 313, 261


#%% Produce and display RGB image with three intensity bands
from matplotlib.colors import LogNorm
rgb = np.zeros((side_len1,side_len2,3))

spec_line1 = 840
spec_line2 = 331#897
spec_line3 = 962

loc = result[:,:,170]

loc = loc > 0.00

cropped = result*loc[:,:,np.newaxis]

# t1 = time.perf_counter()

# for i in range(result.shape[0]):
#   for j in range(result.shape[1]):
#     rgb[i,j,0] = 1*cropped[i,j,spec_line1]/(cropped[:,:,spec_line1].max()) if cropped[i,j,spec_line1] > 0.00 else 0 # R
#     rgb[i,j,1] = 1*cropped[i,j,spec_line2]/(cropped[:,:,spec_line2].max()) if cropped[i,j,spec_line2] > 0.00 else 0 # R
#     rgb[i,j,2] = 1*cropped[i,j,spec_line1]/(cropped[:,:,spec_line3].max()) if cropped[i,j,spec_line1] > 0.00 else 0 # R

# Assuming cropped, spec_line1, spec_line2, spec_line3, and rgb are defined appropriately

# Calculate the maximum values along the third axis once
max_values = np.max(cropped[:, :, [spec_line1, spec_line2, spec_line3]], axis=2)

# Avoid nested loops using NumPy's vectorized operations
rgb[:, :, 0] = np.where(cropped[:, :, spec_line1] > 0.02, cropped[:, :, spec_line1] / max_values, 0)  # R
a = np.where(cropped[:, :, spec_line2] > 0.02, cropped[:, :, spec_line2] / max_values, 0)  # G
b = np.where(cropped[:, :, spec_line3] > 0.01, cropped[:, :, spec_line3] / max_values, 0)  # B
rgb[:,:, 1] = np.max(np.array([a, b]), axis=0)


fig,ax = plt.subplots(1,1)
ax.scatter([-3], [-3], color="red", label="PMMA")
ax.scatter([-3], [-3], color="green", label="PS")
ax.imshow(rgb, cmap="hsv")
ax.legend(loc="upper right")
plt.title("False Color Image of PS and PMMA")
plt.xticks([])
plt.yticks([])
plt.show()

#%% Plot Signal and NRB example


# pmma = [(206,244), (300,308), (345, 204), (311, 227), (230, 222), (387, 235) ]
# ps = [(202,232), (266,220), (293,212), (315, 206), (284, 295), (203,232)]
# print(rgb.shape)
new_rgb = np.reshape(rgb, (400*400, 3))
pmma = np.where(rgb[:, 0] > 0.02, 1, 0)
ps = np.where(rgb[:, 1] > 0.02, 1, 0)
print("pmma shape" + str(pmma.shape))

fig = plt.figure(figsize=(10,6),constrained_layout=False)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

bax1 = brokenaxes(xlims=((200, 400),(600, 1000)), subplot_spec=spec[0], wspace = 0.1)
bax1.set_yticklabels([])
# plt.imshow(pmma, cmap="gray")
# plt.show()
pmma = np.reshape(pmma, (400*400))
pmm = np.reshape(pmm, (400*400, 1000))
pmm = result*pmma[:,:,None]
print(f" nonzero pmm: {np.count_nonzero(pmm)}")
print(f"PMM reshaped before filter: {pmm.shape}")
pmm = pmm[np.where(pmm!=0)]
print(f"PMM FILTER SHAPE: {pmm.shape}")
print(np.count_nonzero(pmma))

#bax1.imshow(ps,c = 'r',linewidth = 1)
print("pmm: " + str(pmm.shape))
pmma_mean = np.nanmean(pmm,axis=0)
pmma_std= np.nanstd(pmm, axis=0)
print("MEAN: " + str(pmma_masked.shape))

# plt.figure()
# plt.plot(pmma_masked,c = 'g',linewidth = 1)
# plt.plot(pmma_std, c='b', linewidth=1)
# plt.show()
#%%
pmma_pos_std = pmma_masked + pmma_std
pmma_neg_std = pmma_masked - pmma_std

# bax1.plot(wn,result[x,y,:],label = 'signal',c='#1874CD',linewidth=1)
# bax1.plot(wn,result[0,-1,:],label = 'glass',c='tab:green',linewidth=1)
#bax1.axs[0].set_ylim(-0.01,0.2)
#bax1.axs[1].set_ylim(-0.01,0.2)
#%%
plt.fill_between(wn,pmma_pos_std, pmma_neg_std, color='C0', alpha=0.3)
#bax1.set_ylim(-0.0,0.1)
bax1.set_xlabel('Raman Shift ($\mathrm{cm}^{-1}$)',labelpad = 20)
bax1.set_ylabel('Intensity (A.U.)',labelpad = 30)
#bax1.legend(loc='upper left')
bax1.set_title('Processed BCARS spectra - NRB Removed')

plt.show()
bax1.legend(loc=2)
plt.savefig('retrieved.png',dpi=600)
#%%
print(cropped.shape)
newcropped = np.reshape(cropped, (400 * 400, 1000))

from sklearn.decomposition import PCA
pca_out = PCA(n_components=2)
pca_covariance = pca_out.fit_transform(newcropped)
pca_img = np.reshape(pca_covariance, (400, 400, 2))
plt.imshow(pca_img[:,:,1])
plt.title("PCA 2")
plt.xticks([])
plt.yticks([])
plt.show()
#%%

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import seaborn as sns
import plotly.express as px

data1 = np.reshape(cropped, (400 * 400, 1000))
print(newdf.shape)

Sc = StandardScaler()
# data1 = pd.read_csv("/home/tim/Desktop/MRes/BCARS/BCARS tools/PMMA_Dataset.csv").T
data1['feature'] = 0
data1.head()

#%%
#%%
data2 = pd.read_csv("/home/tim/Desktop/MRes/BCARS/BCARS tools/PVC_Dataset.csv").T
data2['feature'] = 1
data2.head()
#%%
data = pd.concat([data1, data2])
data.reset_index(inplace=True, drop=True)
data.shape
data.head()
data.columns = data.columns.astype(str)

print(data)
#%%

#%%
pca = PCA(2)
pca.fit(data)
#%%
pca_data = pca.transform(data)
#%%
pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i}' for i in range(1, 3)])
#%%

plt.plot(pca_df)
plt.show()

#%%
pca_df['feature'] = data['feature']
#%%
sns.lmplot(
    x='PC1', 
    y='PC2',
    hue="feature",
    data=pca_df,  
    fit_reg=False, 
    legend=True)

plt.show()
#%%
# df = pd.read_csv("cluster_mean_spectra")
# data = df.to_numpy()
# data = data[1:, 1:]
spontaneousraman = pd.read_csv("/home/tim/Desktop/MRes/SPIE Papers/data/CAL_1.txt", header=None).T
spontaneousraman = spontaneousraman.to_numpy()
#%%
spontaneouswn_axis = pd.read_csv(r'C:\\Users\\19719431\\OneDrive - Maynooth University\\Desktop\\preprocessing\\wl.csv', header=None)
spontaneouswn_axis = spontaneouswn_axis.to_numpy()
print(spontaneouswn_axis.shape)
print(spontaneousraman.shape)
sr_mean = np.mean(spontaneousraman,axis=0)
std = np.std(spontaneousraman, axis=0)
sr_std_pos = sr_mean+std
sr_std_neg = sr_mean-std
print(sr_std_pos.shape)

di_e = 650
di_s = 170
plt.plot(spontaneouswn_axis[0,di_s:di_e], sr_mean[di_s:di_e])
plt.fill_between(spontaneouswn_axis[0,di_s:di_e], sr_std_pos[di_s:di_e], sr_std_neg[di_s:di_e], color='C0', alpha=0.3)
labels =[r'$\bar{x}$', r'$\bar{x}$' + r'$\pm$' + r'$\sigma$']
line_handle, = plt.plot([], [], color="C0")  # Handle for the line
patch_handle = plt.Rectangle((0, 0), 1, 1, color='C0', alpha=0.3) 
handles = [line_handle, patch_handle]
plt.xlabel("Raman Shift (cm$^{-1}$)")
plt.ylabel("AU")
legend = plt.legend(handles, labels, loc='upper left')
plt.gca().set_yticklabels([])
plt.savefig('/home/tim/Desktop/SpontaneousRamanMeanSTD.png',dpi=600)
plt.show()