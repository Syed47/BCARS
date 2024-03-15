# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:44:59 2023

@author: ryanm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:16:58 2022

@author: ryanm
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:56:47 2022

@author: ryanm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:00:21 2022

@author: ryanm
"""

import pandas as pd
from KK_image import *
import numpy as np
import matplotlib.pyplot as plt
import time as time
from drawnow import drawnow
import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import svd as svd
from anscombe_transform import *
import h5py
from scipy.linalg import diagsvd as diagsvd
from spectral import *
from scipy.signal import hilbert
from crikit.cri.kk import KramersKronig
from crikit.cri.error_correction import PhaseErrCorrectALS as PEC
from crikit.cri.error_correction import ScaleErrCorrectSG as SEC

from scipy.signal import find_peaks

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)





right_cutoff = 956
left_cutoff = 1

wl = np.squeeze(np.asarray(pd.read_csv(r'C:\Users\ryanm\OneDrive\Documents\MATLAB\Wavelength calibration\wl.csv',header=None)))
wl=wl[left_cutoff:right_cutoff]


wn = 1e7*(1/wl-1/770.9)
# Setting NRB

nrb = np.squeeze(np.asarray(pd.read_csv(r'C:\Users\ryanm\OneDrive - Maynooth University\Maynooth\Lab documents\CARS spectra\SERS\KK_15_SH_50_2sec_NRB.asc',header=None,sep = '\t')))
nrb = nrb[:,1]

signal = np.squeeze(np.asarray(pd.read_csv(r'C:\Users\ryanm\OneDrive - Maynooth University\Maynooth\Lab documents\CARS spectra\SERS\KK_15_SH_50_2sec.asc',header=None,sep = '\t')))
signal = signal[:,1]

signal = signal[left_cutoff:right_cutoff]
nrb = nrb[left_cutoff:right_cutoff]

nrb_min = nrb#-nrb.min()
nrb_signal = nrb_min#/nrb_min.max()
spec_len = np.size(signal,0)


new_signal = []
new_signal = signal#-signal.min()
new_signal = new_signal#/new_signal.max()
        
      
  
fix_rng = np.hstack((np.arange(4,20))) #370-600
fix_rng=fix_rng.astype(int)

rng = np.arange(0,np.size(nrb))
#idx = np.s_[0:0]
#rng[idx]=0
rng=rng.astype(int)

asym_param = 1e-3*np.ones(spec_len)
asym_param[370:700] = 1e-1
asym_param[0:370] = 1e-3


kk = KramersKronig(conjugate=1, norm_to_nrb=True,pad_factor=1)

retrieved = kk.calculate(signal, nrb_signal)

pec = PEC(wavenumber_increasing=False, smoothness_param=1e3, asym_param=asym_param, redux=1, 
          fix_end_points=False, rng=rng, fix_rng=fix_rng, fix_const=0)

sec = SEC()
phase_corrected = pec.calculate(retrieved)

result = savgol_filter(sec.calculate(phase_corrected).imag,19,7)

peaks, _ = find_peaks(result, prominence=0.005,height = 0.005)
peak_coordinates = list(zip(map(round,wn[peaks]), result[peaks]))
peak_wavenumbers = wn[peaks]

fig, ax= plt.subplots(nrows = 2, ncols = 1,figsize = (9,8))
ax = ax.flatten()

ax[0].plot(np.angle(retrieved),label = 'raw phase')
#plt.plot(np.angle(pec.calculate(phase_corrected)),label = 'phase error')
ax[0].plot(np.angle(retrieved)-np.angle(pec.calculate(phase_corrected)),label = ' error')


ax[1].plot(wn,result)
#plt.xlabel('Raman shift (cm$^{-1}$)')



#plt.autoscale(enable=True, axis='x', tight=True)
#plt.xlim((760,3500))
#plt.savefig('PE04_June_50_ms.png',dpi=600)
#plt.ylim((0,1))

#%
fig, ax= plt.subplots(nrows = 1, ncols = 2,figsize = (10,6),gridspec_kw={'width_ratios': [2, 1]})
ax[1].plot(wn,result,linewidth = 1)
#ax[1].set_xlabel('Raman Shift ($\mathrm{cm}^{-1}$)',labelpad = 2)
ax[0].set_ylabel('Intensity (A.U.)',labelpad = 2)
#bax1.legend(loc='upper left')
fig.suptitle('Retrieved BCARS spectra - KK_15_SH_50')
ax[1].set_xlim(2700,3200)
#plt.plot(wn,result[x,y,:],label = 'signal')
#plt.plot(wn,result[10,2,:],label = 'glass')
#plt.plot(result[84,180,:],label = 'glass')

ax[1].spines[['left', 'top']].set_visible(False)
ax[1].set_yticks([])
ax[0].spines[['right', 'top']].set_visible(False)
fig.supxlabel('Raman Shift ($\mathrm{cm}^{-1}$)')
ax[0].plot(wn,result,linewidth = 1)
ax[0].set_xlim(640,2000)
ax[0].set_ylim(0,0.1)
plt.tight_layout()
#plt.ylim((0,1)
#bax1.legend(loc=2)

for xp,yp in peak_coordinates:
    ax[0].text(xp*0.99, yp*1.08, xp, fontsize=8,rotation = 90)

for xp,yp in peak_coordinates:
    ax[1].text(xp*0.999, yp*1.01, xp, fontsize=8,rotation = 90)



plt.show()
#plt.savefig('KK_15_SH_50.png',dpi=300)
#%%
