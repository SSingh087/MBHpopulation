#!/usr/bin/env python

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from numpy import load, save, array, log10, zeros

from few.waveform import GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper
import matplotlib.pyplot as plt
import cupy as cp
import time
import h5py

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="total events")
args = parser.parse_args()

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'/data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/{events_for_dir}_events/'

from few.trajectory.inspiral import EMRIInspiral

use_gpu = True

inspiral_kwargs = {
    "max_init_len": int(1e7),  # all of the trajectories will be well under max_init_len
}

sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is available for this type of summation
    "pad_output": True,
}

# load data 
injection_params = load(f'{output_dir}/injection_params_T_lt_T_obs.npy')
events = len(injection_params)

M = 10**injection_params[:, 0]
mu = 10**injection_params[:, 1]
a = injection_params[:, 2]

p0 = injection_params[:, 3] # This is for _T_MAX
e0 = injection_params[:, 4] # This is for _T_MAX
Y0 = injection_params[:, 5] # This is for _T_MAX

dist = injection_params[:, 6] # distance in Gpc
qS = injection_params[:, 7] # Sky location polar angle in ecliptic coordinates.
phiS = injection_params[:, 8] # Sky location azimuthal angle in ecliptic coordinates
qK = injection_params[:, 9] # Initial BH spin polar angle in ecliptic coordinates.
phiK = injection_params[:, 10]  # Initial BH spin azimuthal angle in ecliptic coordinates.

Phi_phi0 = injection_params[:, 11] # This is for _T_MAX
Phi_theta0 = injection_params[:, 12] # This is for _T_MAX
Phi_r0 = injection_params[:, 13] # This is for _T_MAX
_T_INJECT_ = injection_params[:, 14] 

dt = 10.0  # seconds 
t0 = 10000.0 # default is 10000.0
order = 25 # default is 25

orbit_file_ESA = "../lisa-on-gpu/orbit_files/esa-trailing-orbits.h5" #ESA 
orbit_kwargs_ESA = dict(orbit_file=orbit_file_ESA)
tdi_gen = "1st generation" # 1st or 2nd or custom (see docs for custom)

index_lambda = 8  # ecliptic longitude within this list of parameters
index_beta = 7 # ecliptic latitude (or ecliptic polar angle) within this list of parameters.

tdi_kwargs_ESA = dict(orbit_kwargs=orbit_kwargs_ESA, order=order, tdi=tdi_gen, tdi_chan="AET")

gen_wave = GenerateEMRIWaveform("Pn5AAKWaveform", sum_kwargs=sum_kwargs, inspiral_kwargs=inspiral_kwargs)

traj = EMRIInspiral(func="pn5", enforce_schwarz_sep=True)

OBSERVING_WINDOW = args.OBSERVING_WINDOW

emri_lisa_MAX_SIGNAL_DURATION = ResponseWrapper(gen_wave, OBSERVING_WINDOW, dt, index_lambda, index_beta,
                                                t0=t0,
                                                flip_hx=True,  # set to True if waveform is h+ - ihx
                                                remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
                                                is_ecliptic_latitude=False,  # False if using polar angle (theta)
                                                remove_garbage=True,  # removes the beginning of the signal that has bad information
                                                use_gpu=use_gpu,
                                                **tdi_kwargs_ESA
                                                )

print(f"Response for {OBSERVING_WINDOW} years ready !")

# LOAD PSD
psd = cp.asarray(load(f'{output_dir}/PSD_LISA_{OBSERVING_WINDOW}_yr.npy'))
len_WINDOW = load(f'{output_dir}/len_WINDOW.npy')

# LOAD TRIM INDEX 
with h5py.File(f'{output_dir}/INDEX.h5', 'r') as f:
    INDEX_TRIM = array(f['indices'])

snrs_T = zeros(events)

AE_channel_gpu_final = cp.zeros((2, len_WINDOW)) # 2 is because we are using 2 channels 

for num_event in range(events):

    #########################################################################
    # Step 1: Calculate the waveform projection and SNR for whole T_max year.
    #########################################################################
    
    # generate the waveform 
    st = time.perf_counter()
 
    # this is right because this response function is calculated forward in time 
    # from the wrapper time above. It goes from p0, e0 to time defined in wrapper
 
    AET_MAX_SIGNAL_DURATION = emri_lisa_MAX_SIGNAL_DURATION(M[num_event], mu[num_event], a[num_event],
                                                            p0[num_event], e0[num_event], Y0[num_event],
                                                            dist[num_event], qS[num_event], phiS[num_event],
                                                            qK[num_event], phiK[num_event],
                                                            Phi_phi0[num_event], Phi_theta0[num_event], Phi_r0[num_event])

    AE_channel_gpu_MAX_SIGNAL_DURATION = cp.asarray(AET_MAX_SIGNAL_DURATION[0:2])
    
    #########################################################################
    # Step 2: Trim the waveform accordint to _T_INJECT_.
    #########################################################################

    AE_channel_gpu_final[:] = 0

    SIGNAL_IN_WINDOW = len(AE_channel_gpu_MAX_SIGNAL_DURATION[0, INDEX_TRIM[num_event]:])
    AE_channel_gpu_final[:, : SIGNAL_IN_WINDOW] = AE_channel_gpu_MAX_SIGNAL_DURATION[:, INDEX_TRIM[num_event]:]
   
    #########################################################################
    # Step 3: Calculate SNR for trimmed waveform 
    #########################################################################

    AE_channel_gpu_final_fd = cp.fft.rfft(AE_channel_gpu_final, axis=1)[:,1:] * dt
    AE_channel_gpu_final_freq = cp.fft.rfftfreq(AE_channel_gpu_final_fd.shape[1], dt)
    
    _snr_T_INJECT_ = (4 * (AE_channel_gpu_final_freq[1] - AE_channel_gpu_final_freq[0]) * ((AE_channel_gpu_final_fd.conj() * AE_channel_gpu_final_fd).real/psd[None,1:]).sum())**0.5
    
    et = time.perf_counter()

    print(num_event, _snr_T_INJECT_, et - st, _T_INJECT_[num_event])
    snrs_T[num_event] = _snr_T_INJECT_.get()

save(f'{output_dir}/snrs_T_lt_T_obs.npy', snrs_T)