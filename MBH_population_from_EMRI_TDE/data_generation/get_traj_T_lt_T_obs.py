#!/usr/bin/env python

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from numpy import load, save, log10, asarray, zeros, where, random, minimum

from few.waveform import GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper
import matplotlib.pyplot as plt
import cupy as cp
import time
import h5py
from few.trajectory.inspiral import EMRIInspiral

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="total events")
args = parser.parse_args()

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'/data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/{events_for_dir}_events/'

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

# this is how long the signal duration
# should be in the observing window
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

len_WINDOW = load(f'{output_dir}/len_WINDOW.npy')

index = zeros(events, dtype=int)

# with h5py.File(f'{output_dir}/new_injection_params_T_lt_T_obs.h5', 'w') as hf:
with h5py.File(f'{output_dir}/CHECKING_new_injection_params_T_lt_T_obs.h5', 'w') as hf:
    # Predefine dataset size (adjust dimensions as needed)
    dataset_size = (events, 15)  # Assuming 14 parameters per event
    traj_data = hf.create_dataset('params', shape=dataset_size, dtype='float64')

    for num_event in range(events):

        st = time.perf_counter()

        #########################################################################
        # Step 2: Find the INDEX of TRIM _T_INJECT_.
        #########################################################################

        frac = (_T_INJECT_[num_event] / OBSERVING_WINDOW)
        len_AE_channel_gpu_trunc = int(frac * len_WINDOW)
        index[num_event] = -len_AE_channel_gpu_trunc
        
        #########################################################################
        # Step 4: Calculate the trajectory from 0's and move forward in time
        #########################################################################
        t_gpu, p_gpu, e_gpu, Y_gpu, Phi_phi_gpu, Phi_r_gpu, Phi_theta_gpu = cp.asarray(
                                                    traj(M[num_event], mu[num_event], a[num_event],
                                                        p0[num_event], e0[num_event], Y0[num_event],
                                                        Phi_phi0=Phi_phi0[num_event], Phi_theta0=Phi_theta0[num_event], Phi_r0=Phi_r0[num_event],
                                                        T=OBSERVING_WINDOW)
                                                    )

        #############################################################################################
        # Step 5: Upsample the paraemeters to save the starting values which are in the LISA band
        #############################################################################################
        
        upsampled_t = cp.linspace(min(t_gpu), max(t_gpu), len_WINDOW)
        

        idx_start = cp.searchsorted(t_gpu, upsampled_t[index[num_event]], side='right') - 1  # Last index where t < upsampled_t[index]
        idx_end = cp.searchsorted(t_gpu, upsampled_t[index[num_event]], side='left')         # First index where t > upsampled_t[index]

        # Use searchsorted for upsampled_t trimming
        idx_trim_start = cp.searchsorted(upsampled_t, t_gpu[idx_start], side='left')
        idx_trim_end = cp.searchsorted(upsampled_t, t_gpu[idx_end], side='right')

        # Trim the upsampled_t array
        upsampled_t_trimmed = upsampled_t[idx_trim_start:idx_trim_end]

        # calculate the first point when LISA is launched even if its not in the band
        traj_index = (cp.where(upsampled_t_trimmed == upsampled_t[index[num_event]])[0][0]).get()

        # Find the corresponding lower and upper indices for interpolation
        # Use searchsorted to find indices in `t` where `upsampled_t_trimmed` values should be inserted

        idx_lower = cp.searchsorted(t_gpu, upsampled_t_trimmed, side='right') - 1  # Indices of the largest t <= upsampled_t_trimmed
        idx_upper = cp.clip(idx_lower + 1, 0, len(t_gpu) - 1)  # Indices of the smallest t > upsampled_t_trimmed


        # denominator = cp.where(denominator == 0, 1, denominator)  # Avoid division by zero

        # Extract the lower and upper values of t and p
        # Compute interpolated values for `upsampled_p`
        # Perform all interpolations on GPU and Transfer to CPU only once
        # Save only one value for GPU RAM issues

        req_upsampled_values = cp.array([
            (p_gpu[idx_lower] + (p_gpu[idx_upper] - p_gpu[idx_lower]) * (upsampled_t_trimmed - t_gpu[idx_lower]) / (t_gpu[idx_upper] - t_gpu[idx_lower]))[traj_index],
            (e_gpu[idx_lower] + (e_gpu[idx_upper] - e_gpu[idx_lower]) * (upsampled_t_trimmed - t_gpu[idx_lower]) / (t_gpu[idx_upper] - t_gpu[idx_lower]))[traj_index],
            (Y_gpu[idx_lower] + (Y_gpu[idx_upper] - Y_gpu[idx_lower]) * (upsampled_t_trimmed - t_gpu[idx_lower]) / (t_gpu[idx_upper] - t_gpu[idx_lower]))[traj_index],
            (Phi_phi_gpu[idx_lower] + (Phi_phi_gpu[idx_upper] - Phi_phi_gpu[idx_lower]) * (upsampled_t_trimmed - t_gpu[idx_lower]) / (t_gpu[idx_upper] - t_gpu[idx_lower]))[traj_index],
            (Phi_theta_gpu[idx_lower] + (Phi_theta_gpu[idx_upper] - Phi_theta_gpu[idx_lower]) * (upsampled_t_trimmed - t_gpu[idx_lower]) / (t_gpu[idx_upper] - t_gpu[idx_lower]))[traj_index],
            (Phi_r_gpu[idx_lower] + (Phi_r_gpu[idx_upper] - Phi_r_gpu[idx_lower]) * (upsampled_t_trimmed - t_gpu[idx_lower]) / (t_gpu[idx_upper] - t_gpu[idx_lower]))[traj_index]
        ]).get()
        
        traj_data[num_event, :] = [
            log10(M[num_event]), log10(mu[num_event]), a[num_event],
            req_upsampled_values[0], req_upsampled_values[1], req_upsampled_values[2],
            dist[num_event], qS[num_event], phiS[num_event],
            qK[num_event], phiK[num_event],
            req_upsampled_values[3], req_upsampled_values[4], req_upsampled_values[5],
            _T_INJECT_[num_event]
        ]
        
        et = time.perf_counter()
        print(num_event, index[num_event], _T_INJECT_[num_event], et-st)

# with h5py.File(f'{output_dir}/INDEX.h5', 'w') as h5f:
#     h5f.create_dataset('indices', data=index)