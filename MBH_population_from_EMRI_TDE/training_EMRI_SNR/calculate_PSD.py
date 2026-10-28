#!/usr/bin/env python
"""Compute and plot the LISA noise ASD for an EMRI observing window."""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from numpy import load, save
import numpy as np

from few.waveform import GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper
import matplotlib.pyplot as plt
from lisatools.sensitivity import SensitivityMatrix, A1TDISens, E1TDISens
import cupy as cp

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="total events")
args = parser.parse_args()

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")

output_dir = f'./data/PRE_TRAIN_DATA/{events_for_dir}_events/'

use_gpu = True

inspiral_kwargs = {
    "max_init_len": int(1e7),  # all of the trajectories will be well under max_init_len
}

sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is available for this type of summation
    "pad_output": True,
}

# load data 
injection_params = load(f'{output_dir}/injection_params_T_gt_T_obs.npy')
events = len(injection_params)

print(f"Loaded {events} events from {output_dir}")
print(f"Observing window: {args.OBSERVING_WINDOW} years")

# Helper function to validate orbital parameters
def is_valid_orbit(a, e0, Y0):
    """
    Check if orbital parameters are physically valid.
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Check bounds
    if not (0.0 < a <= 1.0):
        return False  # spin must be in (0, 1]
    if not (0.0 <= e0 < 1.0):
        return False  # eccentricity must be in [0, 1)
    if not (0.0 < Y0 < 1.0):
        return False  # inclination parameter must be in (0, 1)

    # Check separatrix: p must be greater than separatrix
    try:
        ps = get_separatrix(a, e0, Y0)
        if ps < 0 or ps > 100:  # sanity check on separatrix value
            return False
    except Exception as e:
        print(f"Separatrix check failed: {e}")
        return False
    
    return True


# Find a valid event to compute PSD
valid_event_idx = None
for idx in range(min(100, events)):  # check first 100 events
    a = injection_params[idx, 1]
    e0 = injection_params[idx, 3]
    Y0 = injection_params[idx, 4]
    
    if is_valid_orbit(a, e0, Y0):
        valid_event_idx = idx
        print(f"Found valid event at index {idx}: a={a:.6f}, e0={e0:.6f}, Y0={Y0:.6f}")
        break
    else:
        print(f"Event {idx} invalid: a={a:.6f}, e0={e0:.6f}, Y0={Y0:.6f}")

if valid_event_idx is None:
    raise ValueError("Could not find a valid event in the first 100 events. Check parameter ranges.")

# Extract parameters for the valid event
M = 10**injection_params[valid_event_idx, 0]
mu = 10
a = injection_params[valid_event_idx, 1]

p0 = injection_params[valid_event_idx, 2] # This is for _T_MAX
e0 = injection_params[valid_event_idx, 3] # This is for _T_MAX
Y0 = injection_params[valid_event_idx, 4] # This is for _T_MAX

dist = injection_params[valid_event_idx, 5] # distance in Gpc
qS = injection_params[valid_event_idx, 6] # Sky location polar angle in ecliptic coordinates.
phiS = injection_params[valid_event_idx, 7] # Sky location azimuthal angle in ecliptic coordinates
qK = injection_params[valid_event_idx, 8] # Initial BH spin polar angle in ecliptic coordinates.
phiK = injection_params[valid_event_idx, 9]  # Initial BH spin azimuthal angle in ecliptic coordinates.

Phi_phi0 = injection_params[valid_event_idx, 10] # This is for _T_MAX
Phi_theta0 = injection_params[valid_event_idx, 11] # This is for _T_MAX
Phi_r0 = injection_params[valid_event_idx, 12] # This is for _T_MAX

dt = 10.0  # seconds 
t0 = 10000.0 # default is 10000.0
order = 25 # default is 25

orbit_file_ESA = "../../lisa-on-gpu/orbit_files/esa-trailing-orbits.h5" #ESA 
orbit_kwargs_ESA = dict(orbit_file=orbit_file_ESA)
tdi_gen = "1st generation" # 1st or 2nd or custom (see docs for custom)

index_lambda = 8  # ecliptic longitude within this list of parameters
index_beta = 7 # ecliptic latitude (or ecliptic polar angle) within this list of parameters.

tdi_kwargs_ESA = dict(orbit_kwargs=orbit_kwargs_ESA, order=order, tdi=tdi_gen, tdi_chan="AET")

gen_wave = GenerateEMRIWaveform("Pn5AAKWaveform", sum_kwargs=sum_kwargs, inspiral_kwargs=inspiral_kwargs)

traj = EMRIInspiral(func="pn5")
# st = time.perf_counter()

OBSERVING_WINDOW = args.OBSERVING_WINDOW

# generate waveform once to get the psd which is the lendth of the observinf window
emri_lisa_OBSERVING_WINDOW = ResponseWrapper(gen_wave, OBSERVING_WINDOW, dt, index_lambda, index_beta,
                                            t0=t0,
                                            flip_hx=True,  # set to True if waveform is h+ - ihx
                                            remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
                                            is_ecliptic_latitude=False,  # False if using polar angle (theta)
                                            remove_garbage=True,  # removes the beginning of the signal that has bad information
                                            use_gpu=use_gpu,
                                            **tdi_kwargs_ESA
                                            )

print(f"Response for {OBSERVING_WINDOW} years ready !")

# CALCULATE PSD with error handling
try:
    channel_output_OBSERVING_WINDOW = emri_lisa_OBSERVING_WINDOW(M, mu, a, p0, e0, Y0,
                                                                dist, qS, phiS, qK, phiK,
                                                                Phi_phi0, Phi_theta0, Phi_r0)
    print(f"Successfully generated waveform for event {valid_event_idx}")
    
except ValueError as e:
    print(f"\n*** ERROR: Waveform generation failed for event {valid_event_idx} ***")
    print(f"Parameters: M={M:.2e}, mu={mu}, a={a:.6f}, p0={p0:.6f}, e0={e0:.6f}, Y0={Y0:.6f}")
    raise


AE_channel_gpu = cp.asarray(channel_output_OBSERVING_WINDOW[0:2])
# https://docs.cupy.dev/en/latest/reference/generated/cupy.fft.rfftfreq.html
data_f_arr_gpu = cp.fft.rfftfreq(AE_channel_gpu.shape[1], dt)
sens_mat_gpu = SensitivityMatrix(data_f_arr_gpu.get(), [A1TDISens, E1TDISens])
psd = cp.asarray(sens_mat_gpu.sens_mat[0])

plt.loglog(data_f_arr_gpu.get(), cp.sqrt(psd).get())
plt.xlabel('f(Hz)')
plt.ylabel('ASD')
plt.title(f'for {OBSERVING_WINDOW} year')
plt.savefig(f'ASD_LISA_{OBSERVING_WINDOW}_yr.png')

save(f'{output_dir}/PSD_LISA_{OBSERVING_WINDOW}_yr.npy', psd.get())
save(f'{output_dir}/len_WINDOW.npy',  len(AE_channel_gpu[0]))