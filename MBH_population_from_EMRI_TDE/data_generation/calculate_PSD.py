#!/usr/bin/env python

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from numpy import load, save

from few.waveform import GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper
import matplotlib.pyplot as plt
from lisatools.sensitivity import SensitivityMatrix, A1TDISens, E1TDISens
import cupy as cp

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
injection_params = load(f'{output_dir}/injection_params_T_gt_T_obs.npy')
events = len(injection_params)

M = 10**injection_params[0, 0]
mu = 10**injection_params[0, 1]
a = injection_params[0, 2]

p0 = injection_params[0, 3] # This is for _T_MAX
e0 = injection_params[0, 4] # This is for _T_MAX
Y0 = injection_params[0, 5] # This is for _T_MAX

dist = injection_params[0, 6] # distance in Gpc
qS = injection_params[0, 7] # Sky location polar angle in ecliptic coordinates.
phiS = injection_params[0, 8] # Sky location azimuthal angle in ecliptic coordinates
qK = injection_params[0, 9] # Initial BH spin polar angle in ecliptic coordinates.
phiK = injection_params[0, 10]  # Initial BH spin azimuthal angle in ecliptic coordinates.

Phi_phi0 = injection_params[0, 11] # This is for _T_MAX
Phi_theta0 = injection_params[0, 12] # This is for _T_MAX
Phi_r0 = injection_params[0, 13] # This is for _T_MAX

# this is how long the signal duration
# should be in the observing window
_T_INJECT_ = injection_params[0, 14] 

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

# CALCULATE PSD
channel_output_OBSERVING_WINDOW = emri_lisa_OBSERVING_WINDOW(M, mu, a, p0, e0, Y0,
                                                            dist, qS, phiS, qK, phiK,
                                                            Phi_phi0, Phi_theta0, Phi_r0)


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