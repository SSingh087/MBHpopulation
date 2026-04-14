import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

theta0_EMRI = {}
theta0_TDE = {}

with h5py.File('../fisher_analysis/true_data_EMRI.h5', 'r') as hf:
    for key in hf.keys():
        theta0_EMRI[key] = hf[key][()]

with h5py.File('../fisher_analysis/true_data_TDE_ZTF.h5', 'r') as hf:
    for key in hf.keys():
        theta0_TDE[key] = hf[key][()]


keys_to_save_EMRI = ["MBHspin", "lgMBH_mass"]
keys_to_save_TDE = ["mbh_6",]  # Define keys to save for TDE data

with h5py.File('./true_data_EMRI.h5', 'w') as hf:
    for key in keys_to_save_EMRI:
        hf.create_dataset(key, data=theta0_EMRI[key])

with h5py.File('./true_data_TDE_ZTF.h5', 'w') as hf:
    for key in keys_to_save_TDE:
        values = theta0_TDE[key]
        if key == 'mbh_6':  # Convert mbh_6 back to log10_M for consistency
            values = values + 6  # Convert mbh_6 to log10_M
            key = 'lgMBH_mass'  # Save under the same key as EMRI data for consistency
        hf.create_dataset(key, data=values)


# check saved data
with h5py.File('./true_data_EMRI.h5', 'r') as hf:
    for key in hf.keys():
        print(f"Saved EMRI dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}, type = {type(hf[key][()])}")
            
with h5py.File('./true_data_TDE_ZTF.h5', 'r') as hf:
    for key in hf.keys():
        print(f"Saved TDE dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}, type = {type(hf[key][()])}")

breakpoint()