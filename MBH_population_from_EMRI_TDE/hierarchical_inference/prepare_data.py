"""Convert the Fisher-analysis catalogues into numpy and torch inputs for inference."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import argparse
from utils import Plotting

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--is_clustering", type=int, default=0)
args = parser.parse_args()

theta0_EMRI = {}
theta0_TDE = {}

theta0_EMRI_final = {}
theta0_TDE_final = {}


keys_to_save_EMRI = ["MBHspin", "lgMBH_mass", "CO_mass", "z_gal"]  # Define keys to save for EMRI data
keys_to_save_TDE = ["mbh_6", "stellar_mass", "redshift"]  # Define keys to save for TDE data

loc = f'./data/inference_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

theta0_EMRI_final = {}
theta0_TDE_final_ZTF = {}
theta0_TDE_final_LSST = {}

def to_numpy(x):
    return np.asarray(x, dtype=np.float32)

def to_torch(x, device="cpu"):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


print("=======================")
print("Inspecting EMRI data:")
print("=======================")

with h5py.File(f'./data/fisher_data/{args.GALAXIES}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r') as hf, h5py.File(f'{loc}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w') as hf_out:
    for key in hf.keys():
        print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
        theta0_EMRI[key] = hf[key][()]
        if key in keys_to_save_EMRI:
            theta0_EMRI_final[key] = theta0_EMRI[key]  # Store in final dict for later plotting
            print(f"[SAVING] '{key}' to EMRI dataset...")
            hf_out.create_dataset(key, data=theta0_EMRI[key])
        else:
            print(f"[SKIPPING] '{key}' not in keys_to_save_EMRI, skipping saving to EMRI dataset.")

# Save EMRI data as torch tensors separately
print("\n[SAVING] EMRI torch data...")
torch.save({key: to_torch(theta0_EMRI_final[key]) for key in theta0_EMRI_final.keys()}, 
           f'{loc}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt') 



for survey in ['ZTF', 'LSST']:
    print("===============================================")
    print(f"Inspecting TDE data for survey '{survey}':")
    print("===============================================")

    with h5py.File(f'./data/fisher_data/{args.GALAXIES}/true_data_TDE_{survey}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r') as hf, h5py.File(f'{loc}/true_data_TDE_{survey}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w') as hf_out:
        for key in hf.keys():
            print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
            theta0_TDE[key] = hf[key][()]

            if key in keys_to_save_TDE:
                saved_key = key  # Keep track of original key
                data = theta0_TDE[key].copy()
                
                if key == 'mbh_6':  # Convert mbh_6 back to log10_M for consistency
                    data = data + 6  # Convert mbh_6 to log10_M
                    saved_key = 'lgMBH_mass'
                elif key == 'stellar_mass':
                    saved_key = 'CO_mass'
                elif key == 'redshift':
                    saved_key = 'z_gal'

                if survey == 'ZTF':
                    theta0_TDE_final_ZTF[saved_key] = data
                    print(f"[SAVING] '{saved_key}' to TDE dataset for ZTF...")
                else:
                    theta0_TDE_final_LSST[saved_key] = data
                    print(f"[SAVING] '{saved_key}' to TDE dataset for LSST...")
                hf_out.create_dataset(saved_key, data=data)
                # Also save torch version
                hf_out.create_dataset(f'{saved_key}_torch', data=to_numpy(to_torch(data)))

# Save TDE torch data separately for each survey
print("\n[SAVING] TDE torch data for ZTF...")
torch.save({key: to_torch(theta0_TDE_final_ZTF[key]) for key in theta0_TDE_final_ZTF.keys()}, 
           f'{loc}/true_data_TDE_ZTF_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt')

print("[SAVING] TDE torch data for LSST...")
torch.save({key: to_torch(theta0_TDE_final_LSST[key]) for key in theta0_TDE_final_LSST.keys()}, 
           f'{loc}/true_data_TDE_LSST_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt')

kwargs_sources = {
    "EMRI": theta0_EMRI_final,
    "TDE (ZTF)": theta0_TDE_final_ZTF,
    "TDE (LSST)": theta0_TDE_final_LSST
}

Plotting.plot_comparison_histogram(key="lgMBH_mass", loc=loc, x_label=r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$", **kwargs_sources)
Plotting.plot_comparison_histogram(key="MBHspin", loc=loc, x_label=r"$a$", **kwargs_sources)
Plotting.plot_comparison_histogram(key="CO_mass", loc=loc, x_label=r"$m_{\mathrm{CO}}$", **kwargs_sources)
# Plotting.plot_comparison_histogram(key="e0", loc=loc, x_label=r"$e_0$", **kwargs_sources)
Plotting.plot_comparison_histogram(key="z_gal", loc=loc, x_label=r"$z$", **kwargs_sources)