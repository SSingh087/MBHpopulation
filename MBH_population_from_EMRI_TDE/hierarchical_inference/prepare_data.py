import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
args = parser.parse_args()

theta0_EMRI = {}
theta0_TDE = {}

theta0_EMRI_final = {}
theta0_TDE_final = {}


keys_to_save_EMRI = ["MBHspin", "lgMBH_mass"]
keys_to_save_TDE = ["mbh_6"]  # Define keys to save for TDE data

loc = f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/inference_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

key_latex_map = {
    "e0": r"$e_0$",
    "lgMBH_mass": r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$",
    "MBHspin": r"$a$",
    # "z_gal": r"$z$",
}


def plot_1D_histogram(theta, source):
    for key, values in theta.items():
        x = values.mean(axis=1)

        plt.figure(figsize=(8,6))
        plt.hist(x, bins=40, color='steelblue', edgecolor='black', alpha=0.7)

        plt.xlabel(key_latex_map.get(key, key), fontsize=16)
        plt.ylabel("Counts", fontsize=16)
        plt.title(f"{source}: Distribution of {key_latex_map.get(key, key)}", fontsize=18)

        plt.tight_layout()
        plt.savefig(f'{loc}/{source}_1D_{key}.png')
        plt.close()
        
print("Inspecting EMRI data:")
with h5py.File(f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/fisher_data/{args.GALAXIES}/true_data_EMRI.h5', 'r') as hf:
    for key in hf.keys():
        print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
        theta0_EMRI[key] = hf[key][()]

with h5py.File(f'{loc}/true_data_EMRI.h5', 'w') as hf:
    for key in keys_to_save_EMRI:
        theta0_EMRI_final[key] = theta0_EMRI[key]  # Store in final dict for later plotting
        print(f"[SAVING] '{key}' to EMRI dataset...")
        hf.create_dataset(key, data=theta0_EMRI[key])

# for survey in ['ZTF', 'LSST']:
for survey in ['ZTF']:
    print(f"\nInspecting TDE data for survey '{survey}':")
    with h5py.File(f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/fisher_data/{args.GALAXIES}/true_data_TDE_{survey}.h5', 'r') as hf:
        for key in hf.keys():
            print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
            theta0_TDE[key] = hf[key][()]

    # with h5py.File(f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/fisher_data/{args.GALAXIES}/fisher_results_TDE_{survey}.h5', 'r') as fisher_tde_file:
    #     for gal in fisher_tde_file.keys():
    #         print(f"Galaxy '{gal}' has attributes:")
    #         for attr_key in fisher_tde_file[gal].attrs.keys():
    #             print(f"  - {attr_key}: {fisher_tde_file[gal].attrs[attr_key]}")

    with h5py.File(f'{loc}/true_data_TDE_{survey}.h5', 'w') as hf:
        for key in keys_to_save_TDE:
            values = theta0_TDE[key]
            if key == 'mbh_6':  # Convert mbh_6 back to log10_M for consistency
                values = values + 6  # Convert mbh_6 to log10_M
                key = 'lgMBH_mass'  # Save under the same key as EMRI data for consistency
            theta0_TDE_final[key] = values  # Store in final dict for later plotting
            print(f"[SAVING] '{key}' to TDE dataset...")
            hf.create_dataset(key, data=values)

    hf.close()
    
    plot_1D_histogram(theta0_EMRI_final, source='EMRI')
    plot_1D_histogram(theta0_TDE_final, source=f'TDE_{survey}')