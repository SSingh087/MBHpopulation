import sys, os
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
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


keys_to_save_EMRI = ["MBHspin", "lgMBH_mass"]
keys_to_save_TDE = ["mbh_6"]  # Define keys to save for TDE data

loc = f'./data/inference_data/{args.GALAXIES}'
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
        plt.hist(x, bins=40, color='steelblue', histtype='step')

        plt.xlabel(key_latex_map.get(key, key), fontsize=16)
        plt.ylabel("Counts", fontsize=16)
        plt.title(f"{source}: Distribution of {key_latex_map.get(key, key)}", fontsize=18)

        plt.tight_layout()
        plt.savefig(f'{loc}/{source}_1D_{key}.png')
        plt.close()

theta0_EMRI_final = {}
theta0_TDE_final_ZTF = {}
theta0_TDE_final_LSST = {}



print("Inspecting EMRI data:")
with h5py.File(f'./data/fisher_data/{args.GALAXIES}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r') as hf:
    for key in hf.keys():
        print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
        theta0_EMRI[key] = hf[key][()]

with h5py.File(f'{loc}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w') as hf:
    for key in keys_to_save_EMRI:
        theta0_EMRI_final[key] = theta0_EMRI[key]  # Store in final dict for later plotting
        print(f"[SAVING] '{key}' to EMRI dataset...")
        hf.create_dataset(key, data=theta0_EMRI[key])

for survey in ['ZTF', 'LSST']:
    print(f"\nInspecting TDE data for survey '{survey}':")
    with h5py.File(f'./data/fisher_data/{args.GALAXIES}/true_data_TDE_{survey}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r') as hf:
        for key in hf.keys():
            print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
            theta0_TDE[key] = hf[key][()]

    # with h5py.File(f'./data/fisher_data/{args.GALAXIES}/fisher_results_TDE_{survey}.h5', 'r') as fisher_tde_file:
    #     for gal in fisher_tde_file.keys():
    #         print(f"Galaxy '{gal}' has attributes:")
    #         for attr_key in fisher_tde_file[gal].attrs.keys():
    #             print(f"  - {attr_key}: {fisher_tde_file[gal].attrs[attr_key]}")

    with h5py.File(f'{loc}/true_data_TDE_{survey}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w') as hf:
        for key in keys_to_save_TDE:
            values = theta0_TDE[key]
            if key == 'mbh_6':  # Convert mbh_6 back to log10_M for consistency
                values = values + 6  # Convert mbh_6 to log10_M
                key = 'lgMBH_mass'  # Save under the same key as EMRI data for consistency
            if survey == 'ZTF':
                theta0_TDE_final_ZTF[key] = values  # Store in final dict for later plotting
                print(f"[SAVING] '{key}' to TDE dataset for ZTF...")
            else:
                theta0_TDE_final_LSST[key] = values  # Store in final dict for later plotting
                print(f"[SAVING] '{key}' to TDE dataset for LSST...")
            hf.create_dataset(key, data=values)

    hf.close()
    

# def plot_comparison_histogram(key, loc='.', x_label=None, **kwargs_sources):
#         """
#         Plot comparison histograms for different datasets.

#         Parameters:
#             key (str): The key/column to plot from each dataset.
#             loc (str): Directory path to save the figure.
#             **kwargs_sources: Arbitrary keyword arguments of the form name=dataframe,
#                             where `name` is the label and `dataframe` contains the key.
#         """
#         plt.figure(figsize=(8,6))
        
#         combined_data = []
        
#         for label, df in kwargs_sources.items():
#             if key not in df:
#                 raise ValueError(f"Key '{key}' not found in dataframe '{label}'")
#             data = df[key].mean(axis=1)
#             combined_data.append(data)
#             plt.hist(data, bins=40, histtype='step', label=label)
        
#         # Plot combined histogram
#         combined_array = np.concatenate(combined_data)
#         plt.hist(combined_array, bins=40, color='black', histtype='step', linestyle='--', linewidth=2, label='Combined')
        
#         # Labels and styling
#         plt.xlabel(x_label, fontsize=16)
#         plt.ylabel("Counts", fontsize=16)
#         plt.legend()
#         plt.tight_layout()
        
#         # Save figure
#         plt.savefig(f'{loc}/{key}_comparison_histogram.png')
#         plt.close()


loc = f'./data/inference_data/{args.GALAXIES}'
kwargs_sources = {
    "EMRI": theta0_EMRI_final,
    "TDE (ZTF)": theta0_TDE_final_ZTF,
    "TDE (LSST)": theta0_TDE_final_LSST
}
Plotting.plot_comparison_histogram(key="lgMBH_mass", loc=loc, x_label=r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$", **kwargs_sources)