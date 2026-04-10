import h5py
import numpy as np

PARAMS = ["redshift", "mbh_6", "stellar_mass", "eta", "alpha", "beta"]#, "t0_mjd_transient", "t0", "ra", "dec"]

if __name__ == "__main__":

    with h5py.File('./fisher_results_from_injection_ZTF.h5', 'r') as hf:
        for gal in hf.keys():
            print(f"\n[GALAXY] {gal}")
            for event in hf[gal].keys():
                print(f"  [EVENT] {event}")
                fisher_matrix = np.array(hf[gal][event]["fisher_matrix"])
                covariance_matrix = np.array(hf[gal][event]["covariance_matrix"])
                errors_on_params = np.sqrt(np.diag(covariance_matrix))
                print("    Errors on parameters:")
                for p, e in zip(PARAMS, errors_on_params):
                    print(f"      {p:15s} = {e:.4g}")

    hf.close()

    with h5py.File('./true_data_ZTF.h5', 'r') as hf:
        for key in PARAMS:
            data_array = np.array(hf[key])
            print(f"\n[DATA] '{key}': mean={data_array.mean():.4g}, std={data_array.std():.4g}, shape={data_array.shape}")
    hf.close()