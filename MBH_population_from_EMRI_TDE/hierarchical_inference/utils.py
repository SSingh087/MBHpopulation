#!/usr/bin/env python

import os, sys
# sys.path.insert(0, os.path.abspath('../poplar'))
sys.path.insert(0, os.path.abspath('../../poplar'))
from poplar.distributions import *
import numpy as np

import h5py
import seaborn as sns

device = "cpu"


FLAT_KEYMAP = {
    "A": {
        "lgMBH_mass": {"xc": "xc", "lam_schechter": "lam_schechter"},
        "MBHspin": {"alpha_1": "alpha_1_MBHSpin", "alpha_2": "alpha_2_MBHSpin",
                    "beta_1": "beta_1_MBHSpin", "beta_2": "beta_2_MBHSpin",
                    "weight": "weight_MBHSpin"},
        "e_pl": {"alpha_1": "alpha_1_e_pl", "alpha_2": "alpha_2_e_pl",
                 "beta_1": "beta_1_e_pl", "beta_2": "beta_2_e_pl",
                 "weight": "weight_e_pl"},
        "CO_mass": {"mu_2": "mu_sBH", "mu_1": "mu_star",
                    "sigma_2": "sigma_sBH", "sigma_1": "sigma_star",
                    "weight": "weight_COmass"},
        "z_gal": {"lam": "lam_z_gal"},
    },
    "B": {
        "lgMBH_mass": {"lam": "lambda_M"},
        "MBHspin": {"mu_1": "mu_1_MBHSpin", "sigma_1": "sigma_1_MBHSpin",
                    "mu_2": "mu_2_MBHSpin", "sigma_2": "sigma_2_MBHSpin",
                    "weight": "weight_MBHSpin"},
        "e_pl": {"mu_1": "mu_1_e_pl", "sigma_1": "sigma_1_e_pl",
                 "mu_2": "mu_2_e_pl", "sigma_2": "sigma_2_e_pl",
                 "weight": "weight_e_pl"},
        "CO_mass": {"mu_2": "mu_sBH", "mu_1": "mu_star",
                    "sigma_2": "sigma_sBH", "sigma_1": "sigma_star",
                    "weight": "weight_COmass"},
        "z_gal": {"lam": "lam_z_gal"},
    },
    "MIX": {},
}


def get_nested_from_flat(flat_dict, pop):
    """Build the nested {component: {subkey: value}} dict that
    PopulationDistribution.draw_samples() expects, from one flat sample
    row (e.g. {k: sdict[k][i] for k in sdict}).

    Returns (nested, missing_components):
      - nested: full nested dict, ready to pass to draw_samples(). Any
        hyperparameter this source didn't sample is backfilled with its
        injected truth value (needed because draw_samples always needs
        *some* value for every component -- it draws all 5 components
        in one call).
      - missing_components: list of component names (e.g. ["MBHspin",
        "e_pl"]) for which at least one required hyperparameter was NOT
        present in flat_dict. The caller should NOT treat cat[component]
        as a real predictive draw for these -- the source's data
        contains zero information about them, so any histogram built
        from cat[component] here is just truth-pinned catalogue noise,
        not a posterior predictive distribution. Skip these when
        accumulating PPD histograms.
    """
    tx = get_true_x(pop)
    keymap = FLAT_KEYMAP[pop]
    nested = {}
    missing_components = []
    for component, subkeymap in keymap.items():
        nested[component] = {}
        component_missing = False
        for subkey, flatname in subkeymap.items():
            if flatname in flat_dict:
                nested[component][subkey] = flat_dict[flatname]
            else:
                component_missing = True
                nested[component][subkey] = tx[component][subkey]
        if component_missing:
            missing_components.append(component)
    return nested, missing_components

pal = sns.color_palette(palette="colorblind").as_hex()

# (source, survey, label, color)
# survey="None" for EMRI-only runs, matching the directory naming
# convention f"{SOURCE}_pop_{POPULATION}_{SURVEY}".
SOURCE_SURVEY_COMBOS = [
    ("EMRI",     "None", "EMRI",            pal[4]),
    ("TDE",      "ZTF",  "TDE (ZTF)",       pal[2]),
    ("TDE",      "LSST", "TDE (LSST)",      pal[0]),
    ("TDE",      "ALL",  "TDE (ALL)",       pal[6]),
    ("EMRI_TDE", "ZTF",  "EMRI+TDE (ZTF)",  pal[3]),
    ("EMRI_TDE", "LSST", "EMRI+TDE (LSST)", pal[1]),
    ("EMRI_TDE", "ALL",  "EMRI+TDE (ALL)",  pal[5]),
]


def result_path(base_dir: str, source: str, survey: str, population: str) -> str:
    return f"{base_dir}/{source}_pop_{population}_{survey}/result.hdf5"


def load_samples_dict(path: str) -> dict:
    """Read every field actually present in this run's posterior_samples
    table, instead of assuming a fixed schema. Raises on missing/malformed
    files -- caller decides whether to skip."""
    with h5py.File(path, "r") as hf:
        ds = hf["posterior_samples"]
        return {name: np.asarray(ds[name]) for name in ds.dtype.names}


def load_available_combos(base_dir: str, population: str,
                           combos=SOURCE_SURVEY_COMBOS) -> list:
    """Try to load every (source, survey) combo's result.hdf5. Any combo
    that fails to load is skipped with a warning -- it does NOT stop the
    other combos from loading, and does NOT require touching this
    function or any plotting script.

    Returns a list of dicts (only the combos that loaded successfully):
        {"source", "survey", "label", "color", "samples"}
    in the same order as `combos`.
    """
    available = []
    for source, survey, label, color in combos:
        path = result_path(base_dir, source, survey, population)
        try:
            samples = load_samples_dict(path)
        except (FileNotFoundError, OSError, KeyError) as exc:
            print(f"[skip] {label} (pop {population}): could not load {path} ({exc})")
            continue
        if not samples:
            print(f"[skip] {label} (pop {population}): {path} has no posterior_samples fields")
            continue
        available.append({
            "source": source, "survey": survey, "label": label,
            "color": color, "samples": samples,
        })
    if not available:
        print(f"[warn] No combos loaded at all for pop {population} under {base_dir}")
    return available
 
def make_dist(pop):

    if pop == "A":
        DIST = {
                "lgMBH_mass": FixedLimitSchechterFunction([4, 7], device=device),
                "MBHspin" : DoubleBetaDistribution([0.01, 0.99], device=device),
                "CO_mass" : DoubleGaussianDistribution([0.1, 15], device=device),
                "e_pl" : DoubleBetaDistribution([0.01, 0.99], device=device),
                "z_gal" : FixedLimitsPowerLaw([0.01, 8.0], device=device)
            }   
        return PopulationDistribution(distributions=DIST, data=None)

    elif pop =="B":
        DIST = {
                "lgMBH_mass": FixedLimitsPowerLaw([4, 7], device=device),
                "MBHspin" : DoubleGaussianDistribution([0.01, 0.99], device=device),
                "CO_mass" : DoubleGaussianDistribution([0.1, 15], device=device),
                "e_pl" : DoubleGaussianDistribution([0.01, 0.99], device=device),
                "z_gal" : FixedLimitsPowerLaw([0.01, 8.0], device=device)
            }
        return PopulationDistribution(distributions=DIST, data=None)

    elif pop=="MIX":
        DIST = {

        }
        return PopulationDistribution(distributions=DIST, data=None)

def get_true_x(pop):

    if pop == "A":
        return {
                "lgMBH_mass": {"xc": 6.5, "lam_schechter": 10.0},
                "MBHspin": {'alpha_1': 2.5, 'alpha_2': 9.5, 'beta_1': 5.0, 'beta_2': 5.0, 'weight': 0.7},
                "CO_mass": {"mu_1": 1.0, "sigma_1": 0.5, "mu_2": 10.0, "sigma_2": 0.5, "weight": 0.5},
                "e_pl": {'alpha_1': 2.0, 'alpha_2': 4.0, 'beta_1': 7.0, 'beta_2': 6.0, 'weight': 0.70},
                "z_gal": {"lam": -1.5}
            }

    if pop == "B":
        return{
                "lgMBH_mass": {"lam": -1.5},
                "MBHspin": {'mu_1': 0.3, 'sigma_1': 0.1, 'mu_2': 0.7, 'sigma_2': 0.1, 'weight': 0.7},
                "CO_mass": {"mu_1": 1.0, "sigma_1": 0.5, "mu_2": 10.0, "sigma_2": 0.5, "weight": 0.5},
                "e_pl": {"mu_1": 0.15, "sigma_1": 0.07, "mu_2": 0.35, "sigma_2": 0.12, "weight": 0.70},
                "z_gal": {"lam": -1.5}
            }

    if pop == "MIX":
        return{

                }

    raise ValueError(f"Unknown population: {pop}")

# def get_posterior_samples(pop, hf):
    
#     if pop == "A":
#         return {
#                 "xc": np.array(hf.get('posterior_samples')['xc']),
#                 "lam_schechter": np.array(hf.get('posterior_samples')['lam_schechter']),

#                 "alpha_1_MBHSpin": np.array(hf.get('posterior_samples')['alpha_1_MBHSpin']),
#                 "alpha_2_MBHSpin": np.array(hf.get('posterior_samples')['alpha_2_MBHSpin']),
#                 "beta_1_MBHSpin": np.array(hf.get('posterior_samples')['beta_1_MBHSpin']),
#                 "beta_2_MBHSpin": np.array(hf.get('posterior_samples')['beta_2_MBHSpin']),
#                 "weight_MBHSpin": np.array(hf.get('posterior_samples')['weight_MBHSpin']),
                
#                 "alpha_1_e_pl": np.array(hf.get('posterior_samples')['alpha_1_e_pl']),
#                 "alpha_2_e_pl": np.array(hf.get('posterior_samples')['alpha_2_e_pl']),
#                 "beta_1_e_pl": np.array(hf.get('posterior_samples')['beta_1_e_pl']),
#                 "beta_2_e_pl": np.array(hf.get('posterior_samples')['beta_2_e_pl']),
#                 "weight_e_pl": np.array(hf.get('posterior_samples')['weight_e_pl']),                

#                 "mu_sBH": np.array(hf.get('posterior_samples')['mu_sBH']),
#                 "mu_star": np.array(hf.get('posterior_samples')['mu_star']),
#                 "sigma_sBH": np.array(hf.get('posterior_samples')['sigma_sBH']),
#                 "sigma_star": np.array(hf.get('posterior_samples')['sigma_star']),
#                 "weight_COmass": np.array(hf.get('posterior_samples')['weight_COmass']),
               
#                 "lam_z_gal": np.array(hf.get('posterior_samples')['lam_z_gal'])
#             }

#     if pop == "B":
#         return{
#                 "lambda_M": np.array(hf.get('posterior_samples')['lambda_M']),
                
#                 "mu_1_MBHSpin": np.array(hf.get('posterior_samples')['mu_1_MBHSpin']),
#                 "sigma_1_MBHSpin": np.array(hf.get('posterior_samples')['sigma_1_MBHSpin']),
#                 "mu_2_MBHSpin": np.array(hf.get('posterior_samples')['mu_2_MBHSpin']),
#                 "sigma_2_MBHSpin": np.array(hf.get('posterior_samples')['sigma_2_MBHSpin']),
#                 "weight_MBHSpin": np.array(hf.get('posterior_samples')['weight_MBHSpin']),
                
#                 "mu_1_e_pl":    np.array(hf.get('posterior_samples')['mu_1_e_pl']),
#                 "sigma_1_e_pl": np.array(hf.get('posterior_samples')['sigma_1_e_pl']),
#                 "mu_2_e_pl":    np.array(hf.get('posterior_samples')['mu_2_e_pl']),
#                 "sigma_2_e_pl": np.array(hf.get('posterior_samples')['sigma_2_e_pl']),
#                 "weight_e_pl":  np.array(hf.get('posterior_samples')['weight_e_pl']),

#                 "mu_sBH": np.array(hf.get('posterior_samples')['mu_sBH']),
#                 "mu_star": np.array(hf.get('posterior_samples')['mu_star']),
#                 "sigma_sBH": np.array(hf.get('posterior_samples')['sigma_sBH']),
#                 "sigma_star": np.array(hf.get('posterior_samples')['sigma_star']),
#                 "weight_COmass": np.array(hf.get('posterior_samples')['weight_COmass']),
                
#                 "lam_z_gal": np.array(hf.get('posterior_samples')['lam_z_gal'])
#                 }

#     if pop == "MIX":
#         return{

#                 }

def load_posterior_samples_dynamic(hf):
    """Read whatever fields actually exist in posterior_samples, instead of
    assuming a fixed schema. EMRI-only / TDE-only runs legitimately have
    fewer fields than EMRI_TDE runs."""
    ds = hf['posterior_samples']
    return {name: np.array(ds[name]) for name in ds.dtype.names}

def get_latex_labels(pop):

    if pop == "A":
        return {
            "xc": r"$x_c$",
            "lam_schechter": r"$\lambda_{x_c}$",

            "alpha_1_MBHSpin": r"$\alpha_{1,a}$",
            "alpha_2_MBHSpin": r"$\alpha_{2,a}$",
            "beta_1_MBHSpin": r"$\beta_{1,a}$",
            "beta_2_MBHSpin": r"$\beta_{2,a}$",
            "weight_MBHSpin": r"$w_a$",

            "alpha_1_e_pl": r"$\alpha_{1,e}$",
            "alpha_2_e_pl": r"$\alpha_{2,e}$",
            "beta_1_e_pl": r"$\beta_{1,e}$",
            "beta_2_e_pl": r"$\beta_{2,e}$",
            "weight_e_pl": r"$w_e$",

            "mu_sBH": r"$\mu_{\rm sBH}$",
            "mu_star": r"$\mu_{\star}$",
            "sigma_sBH": r"$\sigma_{\rm sBH}$",
            "sigma_star": r"$\sigma_{\star}$",
            "weight_COmass": r"$w_{\rm CO}$",

            "lam_z_gal": r"$\lambda_z$"
        }

    if pop == "B":
        return {
            "lambda_M": r"$\lambda_M$",

            "mu_1_MBHSpin": r"$\mu_{1,a}$",
            "sigma_1_MBHSpin": r"$\sigma_{1,a}$",
            "mu_2_MBHSpin": r"$\mu_{2,a}$",
            "sigma_2_MBHSpin": r"$\sigma_{2,a}$",
            "weight_MBHSpin": r"$w_a$",

            "mu_1_e_pl": r"$\mu_{1,e}$",
            "sigma_1_e_pl": r"$\sigma_{1,e}$",
            "mu_2_e_pl": r"$\mu_{2,e}$",
            "sigma_2_e_pl": r"$\sigma_{2,e}$",
            "weight_e_pl": r"$w_e$",

            "mu_sBH": r"$\mu_{\rm sBH}$",
            "mu_star": r"$\mu_{\star}$",
            "sigma_sBH": r"$\sigma_{\rm sBH}$",
            "sigma_star": r"$\sigma_{\star}$",
            "weight_COmass": r"$w_{\rm CO}$",

            "lam_z_gal": r"$\lambda_z$"
        }

    if pop == "MIX":
        return {}
    
def get_true_x_flat(pop):

    tx = get_true_x(pop)

    if pop == "A":
        return {
            "xc": tx["lgMBH_mass"]["xc"],
            "lam_schechter": tx["lgMBH_mass"]["lam_schechter"],

            "alpha_1_MBHSpin": tx["MBHspin"]["alpha_1"],
            "alpha_2_MBHSpin": tx["MBHspin"]["alpha_2"],
            "beta_1_MBHSpin":  tx["MBHspin"]["beta_1"],
            "beta_2_MBHSpin":  tx["MBHspin"]["beta_2"],
            "weight_MBHSpin":  tx["MBHspin"]["weight"],

            "alpha_1_e_pl": tx["e_pl"]["alpha_1"],
            "alpha_2_e_pl": tx["e_pl"]["alpha_2"],
            "beta_1_e_pl":  tx["e_pl"]["beta_1"],
            "beta_2_e_pl":  tx["e_pl"]["beta_2"],
            "weight_e_pl":  tx["e_pl"]["weight"],

            "mu_sBH":        tx["CO_mass"]["mu_2"],
            "mu_star":       tx["CO_mass"]["mu_1"],
            "sigma_sBH":     tx["CO_mass"]["sigma_2"],
            "sigma_star":    tx["CO_mass"]["sigma_1"],
            "weight_COmass": tx["CO_mass"]["weight"],

            "lam_z_gal": tx["z_gal"]["lam"]
        }

    if pop == "B":
        return {
            "lambda_M": tx["lgMBH_mass"]["lam"],

            "mu_1_MBHSpin":    tx["MBHspin"]["mu_1"],
            "sigma_1_MBHSpin": tx["MBHspin"]["sigma_1"],
            "mu_2_MBHSpin":    tx["MBHspin"]["mu_2"],
            "sigma_2_MBHSpin": tx["MBHspin"]["sigma_2"],
            "weight_MBHSpin":  tx["MBHspin"]["weight"],

            "mu_1_e_pl":    tx["e_pl"]["mu_1"],
            "sigma_1_e_pl": tx["e_pl"]["sigma_1"],
            "mu_2_e_pl":    tx["e_pl"]["mu_2"],
            "sigma_2_e_pl": tx["e_pl"]["sigma_2"],
            "weight_e_pl":  tx["e_pl"]["weight"],

            "mu_sBH":        tx["CO_mass"]["mu_2"],
            "mu_star":       tx["CO_mass"]["mu_1"],
            "sigma_sBH":     tx["CO_mass"]["sigma_2"],
            "sigma_star":    tx["CO_mass"]["sigma_1"],
            "weight_COmass": tx["CO_mass"]["weight"],

            "lam_z_gal": tx["z_gal"]["lam"]
        }

    if pop == "MIX":
        return {}

class PopulationDistribution:
    # this will work until we make mass dependance explicit.
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, size):
        out = {}
        lgMBH_mass_samples = self.distributions['lgMBH_mass'].draw_samples(**x['lgMBH_mass'], size=size)
        MBHspin_samples = self.distributions['MBHspin'].draw_samples(**x['MBHspin'], size=size)
        COMass_samples = self.distributions['CO_mass'].draw_samples(**x['CO_mass'], size=size)
        e_pl_samples = self.distributions['e_pl'].draw_samples(**x['e_pl'], size=size)
        z_gal_samples = self.distributions['z_gal'].draw_samples(**x['z_gal'], size=size)
        
        out['lgMBH_mass'] = lgMBH_mass_samples
        out['MBHspin'] = MBHspin_samples
        out['CO_mass'] = COMass_samples
        out['e_pl'] = e_pl_samples
        out['z_gal'] = z_gal_samples
        return out