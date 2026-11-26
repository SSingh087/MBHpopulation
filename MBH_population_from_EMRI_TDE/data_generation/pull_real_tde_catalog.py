"""Build the catalogue of real ZTF TDE peak epochs (combined ZTF-I and ZTF-II sample, 38 events) used for the observing-window comparison."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR
import numpy as np

# Combined ZTF-I & ZTF-II TDE sample: name -> t_peak (MJD).
# Primary references cited inline per object:
# - van Velzen et al. 2021 (ApJ 908, 4; arXiv:2001.01409) [vV21]
# - Hammerstein et al. 2023 (ApJ 942, 9; arXiv:2203.01461) [H23]
# - Yao et al. 2023 (ApJL 955, L6; arXiv:2303.06523) [Y23]
# - ZTF Bright Transient Survey (BTS) Explorer: Fremling et al. 2020 (arXiv:1910.12973),
#   Perley et al. 2020 (arXiv:2009.01242), Rehemtulla et al. 2024 (arXiv:2401.15167) [BTS]
# - Langis, Liodakis, Koljonen et al. 2025 compilation of optical TDEs (arXiv:2506.05476) [K25]
ZTF_TDE_TPEAK_MJD = {
    # --- ZTF-I & Early ZTF-II Sample (43 events) ---
    "AT2018zr":   58185.00,  # vV21, H23, Y23 (PS18kh)
    "AT2018iih":  58451.13,  # vV21, H23, Y23
    "AT2018jbv":  58470.36,  # vV21, H23, Y23
    "AT2018lna":  58507.31,  # vV21, H23, Y23
    "AT2019bhf":  58544.78,  # vV21, H23, Y23
    "AT2019azh":  58561.39,  # vV21, H23, Y23
    "AT2019cmw":  58588.82,  # H23, Y23
    "AT2019dsg":  58606.97,  # vV21, H23, Y23; Stein et al. 2021 (arXiv:2005.05340)
    "AT2019ehz":  58618.69,  # vV21, H23, Y23
    "AT2019qiz":  58766.50,  # Nicholl et al. 2020 (arXiv:2006.02454); vV21, H23, Y23
    "AT2019vcb":  58819.83,  # H23, Y23; Quintin et al. 2023 (arXiv:2306.00438)
    "AT2020pj":   58866.42,  # H23, Y23
    "AT2020ocn":  58972.00,  # vV21, H23
    "AT2020neh":  59030.93,  # Angus et al. 2022 (arXiv:2209.00018); H23, Y23
    "AT2020mot":  59082.04,  # H23, Y23; Liodakis et al. 2023 (arXiv:2209.06227)
    "AT2020ysg":  59094.32,  # H23, Y23
    "AT2020vdq":  59113.09,  # Y23; Somalwar et al. 2023 (arXiv:2310.03782)
    "AT2020wey":  59155.84,  # Charalampopoulos et al. 2023 (arXiv:2209.12913); H23, Y23
    "AT2020vwl":  59166.88,  # Y23; Goodwin et al. 2023 (arXiv:2303.00912)
    "AT2020yue":  59179.44,  # Y23
    "AT2020zso":  59184.00,  # Wevers et al. 2022 (arXiv:2202.08268); H23
    "AT2020acka": 59217.15,  # Y23
    "AT2021axu":  59252.50,  # Y23
    "AT2021crk":  59273.90,  # Y23
    "AT2021ehb":  59314.51,  # Yao et al. 2022 (arXiv:2206.12713); Y23
    "AT2021jjm":  59327.68,  # Y23
    "AT2021mhg":  59370.28,  # Y23; Somalwar et al. 2023 (arXiv:2310.03782)
    "AT2021qth":  59401.88,  # Y23
    "AT2021nwa":  59402.51,  # Y23
    "AT2021sdu":  59419.36,  # Y23
    "AT2021uqv":  59446.39,  # Y23
    "AT2021utq":  59457.51,  # Y23
    "AT2021yte":  59484.99,  # Y23
    "AT2021yzv":  59511.50,  # Y23
    "AT2022cmc":  59622.48,  # Andreoni et al. 2022 (arXiv:2211.16530); Pasham et al. 2023

    # --- Added ZTF-II & BTS Post-2021 Detections (2022–2025; unverified entries removed) ---
    "AT2022wtn":  59894.73,  # Onori et al. 2025 (arXiv:2504.21686); K25
    "AT2024tvd":  60572.67,  # Yao et al. 2025 (ApJL 985, L48); Guolo et al. 2026 (arXiv:2602.12272)
    "AT2025abcr": 61006.75, # BTS / TNS ZTF Classification (2025)
}

assert len(ZTF_TDE_TPEAK_MJD) == 38, \
    f"expected 38 ZTF TDEs, got {len(ZTF_TDE_TPEAK_MJD)}"

# Sort chronologically by peak MJD (change key to None for alphabetical sorting)
names = np.array(sorted(ZTF_TDE_TPEAK_MJD.keys(), key=lambda k: ZTF_TDE_TPEAK_MJD[k]))
discovery_mjd = np.array([ZTF_TDE_TPEAK_MJD[n] for n in names])

print(f"Loaded {len(discovery_mjd)} ZTF TDE peak epochs (2018–2025).")
print(f"Names: {names.tolist()}")

# ------------------------------------------------------------------
# Save -- every event here IS a ZTF discovery (that's what this sample is),
# no survey-guessing needed. LSST has no confirmed real TDE catalog yet, so
# lsst_mjd/lsst_names stay empty -- NOT fabricated -- until a real LSST TDE
# catalog exists to pull from.
# ------------------------------------------------------------------
out_path = f"{ASTROPHYSICAL_DATA_DIR}/real_tde_detections.npz"
np.savez(out_path,
         ztf_mjd=discovery_mjd, ztf_names=names,
         lsst_mjd=np.array([]), lsst_names=np.array([], dtype=object))

print(f"\nSaved {out_path}: {len(discovery_mjd)} ZTF epochs, "
      f"0 LSST (no real LSST TDE catalog available yet) epochs.")
