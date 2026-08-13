"""
Pull the real, published TDE sample from OTTER (Franz et al. 2025, arXiv:2509.05405)
and extract discovery-time arrays (MJD), split by discovery survey, for overlay on
the simulated cumulative-detection envelope plots.

RUN THIS anywhere with normal internet access -- otter.idies.jhu.edu
is not reachable from this sandbox, so this script is written but not live-tested
against the real API. It IS tested against the astro-otter package's local schema/
class structure (confirmed installable, confirmed Otter.get_meta() returns an
astropy Table). Please run the diagnostic print block first and paste the output
back if column names differ from what's assumed below.

    pip install astro-otter --break-system-packages
"""

import numpy as np
from astropy.time import Time
import otter

# ------------------------------------------------------------------
# Step 1: connect and pull metadata for all spectroscopically-classed TDEs
# ------------------------------------------------------------------
db = otter.Otter()  # default url=https://otter.idies.jhu.edu/api, guest login

meta = db.get_meta(classification="TDE", spec_classed=True)
print(f"Pulled metadata for {len(meta)} TDEs from OTTER")
print("Available columns:", meta.colnames)
print(meta[:5])  # inspect a few rows -- CHECK this before trusting the rest of the script

# ------------------------------------------------------------------
# Step 2: locate the discovery-date column
# ------------------------------------------------------------------
# OTTER's schema.DateSchema suggests dates are stored with a "date_type" (e.g.
# "discovery", "peak") and a "value" (+ "date_format", e.g. "MJD" or "UTC" ISO).
# get_meta() may flatten this into columns like "discovery_date" directly, or
# leave it as a nested list needing unpacking -- inspect meta.colnames above and
# adjust the extraction below to match what you actually see.

def extract_discovery_mjd(row):
    """
    Adjust this function once you've inspected `meta.colnames` / a sample row.
    Two likely shapes are handled below; delete whichever doesn't match.
    """
    # Shape A: a flat "discovery_date" column already in MJD or ISO string
    if "discovery_date" in row.colnames:
        val = row["discovery_date"]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return float(val)  # already MJD
        except (TypeError, ValueError):
            return Time(str(val)).mjd  # ISO string -> MJD

    # Shape B: nested "date" list of dicts, e.g. [{"date_type": "discovery",
    # "value": "2020-11-12", "date_format": "iso"}, ...]
    if "date" in row.colnames:
        for d in row["date"]:
            if d.get("date_type") == "discovery":
                val = d["value"]
                fmt = d.get("date_format", "iso").lower()
                if fmt == "mjd":
                    return float(val)
                return Time(str(val)).mjd

    return None


def guess_survey(row):
    """
    Best-effort discovery-survey tag. OTTER names TDEs discovered by ZTF with a
    ZTF-internal name (e.g. "ZTF19aapreis") alongside the public AT/IAU name, and
    typically stores this in a "name" or "alias" field. LSST-discovered TDEs are
    too recent (survey started mid-2025) to be numerous yet; check for "LSST" the
    same way. Adjust the field name below once you've inspected meta.colnames.
    """
    name_fields = []
    for key in ("name", "default_name", "alias", "names"):
        if key in row.colnames:
            v = row[key]
            name_fields.append(str(v))
    joined = " ".join(name_fields).upper()
    if "ZTF" in joined:
        return "ZTF"
    if "LSST" in joined:
        return "LSST"
    return "unknown"


discovery_mjd = []
survey_tag = []
names = []
for row in meta:
    mjd = extract_discovery_mjd(row)
    if mjd is None:
        continue
    discovery_mjd.append(mjd)
    survey_tag.append(guess_survey(row))
    names.append(row["name"] if "name" in row.colnames else "?")

discovery_mjd = np.array(discovery_mjd)
survey_tag = np.array(survey_tag)
names = np.array(names)

print(f"\nExtracted discovery MJD for {len(discovery_mjd)}/{len(meta)} TDEs")
for s in np.unique(survey_tag):
    print(f"  {s}: {np.sum(survey_tag == s)}")

# ------------------------------------------------------------------
# Step 3: save per-survey arrays for direct use in plot_cumulative_all_universes.py
# ------------------------------------------------------------------
ztf_mask = survey_tag == "ZTF"
lsst_mask = survey_tag == "LSST"

np.savez("real_tde_detections.npz",
         ztf_mjd=discovery_mjd[ztf_mask], ztf_names=names[ztf_mask],
         lsst_mjd=discovery_mjd[lsst_mask], lsst_names=names[lsst_mask])

print(f"\nSaved real_tde_detections.npz: "
      f"{ztf_mask.sum()} ZTF, {lsst_mask.sum()} LSST discovery dates.")
print("Feed these into plot_cumulative_all_universes.py as REAL_T_MJD['ZTF'] / ['LSST'].")