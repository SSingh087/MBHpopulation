#!/bin/sh

EVENTS=1000000

# The redback TDE models simulated in parallel DAG branches, each trained as
# its own independent classifier (space-separated for shell iteration).
# bpl_cooling_envelope and gaussianrise_cooling_envelope are excluded: both
# crash deterministically (KeyError: 'bands') under redback's
# SimulateOpticalTransient, which always calls string-model TDE functions
# with output_format='sncosmo_source' -- neither model's source code has a
# branch for that value (confirmed present in both redback 1.15.3 and
# 1.18.0, i.e. still an open upstream bug, not a version-lag issue).
MODELS="cooling_envelope stream_stream_tde tde_analytical tde_fallback"

# Total injection budget split evenly across the 4 active models. Integer
# division means 4 * EVENTS_PER_MODEL may be slightly less than EVENTS -- this
# is expected/fine (e.g. 1000000/4 = 250000, so 4*250000 = 1000000 total).
EVENTS_PER_MODEL=$((EVENTS / 4))

LENGTH_OF_TDE_ZTF=500  # this is in days -- typical TDE lightcurve: rise ~30d, decline
                        # 200-400d (Andreoni et al. 2022, ApJ 938, L17); this window
                        # deliberately does NOT capture the rare >1000-day outlier tail
                        # (e.g. Subrayan et al. 2023 "Scary Barbie" AT2021lwx) -- note
                        # this as a stated limitation in the paper if using this window.
LENGTH_OF_TDE_LSST=700  # this is in days -- same caveat as above.

BANDS_ZTF="ztfg ztfr ztfi" # default for ZTF
BANDS_LSST="lsstu lsstg lsstr lssti lsstz lssty"

Z_GAL_MIN_ZTF=0.0
Z_GAL_MAX_ZTF=0.5
# ZTF: real ZTF TDEs observed to date top out at z=0.21 (van Velzen et al. 2021,
# ApJ 908, 4); 0.5 is a reasonable conservative buffer above the observed max.

Z_GAL_MIN_LSST=0.0
Z_GAL_MAX_LSST=4.0
# LSST: bulk of the LSST-detectable TDE population is expected at z <~ 1.5, but this
# ceiling is intentionally kept wide to cover the physically-motivated high-z discovery
# tail LSST is specifically expected to probe -- jetted TDEs already observed to z=1.2
# (Andreoni et al. 2022), lensed TDEs with median z=1.5-2 (Mamuzic et al. 2025/2026,
# arXiv:2502.19495), and forecasts explicitly discussing LSST-era discovery of TDEs
# beyond z~1 and potentially z>3 (TDE rate forecasts across cosmic time for LSST/Roman/
# JWST, ApJ, https://iopscience.iop.org/article/10.3847/1538-4357/ae7a49). The
# detectability classifier needs to see this full tail (correctly labeled as mostly
# non-detectable) to learn an accurate decision boundary -- report the *detected*
# population separately, which should concentrate at z <~ 1.5 as expected from the
# literature above.


# ────────────────────────────────────────────────────────────────────────────
# Survey observation windows (MJD)
# ────────────────────────────────────────────────────────────────────────────

# ZTF: March 17, 2018 (MJD 58200) -> Dec 22, 2026 (MJD 61396), 8.75 yr
#   (Bellm et al. 2019, PASP 131, 018002; Graham et al. 2019, PASP 131, 078001)

# LSST/Rubin: June 29, 2026 (MJD 61220) -> June 28, 2036 (MJD 64872), 10.00 yr
#   CORRECTED from the earlier 2025-06-01 estimate. LSST officially began its
#   10-year Legacy Survey of Space and Time on June 29, 2026, following an
#   extended commissioning/early-operations period (first light imaging April
#   2025; "First Look" event June 23, 2025; operations handover Oct 25, 2025).
#   Sources:
#     - Rubin Observatory survey-strategy progress page (authoritative, live):
#       https://survey-strategy.lsst.io/progress/index.html
#     - Official announcement (June 30, 2026):
#       https://rubinobservatory.org/news/action-rubin-lsst-begins
#     - UW News (June 30, 2026):
#       https://www.washington.edu/news/2026/06/30/rubin-observatory-legacy-survey-space-time-lsst/


# van Velzen et al. (2021) : "Seventeen Tidal Disruption Events from the First Half of ZTF Survey Observations: Entering a New Era of Population Studies"

# ApJ, 908, 4 | arXiv: 2001.01409

# → ZTF highest-z TDE at z=0.21; rise ~30d, decline 200–400d; spectroscopic subclasses
# Hammerstein et al. (2023) : "The Final Season Reimagined: 30 Tidal Disruption Events from the ZTF-I Survey"

# ApJ, 942, 9 | IOPscience: 10.3847/1538-4357/aca283

# → ZTF-I full sample; forced photometry up to ~800d post-peak; z_max per event
# Yao et al. (2023) : "Tidal Disruption Event Demographics with the Zwicky Transient Facility: Volumetric Rates, Luminosity Function, and Implications for the Local Black Hole Mass Function"

# arXiv: 2303.06523

# → ZTF-I/II rates; luminosity function; z_max per source at r=19.5–20.5 mag
# van Velzen et al. (2019) : "The First Tidal Disruption Flare in ZTF: From Photometric Selection to Multi-wavelength Characterization"

# ApJ, 872, 198 | arXiv: 1809.02608

# → AT2018zr; ZTF survey depth ~20.5 mag; TDE selection from multi-band optical surveys
# van Velzen et al. (2022) : "AT2020wey and the class of faint and fast Tidal Disruption Events"

# arXiv: 2209.12913

# → V_max method; ZTF survey area 23,675 deg²; luminosity function per magnitude bin


# LSST/Rubin TDE Forecasts & Redshift Reach

# Bricman & Gomboc (2020) : "The Prospects of Observing Tidal Disruption Events with the Large Synoptic Survey Telescope"

# ApJ, 890, 73 | IOPscience: 10.3847/1538-4357/ab6989 | ADS: 2020ApJ...890...73B

# → Key paper: LSST simulations done for z < 3.0; events at larger z too dim; MOSFiT light curves for M_BH = 10^5–10^7 M☉
# van Velzen et al. (2011) : "Optical Discovery of Probable Stellar Tidal Disruption Flares"

# (also cited as the source of ~1000 TDEs/yr LSST forecast in the Rubin Cadence Note)
# Rubin Cadence Note: Tidal Disruption Events : Bricman & Gomboc

# LSST DocuShare Document-37622

# → ~1000 TDEs/yr from LSST; detection criteria pre/post-peak; u-band importance
# Auchettl, Ramirez-Ruiz & Guillochon (2018) : "Identifying Tidal Disruption Events via Prior Photometric Selection of Their Preferred Hosts"

# arXiv: 1810.09507

# → Key paper: Typical TDE at M_r = −19 detectable to z ~ 0.8 with LSST (m_r = 24.5); median detected TDE at z ~ 0.6
# Ryu et al. / Prospects for Measuring BH Masses (2024) : "Prospects for Measuring Black Hole Masses using TDEs with the Vera C. Rubin Observatory"

# arXiv: 2512.13409

# → Best-studied Rubin sample: 0 < z < 0.3; spectroscopic follow-up feasible (TiDES/4MOST) for z < 0.3
# Pfister et al. (2020) : "Multiple Tidal Disruptions as an Indicator of Binary Super-Massive Black Hole Systems"

# arXiv: 1011.5874

# → LSST z_max ~ 0.27–0.43 for known TDE luminosities with 2-mag buffer; practical limit z ~ 0.35
# Steinhardt et al. (2022) : "Searching for Ultra-light Bosons and Constraining Black Hole Spin Distributions with Stellar Tidal Disruption Events"

# arXiv: 2202.01215

# → Maximum redshift formula for magnitude-limited surveys applied to LSST; m_lim = 22.8 in g-band

# (NEW) TDE Rates across Cosmic Time: Forecasts for LSST, Roman, and JWST (2026)

# ApJ | IOPscience: 10.3847/1538-4357/ae7a49

# → Very few LSST TDEs expected beyond z~1.5; upcoming surveys (incl. LSST) expected to
#   probe first TDE populations beyond z~1 and potentially z>3, constraining the
#   low-mass end of the SMBH mass function -- directly relevant to this project's
#   EMRI-TDE joint population inference.
# Andreoni et al. (2022) : "The Jetted Tidal Disruption Event AT 2022cmc" -- jetted TDE
#   observed to z=1.2. ApJ 938, L17 | ADS: 2022ApJ...938L..17A

# Mamuzic et al. (2025/2026) : "Rates of Strongly Lensed Tidal Disruption Events"

# arXiv: 2502.19495

# → Median redshift of a lensed TDE expected between z=1.5 and 2 for LSST.


# TDE Light Curve Timescales

# Andreoni et al. / AT2022cmc (2024) : "The Jetted Tidal Disruption Event AT 2022cmc"

# ApJ | IOPscience: 10.3847/1538-4357/ae1838

# → Rise ~30d; fade 200–400d; ZTF sample comparison
# Gu et al. (2026) : "Similar ratios of rise timescale to decline timescale of optical light curves in common tidal disruption events"

# A&A | arXiv: 2603.27897

# → Systematic t_1/2,rise and t_1/2,decline measurements across ZTF sample
# Subrayan et al. (2023) : "Scary Barbie: AT2021lwx at z=0.995"

# arXiv: 2302.10932

# → Extreme TDE: light curve spanning >1000 observer-frame days; M_BH ~ 10^8 M☉
# IOPscience (2023) : "Discovery of a Years-delayed Radio Flare from an Unusually Slowly Evolving TDE Candidate"

# IOPscience: 10.3847/1538-4357/ae29f0

# → Slow TDE: t_1/2,rise ~ 61d, t_1/2,decline ~ 228d; outlier in ZTF sample


# Survey Instrument References

# Bellm et al. (2019) : ZTF System Overview, PASP 131, 018002 | ADS: 2019PASP..131a8002B
# Graham et al. (2019) : ZTF Science Objectives; limiting magnitude ~20.5 mag; 47 deg² FoV; g+r every 3 days | ADS: 2019PASP..131g8001G
# Dekany et al. (2020) : ZTF Observing System (camera, chip-gap/vignetting losses) | ADS: 2020PASP..132c8001D
# Ivezić et al. / LSST Science Collaboration : LSST Science Requirements Document; single-visit depth r ~ 24.5; coadded r ~ 27.5; 20,000 deg² in ugrizy
# Rubin Observatory Survey Strategy (2026) : survey-strategy.lsst.io; confirms LSST 10-yr survey start June 29, 2026