#!/bin/sh

EVENTS=1000000

OBSERVING_WINDOW_TDE_ZTF=500 # this is in days
OBSERVING_WINDOW_TDE_LSST=700 # this is in days

BANDS_ZTF="ztfg ztfr ztfi" # default for ZTF
BANDS_LSST="lsstu lsstg lsstr lssti lsstz lssty"

Z_GAL_MIN_ZTF=0.0
Z_GAL_MAX_ZTF=0.5

Z_GAL_MIN_LSST=0.0
Z_GAL_MAX_LSST=4.0


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

# ApJ, 890, 73 | IOPscience: 10.3847/1538-4357/ab6989

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

# Graham et al. (2019) : ZTF instrument/survey description; limiting magnitude ~20.5 mag; 47 deg² FoV; g+r every 3 days
# Ivezić et al. / LSST Science Collaboration : LSST Science Requirements Document; single-visit depth r ~ 24.5; coadded r ~ 27.5; 20,000 deg² in ugrizy
# Rubin Observatory DP1 Documentation (2025) : dp1.lsst.io; commissioning data; survey start confirmed 2025/2026