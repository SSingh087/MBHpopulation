#!/bin/sh

GALAXIES=100
OBSERVING_WINDOW_EMRI=10 # this is in years

OBSERVING_WINDOW_TDE_ZTF=400 # this is in days
OBSERVING_WINDOW_TDE_LSST=1000 # this is in days

BANDS_ZTF="ztfg ztfr ztfi" # default for ZTF
BANDS_LSST="lsstu lsstg lsstr lssti lsstz lssty"



# PRIORS for the population B hyper-parameters 

B_PRIOR_LAMBDA_M_LOW=-3
B_PRIOR_LAMBDA_M_HIGH=-1.01

# mass dist of the CO
B_PRIOR_LAMBDA_MU_LOW=-4
B_PRIOR_LAMBDA_MU_HIGH=-1.5

# spin 
B_PRIOR_MU_A_LOW=0.1
B_PRIOR_MU_A_HIGH=0.7

B_PRIOR_SIGMA_A_LOW=0.001
B_PRIOR_SIGMA_A_HIGH=0.05