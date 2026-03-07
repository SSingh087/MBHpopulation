from galaxy import *

Mstar = np.logspace(6, 12, 500)     # Msun, linear
lgMgal = np.log10(Mstar)
lgMBH  = []

for i in range(len(lgMgal)):
    gal = Galaxy.check_nucleation(lgMgal[i])  # returns None if no nucleation
    if gal is None:
        lgMBH.append(np.nan)
    else:
        lgMBH.append(gal.lgMBH_mass())

Plotting.plot_lgMgal_vs_lgMBH(lgMgal, lgMBH)
