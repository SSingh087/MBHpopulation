from utils import *
from cosmology import CosmologyModel

_f_NSC_Hannah_   = lambda lgMgal: np.exp(-((lgMgal - 7.6)**4) / 400.0)  # arXiv:2407.10911 Fig 4
_f_NSC_Neumayer_ = lambda lgMgal: np.exp(-((lgMgal - 9.0)**4) /  50.0)  # arXiv:2001.03626 Fig 3

class Galaxy(CosmologyModel):
    """
    Galaxy with simple size-mass, virial sigma, and M-sigma MBH.
    """

    def __init__(self, lgMgal, z_gal, nucleation_occurs=True):
        """
        Parameters
        ----------
        lgMgal : float
            log10(Mstar/Msun)
        nucleation_occurs : bool
            Whether nucleation occurs for this galaxy (flag decided externally).
        """
        self.lgMgal = float(lgMgal)
        self.z_gal = float(z_gal)
        self.nucleation_occurs = bool(nucleation_occurs)

    @classmethod
    def check_nucleation(cls, lgMgal, z_gal):
        """
        returns a Galaxy instance if nucleation occurs, else None.
        """
        f = np.random.random()  # scalar in [0,1)

        if f < _f_NSC_Hannah_(lgMgal):
            # print(f"nucleation will occur for {lgMgal} (Hannah)")
            return cls(lgMgal, z_gal, nucleation_occurs=True)
        elif f < _f_NSC_Neumayer_(lgMgal):
            # print(f"nucleation will occur for {lgMgal} (Neumayer)")
            return cls(lgMgal, z_gal, nucleation_occurs=True)
        else:
            # print(f"NO nucleation for {lgMgal}")
            return None

    def R_eff(self, lg_A=0.82, B=0.24, Re_scatter=0.20, unit='kpc'):
        """
        Half-light (effective) radius from size-mass relation:
        log10(Re/kpc) = B * (lgMgal - 10.7) + lg_A + N(0, Re_scatter)
        """
        lgreff = B * (self.lgMgal - 10.7) + lg_A
        lgreff += np.random.normal(0.0, Re_scatter)
        Re_kpc = 10.0**lgreff

        if unit == 'kpc':
            return Re_kpc
        elif unit == 'cm':
            return Re_kpc * kpc_to_cm
        else:
            raise ValueError("Invalid unit. Must be 'kpc' or 'cm'")

    def sigma(self, kvir=1.0, unit='km/s'):
        """
        Virial sigma ~ sqrt( G * M_enclosed / (kvir * Re) ).
        Here use M_enclosed = 0.5 * Mstar (in grams), Re in cm.
        """
        Re_cm = self.R_eff(unit='cm')
        Mstar_linear = 10.0**self.lgMgal
        M_enclosed_grams = 0.5 * Mstar_linear * Msun_to_grams

        sig_cgs = np.sqrt(G_cgs * M_enclosed_grams / (kvir * Re_cm))  # cm/s

        if unit == 'km/s':
            return sig_cgs / 1e5
        if unit == 'cm/s':
            return sig_cgs
        elif unit == 'pc/year':
            return sig_cgs * sec_per_year / pc_to_cm  # pc/year from m/s
        else:
            raise ValueError("Invalid unit. Must be 'km/s' or 'pc/year'")

    def lgMBH_mass(self, A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53, unit='Msun'):
        """
        log10(MBH/Msun) from M-sigma; add Gaussian scatter.
        Defaults match your Greene+20-like parameters.
        """
        sigma = self.sigma(unit='km/s')
        lgMBH = A + B * np.log10(sigma / sigma_0)
        lgMBH += np.random.normal(0.0, MBH_scatter)
        return lgMBH



class NSC(Galaxy):
    """
    Nuclear Star Cluster / BH scales built on top of Galaxy.
    """

    def __init__(self, lgMgal, z_gal, nucleation_occurs=True, lgMBH=None, MBH_params=None):

        """
        Parameters
        ----------
        lgMgal : float
            log10(Mstar/Msun) for host.
        nucleation_occurs : bool
            For completeness; typically True if created via check_nucleation.
        lgMBH : float or None
            If provided, fixes log10(MBH/Msun). Else sampled via M-sigma.
        MBH_params : dict or None
            Keys: A, B, sigma_0, MBH_scatter
        """
        super().__init__(lgMgal, z_gal, nucleation_occurs=nucleation_occurs)

        if MBH_params is None:
            MBH_params = dict(A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53)

        if lgMBH is None:
            self._lgMBH = super().lgMBH_mass(
                A=MBH_params.get('A', 7.87),
                B=MBH_params.get('B', 4.55),
                sigma_0=MBH_params.get('sigma_0', 160.0),
                MBH_scatter=MBH_params.get('MBH_scatter', 0.53),
            )
        else:
            self._lgMBH = float(lgMBH)

    @property
    def lgMBH(self):
        """log10(MBH/Msun) fixed for this NSC instance."""
        return self._lgMBH

    def influence_radius(self, kvir=1.0, unit='pc'):
        """
        r_inf = G * M_bh / sigma^2
        """
        MBH_mass = 10 ** self._lgMBH
        rh_pc = G_pc3_per_Msun_yr2 * MBH_mass / (self.sigma(unit='pc/year')**2)
        if unit == 'pc':
            return rh_pc
        elif unit == 'cm':
            return rh_pc * pc_to_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")

    def scale_radius(self, kvir=1.0, factor=4.0, unit='pc'):
        """Dehnen scale radius r_a = factor * r_h """
        return factor * self.influence_radius(kvir=kvir, unit=unit)

    def capture_radius(self, unit='pc'):
        """
        Newtonian direct-capture proxy for compact objects:
        r_sBH = 8 * G * M_bh / c^2
        """
        r_sBH_pc = 8.0 * G_pc3_per_Msun_yr2 * (10**self._lgMBH) / (c_pc_per_year**2)
        if unit == 'pc':
            return r_sBH_pc
        elif unit == 'cm':
            return r_sBH_pc * pc_to_cm  # Convert back to cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")

    def tidal_radius_star(self, m_star=1.0*Msun, R_star=1.0*Rsun, unit='pc'):
        """
        Stellar tidal disruption radius:
        r_t = R_star * (M_bh / m_star)^(1/3)
        """
        # THIS CALCULATION IS STILL WRONG SINCE THE R_STAR AND M_STAR IS INCORRECT
        # THIS SHOULD BE DISCUSSED IN THE NEXT MEETING
        rt_pc = R_star * ((10**self._lgMBH) / m_star)**(1.0/3.0)
        if unit == 'pc':
            return rt_pc
        elif unit == 'cm':
            return rt_pc * pc_to_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")


    def dehnen_number_density(self, r, Ntot, gamma=1.5, kind='TDE', unit='pc'):
        """
        Dehnen 3D number-density profile n_i(r) for species i (stars or sBHs)
        n_i(r) = (3-$\gamma$)/(4π) * Ntot * r_a / [ r^$\gamma$ (r + r_a)^(4-$\gamma$) ] 

        Parameters
        ----------
        r         : array-like, radii [pc] at which to evaluate
        Ntot      : float, total number of objects (dimensionless)
        gamma     : float, inner slope
        kind      : str, 'TDE' or 'EMRI' (for potential future use in different truncations)

        Returns
        -------
        n_i : ndarray, number density [1/pc^3] same shape as r
        """
        r_a = self.scale_radius(kvir=1.0, factor=4.0, unit='pc')

        coef = (3.0 - gamma) * float(Ntot) / (4.0 * np.pi)

        r_k = self.capture_radius(unit='pc') if kind.upper() == 'EMRI' else self.tidal_radius_star(unit='pc')

        n = coef * (r_a / (np.power(r, gamma) * np.power(r + r_a, 4.0 - gamma))) * np.heaviside(r - r_k, r_k)  # 1/pc^3
        n = np.where(np.isfinite(n), n, 0.0)

        return n  # 1/pc^3

    def radial_number_distribution(self, r, Ntot, gamma=1.5, kind='TDE'):
        """
        Shell number distribution:
            n_r(r) = 4π r^2 n_i(r)    [units: 1/pc]
        """
        return 4.0 * np.pi * r**2 * self.dehnen_number_density(r, Ntot=Ntot, gamma=gamma, kind=kind)  # 1/pc

    def cumulative_number(self, r, Ntot, gamma=1.5, kind='TDE'):
        """
        Cumulative number:
            N(<r) = ∫_0^r 4π r'^2 n_i(r') dr'
        """
        nr = self.radial_number_distribution(r, Ntot=Ntot, gamma=gamma, kind=kind)  # 1/pc

        sort = np.argsort(r)
        r_s = r[sort]
        nr_s = nr[sort]

        # trapezoid cumulative
        if r_s.size > 1:
            partial = np.concatenate(([0.0], np.cumsum(0.5 * (nr_s[1:] + nr_s[:-1]) * np.diff(r_s))))
        else:
            partial = np.zeros_like(r_s)

        # unsort
        inv = np.argsort(sort)
        return partial[inv]

    def mass_density(self, r, Ntot, component_masses, gamma=1.5, kind='TDE', renormalize=False, unit='Msun/pc^3'):
        """
        Total 3D mass density: rho(r) = sum_i m_i * n_i(r)

        Parameters
        ----------
        r       : array-like radii (pc)
        gamma   :
        Ntot    :
        component_masses : in units of solar mass
        renormalize: if True, rescale n_i so that ∫ 4π r^2 n_i dr (over the provided grid) = Ntot
        """
        r = np.asarray(r, dtype=float)
        m_i = np.asarray(component_masses, dtype=float) # in units of solar mass
        rho_Msun_pc3 = np.zeros_like(r, dtype=float)  # accumulate in Msun/pc^3

        n_i = self.dehnen_number_density(r, Ntot=Ntot, gamma=gamma, kind=kind)  # 1/pc^3

        for i in range(len(m_i)):
            m_Msun = m_i[i]


            if renormalize:
                nr = 4.0 * np.pi * r**2 * n_i  # 1/pc
                sort = np.argsort(r)
                r_s, nr_s = r[sort], nr[sort]
                N_calc = np.trapz(nr_s, r_s) if r_s.size > 1 else 0.0
                if np.isfinite(N_calc) and N_calc > 0.0:
                    n_i = n_i * (Ntot / N_calc)
            # Add component mass density (Msun/pc^3)
            rho_Msun_pc3 += m_Msun * n_i

        if unit == 'Msun/pc^3':
            return rho_Msun_pc3
        elif unit == 'g/pc^3':
            return rho_Msun_pc3 * Msun_to_grams
        else:
            raise ValueError("unit must be 'Msun/pc^3' or 'g/pc^3'.")

    def rho_at_rinfl(self, Ntot, component_masses, gamma=1.5, kvir=1.0, kind='TDE', unit='Msun/pc^3', renormalize=False):
        """
        evaluate rho(r) at r = r_infl (influence radius).
        Returns scalar density.
        """
        r_inf_pc = float(self.influence_radius(kvir=kvir, unit='pc'))
        rho_arr = self.mass_density([r_inf_pc], Ntot=Ntot, component_masses=component_masses, gamma=gamma, kind=kind, unit=unit, renormalize=renormalize)
        return float(rho_arr[0])

    def t_relax(self, rho_r, Ntot, component_masses, gamma=1.5, kvir=1.0, kind='TDE', mbar=None, lnLambda=15.0, unit='yr'):
        """
        Two-body (non-resonant) relaxation time at r_infl.:
            t_rlx ≈ 0.34 * $\sigma$^3 / (G^2 * m_bar * $\rho(r_{infl})$ * lnΛ)

        Parameters
        ----------
        kvir       : float, virial coefficient in $\sigma$ definition
        mbar     : float [g], mean mass per scatterer; default = 1 Msun
        lnLambda   : float, Coulomb logarithm (10-15 typical)
        unit   :     'yr' or 'Gyr'

        Returns
        -------
        t_rlx : float, relaxation time in requested unit
        """

        t_yr = 0.34 * (self.sigma(unit='pc/year')**3) / (G_pc3_per_Msun_yr2**2 * mbar * rho_r * lnLambda)

        if unit == 'yr':
            return t_yr
        elif unit == 'Gyr':
            return t_yr / 1.0e9
        else:
            raise ValueError("unit must be one of: 'yr', 'Gyr'")

    def t_relax_at_rinfl(self, Ntot, component_masses, gamma=1.5, kvir=1.0, kind='TDE', mbar=None, lnLambda=15.0, unit='yr'):

        rho_at_rinfl = self.rho_at_rinfl(Ntot=Ntot, component_masses=component_masses, gamma=gamma, kvir=kvir, kind=kind, unit='Msun/pc^3')

        return self.t_relax(rho_r=rho_at_rinfl, Ntot=Ntot, component_masses=component_masses, gamma=gamma, kvir=kvir, kind=kind, mbar=mbar, lnLambda=lnLambda, unit=unit)

    def cusp_turn_on_time(self):
        kappa = 0.25  # fraction of t_relax for cusp regrowth; can be tuned
        t_LMM = self.cosmo.sample_lmm_times_Gyr(self.z_gal, m=2.0, size=1)[0]
        return t_LMM + kappa * self.t_relax_at_rinfl(Ntot=1e7, component_masses=np.random.uniform(1., 100, 100000), gamma=1.5, kvir=1.0, kind='EMRI', mbar=10, lnLambda=15.0, unit='Gyr')

    def cusp_age(self):
        return max(0, self.cosmo.age_Gyr(self.z_gal) - self.cusp_turn_on_time())