"""
Alternative galaxy major-merger-rate models for LastMajorMerger.

All rates are per-galaxy major-merger rates in Gyr^-1 (same units/meaning as
LastMajorMerger.lambda_MM in the repo). The repo code is NOT modified: each
model is a subclass overriding lambda_MM only.

Sources
-------
baseline : current repo model, 0.25 (1+z)^2
mundy17  : Mundy+2017 (arXiv:1705.07986), p.17 fractional merger rate R(z)
           n>1e-4 Mpc^-3 : R = 1.8e-2 (1+z)^1.3   (best constrained; default)
           n>5e-4 Mpc^-3 : R = 2.0e-2 (1+z)^1.4
duan24   : Duan+2024 (arXiv:2407.09472v3) Table 4 merger rate, JWST z=4.5-11.5 + Casteels+14 zero point,
           galaxies with 8 < log10(M*/Msun) < 10 (per galaxy, Gyr^-1):
           power law      R = 0.013 (1+z)^2.818
           power law+exp  R = 0.013 (1+z)^3.359 exp[-0.138 (1+z)]   (paper's preferred form; saturates at high z)
oleary21 : O'Leary+2021 EMERGE (arXiv:2001.02687), Eq.18 + Table 3 rows without
           R_proj: R = alpha (1+z)^m exp[beta (1+z)]  (major mergers mu<=4)
           log10(alpha), m, beta for mass cuts log10(M*/Msun) >= 9.7/10.3/11.0
"""
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "astrophysical_setup"))
from cosmology import LastMajorMerger, CosmologyModel  # noqa: E402

# (log10 alpha, m, beta) from O'Leary+2021 Table 3, intrinsic-rate rows
OLEARY_TABLE3 = {
    9.7:  (-1.785, 1.277, -0.100),
    10.3: (-1.530, 1.519, -0.160),
    11.0: (-1.189, 1.774, -0.133),
}


class _PowerLaw(LastMajorMerger):
    R0 = 0.25   # Gyr^-1
    m = 2.0
    label = "baseline"

    def lambda_MM(self, z):
        """Major-merger rate at redshift z."""
        return self.R0 * (1.0 + np.asarray(z)) ** self.m


class Baseline(_PowerLaw):
    """Original 0.25 (1+z)^2 merger rate."""
    label = r"Baseline $0.25(1+z)^{2}$"


class NoMerger(_PowerLaw):
    """Limit case: essentially no major mergers -> every galaxy censored at z_max. Sets the floor of the active fraction."""
    R0, m = 1e-6, 0.0
    label = "No mergers (limit)"


class Duan24_PL(_PowerLaw):
    """Duan et al. 2024 power-law merger rate."""
    R0, m = 0.013, 2.818
    label = r"Duan+24 $0.013(1+z)^{2.818}$"


class Duan24_PLexp(LastMajorMerger):
    """Duan+2024 Table 4 power-law + exponential merger rate (M* = 1e8-1e10)."""
    F0, M, TAU = 0.013, 3.359, -0.138
    label = r"Duan+24 PL+exp"

    def lambda_MM(self, z):
        """Major-merger rate at redshift z."""
        x = 1.0 + np.asarray(z)
        return self.F0 * x ** self.M * np.exp(self.TAU * x)


class Mundy17(_PowerLaw):
    """Mundy et al. 2017 merger rate."""
    R0, m = 1.8e-2, 1.3
    label = r"Mundy+17 $0.018(1+z)^{1.3}$"


class Mundy17_n5e4(_PowerLaw):
    """Mundy et al. 2017 merger rate with a fixed density of 5e4."""
    R0, m = 2.0e-2, 1.4
    label = r"Mundy+17 $0.020(1+z)^{1.4}$"


class OLeary21(LastMajorMerger):
    """EMERGE power-law-exponential; mass cut selects the Table 3 row."""

    def __init__(self, cosmology_model=None, logM_cut=9.7):
        super().__init__(cosmology_model)
        la, self.m, self.beta = OLEARY_TABLE3[logM_cut]
        self.alpha = 10.0 ** la
        self.label = rf"O'Leary+21 EMERGE $M_*\geq10^{{{logM_cut}}}$"

    def lambda_MM(self, z):
        """Major-merger rate at redshift z."""
        x = 1.0 + np.asarray(z)
        return self.alpha * x ** self.m * np.exp(self.beta * x)


class _MassDependent(LastMajorMerger):
    """
    Per-galaxy rate lambda(z | lgMgal). set_mass(lgMgal) must be called with one stellar mass
    per galaxy (same order as z_obs_array) before sampling; lambda_MM then broadcasts over
    the (N, n_grid) hazard grid. rate(z, lgM) is the stateless form used for plotting.
    """
    label = "mass-dependent"
    _lgM = None

    def set_mass(self, lgMgal):
        """Set the galaxy stellar mass."""
        self._lgM = np.asarray(lgMgal, float)

    def rate(self, z, lgM):
        """Merger rate at redshift z and mass lgM."""
        raise NotImplementedError

    def lambda_MM(self, z):
        """Major-merger rate at redshift z."""
        z = np.asarray(z)
        if self._lgM is None:
            raise RuntimeError("call set_mass(lgMgal) first")
        if z.ndim == 2 and z.shape[0] == self._lgM.size:
            return self.rate(z, self._lgM[:, None])
        return self.rate(z, self._lgM)


class BaselineDocstringM(_MassDependent):
    """Repo docstring example: lambda0=0.25, m(M)=2+0.3*log10(M/1e10)  (NOT what the code runs)."""
    label = r"Baseline + docstring $m(M)$"

    def rate(self, z, lgM):
        """Merger rate at redshift z and mass lgM."""
        return 0.25 * (1.0 + np.asarray(z)) ** (2.0 + 0.3 * (lgM - 10.0))


class OLeary21Mass(_MassDependent):
    """
    EMERGE Table-3 (log10 alpha, m, beta) linearly interpolated in log10(M*) between the
    >=9.7 / 10.3 / 11.0 rows, clamped outside. CAVEAT: rows are cumulative (M*>=cut) rates, i.e. an
    average over all galaxies above the cut, not the rate AT that mass.
    """
    label = "O'Leary+21 mass-interpolated"
    _K = np.array(sorted(OLEARY_TABLE3))

    def rate(self, z, lgM):
        """Merger rate at redshift z and mass lgM."""
        tab = np.array([OLEARY_TABLE3[k] for k in self._K])
        la, m, b = (np.interp(lgM, self._K, tab[:, i]) for i in range(3))
        x = 1.0 + np.asarray(z)
        return 10.0 ** la * x ** m * np.exp(b * x)


def make_models(cosmo=None):
    """Instantiate all merger models."""
    cosmo = cosmo or CosmologyModel()
    return {
        "repo":          LastMajorMerger(cosmo),   # untouched repo class, for equality checks
        "none":          NoMerger(cosmo),
        "baseline_docM": BaselineDocstringM(cosmo),
        "oleary_mass":   OLeary21Mass(cosmo),
        "baseline":      Baseline(cosmo),
        "duan24_pl":     Duan24_PL(cosmo),
        "duan24_plexp":  Duan24_PLexp(cosmo),
        "mundy17":       Mundy17(cosmo),
        "mundy17_n5e4":  Mundy17_n5e4(cosmo),
        "oleary_9.7":    OLeary21(cosmo, 9.7),
        "oleary_10.3":   OLeary21(cosmo, 10.3),
        "oleary_11.0":   OLeary21(cosmo, 11.0),
    }
