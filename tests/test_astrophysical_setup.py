"""Fast checks of the astrophysical setup (numpy, scipy and astropy only)."""
import numpy as np
import pytest

import config
import cosmology
import galaxy
from galaxy import Galaxy, log10_Re_evolution
from rate import UniversalRate


# ---------- size evolution Re(z) ----------

def test_re_evolution_is_zero_today():
    assert log10_Re_evolution(0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("z", [0.5, 1.0, 3.0])
def test_re_evolution_1pz(z):
    assert log10_Re_evolution(z, alpha_Re=0.75, mode="1pz") == pytest.approx(-0.75 * np.log10(1 + z))


def test_re_evolution_hubble_mode():
    z = 2.0
    expected = -0.75 * 0.5 * np.log10(0.3 * (1 + z) ** 3 + 0.7)
    assert log10_Re_evolution(z, alpha_Re=0.75, mode="H") == pytest.approx(expected)


def test_re_evolution_cap_freezes_high_z():
    assert log10_Re_evolution(5.0, alpha_Re=0.75, mode="1pz", zcap=3.0) == pytest.approx(
        log10_Re_evolution(3.0, alpha_Re=0.75, mode="1pz"))


def test_re_evolution_off_when_alpha_zero():
    assert np.all(log10_Re_evolution(np.linspace(0, 8, 20), alpha_Re=0.0) == 0.0)


def test_re_evolution_bad_mode():
    with pytest.raises(ValueError):
        log10_Re_evolution(1.0, mode="bad")


def test_config_defaults():
    assert config.ALPHA_RE == 0.75
    assert config.RE_EVOL_MODE == "1pz"


# ---------- MBH mass from galaxy mass ----------

def test_lgmbh_from_mgal_default_is_z0():
    lgM = np.linspace(8, 12, 9)
    a = Galaxy.lgMBH_from_Mgal(lgM, A=config.MBH_A_sigma, B=config.MBH_B_sigma)
    b = Galaxy.lgMBH_from_Mgal(lgM, A=config.MBH_A_sigma, B=config.MBH_B_sigma, z=0.0)
    np.testing.assert_allclose(a, b)


def test_compact_galaxies_at_high_z_host_heavier_mbhs():
    # Smaller Re at fixed M* -> larger sigma -> larger MBH through M-sigma.
    lgM = np.array([10.0])
    lo = Galaxy.lgMBH_from_Mgal(lgM, A=config.MBH_A_sigma, B=config.MBH_B_sigma, z=0.0)
    hi = Galaxy.lgMBH_from_Mgal(lgM, A=config.MBH_A_sigma, B=config.MBH_B_sigma, z=2.0)
    # sigma ∝ Re^-1/2, so the shift is B * 0.5 * alpha * log10(1+z)
    assert hi - lo == pytest.approx(config.MBH_B_sigma * 0.5 * 0.75 * np.log10(3.0))


def test_galaxy_arrays_stay_aligned_after_nucleation_cut():
    rng = np.random.default_rng(1)
    n = 200
    z = rng.uniform(0.01, 3.0, n)
    lgM = rng.uniform(8.0, 11.5, n)
    nucleated = rng.random(n) < 0.6
    gal = Galaxy(z_gal=z, lgMgal=lgM, nucleation_occurs=nucleated, rng=rng)
    k = nucleated.sum()
    assert gal.lgMgal.shape == gal.z_gal.shape == gal.lgMBH_mass.shape == gal.sigma_km_s.shape == (k,)
    np.testing.assert_allclose(gal.z_gal, z[nucleated])
    assert np.all(np.isfinite(gal.lgMBH_mass))


# ---------- mass functions ----------

def test_gsmf_interpolator_reproduces_table():
    gsmf = cosmology.GalaxyStellarMassFunction()
    vals = gsmf.interp(np.column_stack([np.zeros(gsmf.lgMgal_data.size), gsmf.lgMgal_data]))
    np.testing.assert_allclose(vals, gsmf.gsmf_data[0.0])


@pytest.fixture
def mbhmf():
    return cosmology.MBHMassFunction(cosmology.GalaxyStellarMassFunction())


@pytest.mark.parametrize("z", [0.0, 1.0])
def test_mbhmf_conserves_number_density(mbhmf, z):
    # Change of variables must keep the integral of phi over log mass.
    lgMgal, lphi_gal = mbhmf.gsmf.get_gsmf(np.array([z]), n_points_mass=2000)
    lgMBH, lphi_bh = mbhmf.get_mbhmf(np.array([z]), n_points_mass=2000)
    n_gal = np.trapezoid(10 ** lphi_gal[0], lgMgal)
    n_bh = np.trapezoid(10 ** lphi_bh[0], lgMBH)
    assert n_bh == pytest.approx(n_gal, rel=0.02)


def test_mbhmf_without_evolution_matches_old_path(mbhmf, monkeypatch):
    # With ALPHA_RE = 0 the redshift-dependent branch must reduce to the z-independent one.
    monkeypatch.setattr(galaxy, "ALPHA_RE", 0.0)
    z = np.array([0.5, 2.0])
    lgMgal, lphi_gal = mbhmf.gsmf.get_gsmf(z, n_points_mass=500)
    grid_old, old = mbhmf._compute_phi_MBH(lgMgal, lphi_gal)
    grid_new, new = mbhmf._compute_phi_MBH(lgMgal, lphi_gal, z_vals=z)
    np.testing.assert_allclose(grid_new, grid_old)
    finite = old > -100
    np.testing.assert_allclose(new[finite], old[finite], atol=1e-6)


# ---------- cosmology ----------

@pytest.fixture(scope="module")
def cosmo():
    return cosmology.CosmologyModel()


def test_age_today(cosmo):
    assert cosmo.age_Gyr(0.0) == pytest.approx(13.47, abs=0.05)


def test_comoving_distance_inversion(cosmo):
    z = np.array([0.1, 0.5, 1.0, 3.0])
    np.testing.assert_allclose(cosmo.z_from_comoving_distance_fast(cosmo.comoving_distance(z)), z, rtol=1e-4)


# ---------- event-rate shapes ----------

def test_emri_rate_peaks_at_one():
    tau = np.linspace(0.01, 20, 4001)
    r = UniversalRate.EMRI_rate(tau)
    assert tau[np.argmax(r)] == pytest.approx(1.0, abs=0.01)
    assert r.max() == pytest.approx(1.0)
    assert UniversalRate.EMRI_rate(0.0) == 0.0


def test_tde_rate_decreases():
    r = UniversalRate.TDE_rate(np.linspace(0, 20, 500))
    assert np.all(np.diff(r) < 0)
