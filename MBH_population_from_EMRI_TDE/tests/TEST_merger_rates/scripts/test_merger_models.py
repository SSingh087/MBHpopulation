"""Sanity tests for merger_models. Run: python test_merger_models.py  (or pytest)."""
import numpy as np
from merger_models import make_models, OLEARY_TABLE3
from cosmology import LastMajorMerger, CosmologyModel

COSMO = CosmologyModel()
MODELS = make_models(COSMO)
ZG = np.linspace(0.0, 10.0, 400)
ZMODELS = {k: m for k, m in MODELS.items() if not hasattr(m, "set_mass")}   # z-only models
MMODELS = {k: m for k, m in MODELS.items() if hasattr(m, "set_mass")}     # need set_mass(lgMgal)


def test_baseline_matches_repo():
    ref = LastMajorMerger(COSMO)
    assert np.allclose(MODELS["baseline"].lambda_MM(ZG), ref.lambda_MM(ZG))


def test_rates_positive_finite_and_shape():
    z2d = np.random.default_rng(0).uniform(0, 10, (5, 7))
    for k, m in ZMODELS.items():
        r = m.lambda_MM(ZG)
        assert np.all(np.isfinite(r)) and np.all(r > 0), k
        assert m.lambda_MM(z2d).shape == z2d.shape, k   # hazard grid is 2D


def test_alternatives_below_baseline_at_low_z():
    for k in ("mundy17", "mundy17_n5e4", "oleary_9.7", "oleary_10.3"):
        assert MODELS[k].lambda_MM(0.0) < 0.25 * MODELS["baseline"].lambda_MM(0.0), k


def test_paper_values():
    assert np.isclose(MODELS["mundy17"].lambda_MM(0.0), 1.8e-2)
    la, m, b = OLEARY_TABLE3[10.3]
    assert np.isclose(MODELS["oleary_10.3"].lambda_MM(1.0),
                      10**la * 2**m * np.exp(2 * b))


def test_mass_dependent_models():
    lg = np.array([8.0, 9.0, 10.0, 11.0, 11.5])
    for k, m in MMODELS.items():
        m.set_mass(lg)
        z2d = np.tile(np.linspace(0, 10, 7), (lg.size, 1))
        r = m.lambda_MM(z2d)                       # (N, n) hazard-grid shape
        assert r.shape == z2d.shape and np.all(np.isfinite(r)) and np.all(r > 0), k
    # docstring baseline: rate slope steepens with mass, so at z=4 rate rises with M
    m = MMODELS["baseline_docM"]; m.set_mass(np.array([8.0, 10.0, 11.0]))
    r4 = m.lambda_MM(np.full((3, 2), 4.0))[:, 0]
    assert r4[0] < r4[1] < r4[2]
    assert np.isclose(r4[1], 0.25 * 5.0 ** 2.0)     # M=1e10 -> m=2 -> baseline
    # EMERGE interpolation reproduces the Table 3 rows exactly at the tabulated masses
    m = MMODELS["oleary_mass"]; m.set_mass(np.array([9.7, 10.3, 11.0]))
    got = m.lambda_MM(np.full((3, 2), 1.0))[:, 0]
    want = [MODELS[k].lambda_MM(1.0) for k in ("oleary_9.7", "oleary_10.3", "oleary_11.0")]
    assert np.allclose(got, want)
    try:
        MMODELS["oleary_mass"]._lgM = None; MMODELS["oleary_mass"].lambda_MM(1.0); assert False
    except RuntimeError:
        pass


def test_duan24_values():
    assert np.isclose(MODELS["duan24_pl"].lambda_MM(0.0), 0.013)
    assert np.isclose(MODELS["duan24_plexp"].lambda_MM(1.0), 0.013 * 2**3.359 * np.exp(-0.138 * 2))
    assert np.isclose(MODELS["duan24_plexp"].lambda_MM(0.0), 0.013 * np.exp(-0.138))
    # paper: rate saturates at high z -> PL+exp grows much slower than PL beyond z~6
    r = lambda k, z: MODELS[k].lambda_MM(z)
    assert r("duan24_plexp", 11.5) / r("duan24_plexp", 6.0) < r("duan24_pl", 11.5) / r("duan24_pl", 6.0)


def test_repo_cosmology_matches_study_models():
    """The CLI-selectable models in astrophysical_setup/cosmology.py must equal the study models used for the impact analysis."""
    from cosmology import MERGER_MODEL_CHOICES
    pairs = {"baseline": "baseline", "mundy17": "mundy17", "oleary21": "oleary_9.7", "duan24": "duan24_plexp"}
    assert set(pairs) == set(MERGER_MODEL_CHOICES)
    for repo_name, study_name in pairs.items():
        got = LastMajorMerger(COSMO, merger_model=repo_name).lambda_MM(ZG)
        assert np.allclose(got, MODELS[study_name].lambda_MM(ZG), rtol=1e-12), repo_name
    try:
        LastMajorMerger(COSMO, merger_model="nope"); assert False
    except ValueError:
        pass
    # default (no argument) is bit-identical to the original hard-coded 0.25 (1+z)^2
    assert np.array_equal(LastMajorMerger(COSMO).lambda_MM(ZG), 0.25 * (1 + ZG) ** 2)


def test_utils_choices_match_cosmology():
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from utils import MERGER_MODEL_CHOICES as U, DEFAULT_MERGER_MODELS
    from cosmology import MERGER_MODEL_CHOICES as C
    assert set(U) == set(C) and set(DEFAULT_MERGER_MODELS) <= set(U) and "baseline" not in DEFAULT_MERGER_MODELS


def test_survival_monotone_in_unit_interval():
    for k, m in ZMODELS.items():
        z, S = m.survival_function(np.linspace(0, 10, 500))
        assert S[0] == 1.0 and np.all(np.diff(S) <= 1e-12), k
        assert np.all((S >= 0) & (S <= 1)), k


def test_lower_rate_gives_less_recent_lmm():
    """Median z_LMM must be larger (earlier merger) for lower-rate models."""
    np.random.seed(1)
    zobs = np.full(4000, 0.5)
    med = {k: np.median(MODELS[k].sample_z_LMM(zobs)) for k in ("baseline", "mundy17", "oleary_9.7")}
    assert med["baseline"] < med["mundy17"] and med["baseline"] < med["oleary_9.7"], med


def test_sampled_zLMM_matches_analytic_cdf():
    np.random.seed(2)
    m = MODELS["mundy17"]
    z0, n = 0.3, 20000
    zs = m.sample_z_LMM(np.full(n, z0), size=1)
    zg, Lam = m.build_hazard_grid(np.array([z0]))
    p_merge = 1.0 - np.exp(-Lam[0, -1])   # P(any merger before z_max); rest is clamped to z_max
    for q in (0.25 * p_merge, 0.5 * p_merge, 0.75 * p_merge):
        zq = np.interp(-np.log1p(-q), Lam[0], zg[0])
        frac = np.mean(zs <= zq)
        assert abs(frac - q) < 0.03, (q, frac)


def test_censored_draws_clamp_to_zmax():
    """Draws with H* > Lambda(z_max) (no merger before z_max) are clamped to z_max."""
    np.random.seed(3)
    m = MODELS["oleary_9.7"]
    zs = m.sample_z_LMM(np.full(5000, 0.3))
    _, Lam = m.build_hazard_grid(np.array([0.3]))
    p_none = np.exp(-Lam[0, -1])
    assert abs(np.mean(zs >= m.z_max - 1e-9) - p_none) < 0.03


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("PASS", name)
