"""
Plot every public TDE model in redback.transient_models.tde_models,
in LSST photometric bands, on its own subplot.

Key design choice (this is the "smart solution" part):
------------------------------------------------------
TDE models in redback do NOT share a common parameter set
(cooling_envelope uses mbh_6/stellar_mass/eta/alpha/beta, tde_analytical uses
l0/t_0_turn/mej/vej/..., tde_fallback uses mbh6/mstar/tvisc/..., etc).
Hand-maintaining one fixed dict of parameters that you slice per-model is
fragile and breaks every time redback adds/renames a model.

Instead we exploit the fact that redback ships a matching bilby Prior for
every registered model name (`redback.priors.get_priors(model=name)`).
We draw ONE fixed sample per model (fixed seed -> reproducible "fixed
parameters", but self-consistent per model) and reuse it every run.
This makes the code robust to redback version changes with zero manual
parameter bookkeeping.
"""

import inspect
import os
import numpy as np
import matplotlib.pyplot as plt

import redback
import redback.transient_models.tde_models as tde_models
from redback.utils import bands_to_frequency

# redback's plot style forces text.usetex=True on import; disable it since a
# full LaTeX install usually isn't available in plotting environments.
plt.rcParams["text.usetex"] = False

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
SURVEYS = {
    "ZTF": {
        "bands": ["ztfg", "ztfr", "ztfi"],
        "colors": {"ztfg": "#00b200", "ztfr": "#e30000", "ztfi": "#8b0000"},
        "time_days": np.linspace(0.1, 500, 200),
    },
    "LSST": {
        "bands": ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"],
        "colors": {
            "lsstu": "#5e00ff", "lsstg": "#0057ff", "lsstr": "#00b200",
            "lssti": "#ff8c00", "lsstz": "#e30000", "lssty": "#8b0000",
        },
        "time_days": np.linspace(0.1, 700, 200),
    },
}
REDSHIFT = 0.05          # shared fixed redshift used for every model
OUTPUT_FORMAT = "magnitude"
OUTDIR = "plots_tde_models"

os.makedirs(OUTDIR, exist_ok=True) 

# ----------------------------------------------------------------------
# Discover public TDE model functions
# ----------------------------------------------------------------------
# 'fitted', 'fitted_exp_decay', 'fitted_pl_decay' import the external
# `FitTeD` package (github.com/paolamartinez/FitTeD or similar), which most
# users won't have compiled. Excluded by default; set INCLUDE_FITTED=True
# if you have it installed.
INCLUDE_FITTED = False
_FITTED_MODELS = {"fitted", "fitted_exp_decay", "fitted_pl_decay"}


def get_tde_model_names():
    """Names of the available redback TDE models."""
    names = [
        name for name, obj in inspect.getmembers(tde_models, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == tde_models.__name__
    ]
    if not INCLUDE_FITTED:
        names = [n for n in names if n not in _FITTED_MODELS]
    return sorted(names)


# ----------------------------------------------------------------------
# Build one fixed, self-consistent parameter draw per model
# ----------------------------------------------------------------------
def sample_fixed_parameters(model_name, seed=SEED):
    """Draw one fixed sample from the model's redback prior."""
    priors = redback.priors.get_priors(model=model_name)
    np.random.seed(seed)
    sample = priors.sample()
    # force a common redshift where the model actually has that parameter,
    # so every plot is at a comparable distance
    if "redshift" in sample:
        sample["redshift"] = REDSHIFT

    # --- known redback prior/signature name mismatches (as of v1.18.0) ---
    # fitted_exp_decay / fitted_pl_decay: prior key is 'a_mh' but the
    # function signature expects 'a_bh'. Patch here rather than upstream.
    if "a_mh" in sample and "a_bh" not in sample:
        sample["a_bh"] = sample.pop("a_mh")

    return sample


# ----------------------------------------------------------------------
# Evaluate + plot one model
# ----------------------------------------------------------------------
def _is_network_error(exc):
    msg = str(exc)
    return "HTTP Error" in msg or "URLError" in msg or isinstance(
        exc, (ConnectionError, TimeoutError)
    )


def plot_model(model_name, ax, bands_list, band_colors, time_days):
    """Plot one TDE model's light curves."""
    func = getattr(tde_models, model_name)
    params = sample_fixed_parameters(model_name)

    is_bolometric = "bolometric" in model_name or "redshift" not in params

    try:
        if is_bolometric:
            # bolometric models: no bands/output_format, just time (rest frame, days)
            lbol = func(time_days, **params)
            ax.plot(time_days, lbol, color="k")
            ax.set_yscale("log")
            ax.set_ylabel("Bolometric luminosity [erg/s]")
        else:
            try:
                # Preferred: true synthetic photometry through registered
                # sncosmo bandpasses. Requires sncosmo to fetch (once,
                # then caches) the throughput curves from its remote data
                # server -> needs outbound network access on first use.
                for band in bands_list:
                    bands_arr = np.array([band] * len(time_days))
                    mag = func(
                        time_days,
                        bands=bands_arr,
                        output_format=OUTPUT_FORMAT,
                        **params,
                    )
                    mag = np.asarray(mag, dtype=float)
                    ax.plot(time_days, mag, color=band_colors[band], label=band)
                ax.invert_yaxis()  # magnitudes: brighter = up
                ax.set_ylabel("Apparent magnitude")
                ax.legend(fontsize=6, ncol=3, loc="lower right")
            except Exception as band_err:
                if not _is_network_error(band_err):
                    raise
                # Offline fallback: use flux_density at each band's
                # effective central frequency (local lookup table, no
                # network needed) instead of full synthetic photometry.
                for band in bands_list:
                    freq = bands_to_frequency([band])[0]
                    frequency_arr = np.ones(len(time_days)) * freq
                    fd = func(
                        time_days,
                        frequency=frequency_arr,
                        output_format="flux_density",
                        **params,
                    )
                    fd = np.asarray(fd, dtype=float)
                    ax.plot(time_days, fd, color=band_colors[band], label=band)
                ax.set_yscale("log")
                ax.set_ylabel("Flux density [mJy] (offline fallback,\n"
                               "central-frequency approx, not true bandpass)")
                ax.legend(fontsize=6, ncol=3, loc="best")

        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Time [days]")
        return True, None

    except Exception as e:
        ax.text(0.5, 0.5, f"FAILED:\n{type(e).__name__}: {e}",
                 ha="center", va="center", wrap=True, fontsize=7, color="red",
                 transform=ax.transAxes)
        ax.set_title(model_name, fontsize=10)
        return False, str(e)


# ----------------------------------------------------------------------
# Write one table of all models' fixed parameters (blank where unused)
# ----------------------------------------------------------------------
def write_parameter_table(model_names, out_path=f"{OUTDIR}/model_parameters.txt"):
    """Write the parameter table of the models."""
    per_model_params = {name: sample_fixed_parameters(name) for name in model_names}

    all_keys = sorted({key for params in per_model_params.values() for key in params})

    name_width = max(len(n) for n in model_names) + 2
    col_width = max((len(k) for k in all_keys), default=8) + 2
    col_width = max(col_width, 12)

    def fmt(value):
        """Format a value for output."""
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines = []
    header = f"{'model':<{name_width}}" + "".join(f"{k:<{col_width}}" for k in all_keys)
    lines.append(header)
    lines.append("-" * len(header))
    for name in model_names:
        params = per_model_params[name]
        row = f"{name:<{name_width}}" + "".join(
            f"{fmt(params.get(k, '')):<{col_width}}" for k in all_keys
        )
        lines.append(row)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------
# Main: one figure per survey, one subplot per model (grid layout)
# ----------------------------------------------------------------------
def make_survey_figure(survey_name, survey_cfg, model_names):
    """Figure of all models for a survey."""
    bands_list = survey_cfg["bands"]
    band_colors = survey_cfg["colors"]
    time_days = survey_cfg["time_days"]

    n = len(model_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    report = {}
    for ax, name in zip(axes, model_names):
        ok, err = plot_model(name, ax, bands_list, band_colors, time_days)
        report[name] = "OK" if ok else err

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"redback TDE models — {survey_name} bands, fixed per-model prior sample "
        f"(seed={SEED}, z={REDSHIFT}, {int(time_days[-1])} days)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = f"{OUTDIR}/tde_models_{survey_name.lower()}_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return report


def main():
    model_names = get_tde_model_names()

    write_parameter_table(model_names)

    all_reports = {}
    for survey_name, survey_cfg in SURVEYS.items():
        print(f"\n=== {survey_name} ===")
        report = make_survey_figure(survey_name, survey_cfg, model_names)
        all_reports[survey_name] = report

        print(f"--- {survey_name} model status report ---")
        for name, status in report.items():
            print(f"{name:55s}: {status}")

    return all_reports


if __name__ == "__main__":
    main()