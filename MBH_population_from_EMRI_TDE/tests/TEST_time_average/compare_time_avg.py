"""Compare time-dependent and time-averaged EMRI/TDE rates on the same galaxy population."""
# Compare time-DEPENDENT rate (UniversalRate(tau), CuspEvolution.expected_objects_in_time)
# with the time-AVERAGED / constant rate (N_x/t_f * Theta(t_rlx - T_c), as in sources_num.py)
# on the SAME galaxy population (same seed, same cusp ages).
# Setup (galaxy sampling, NSC, relaxation, CuspEvolution) is exec'd from data_generation/sources_num.py
# up to the MBH_MASK line, so it can't drift from the production pipeline.
# Run from the repo root:  python tests/TEST_time_average/compare_time_avg.py <same args as sources_num.py> [--OUT x.json]
import sys, json
out_path = "tests/TEST_time_average/time_avg_result.json"
if "--OUT" in sys.argv:
    i = sys.argv.index("--OUT"); out_path = sys.argv[i + 1]; del sys.argv[i:i + 2]

src = open("data_generation/sources_num.py").read()
exec(compile(src.split("MBH_MASK =")[0], "sources_num_setup", "exec"))

MBH_MASK = (gal_obj.lgMBH_mass > 4.0) & (gal_obj.lgMBH_mass <= 7.0) & gal_obj.occupation_occurs
w = mc_weight_i[nucleation_indices]   # weights of nucleated galaxies only
z = NSC_obj.gal.z_gal
T_obs = {'EMRI': args.T_OBS_EMRI, 'TDE': args.T_OBS_TDE}   # yr
T_obs_Gyr = {'EMRI': T_obs_EMRI_Gyr, 'TDE': T_obs_TDE_Gyr}

cusp_evolution_object.cusp_turn_on_time(kvir=1.0)  # keep the already-sampled turn-on times
active = (np.atleast_1d(cusp_evolution_object.cusp_age(kvir=1.0)) < t_rlx_Gyr).astype(float)

def wq(x, ww, q):
    o = np.argsort(x); c = np.cumsum(ww[o]); return float(x[o][np.searchsorted(c, q * c[-1])])

res = {}; dump = dict(z=z, w=w, active=active, mask=MBH_MASK, lgMBH=gal_obj.lgMBH_mass)
for kind in ['EMRI', 'TDE']:
    lam_td = np.where(MBH_MASK, cusp_evolution_object.expected_objects_in_time(T_obs=T_obs[kind], kvir=1.0, kind=kind), 0.0)
    lam_avg = np.where(MBH_MASK, const_rate_obj.constant_rate(kind=kind) * T_obs_Gyr[kind] * active / (1.0 + z), 0.0)
    dump[f'lam_avg_nocut_{kind}'] = np.where(MBH_MASK, const_rate_obj.constant_rate(kind=kind) * T_obs_Gyr[kind] / (1.0 + z), 0.0)  # no T_c<t_rlx cut
    dump[f'lam_td_{kind}'] = lam_td; dump[f'lam_avg_{kind}'] = lam_avg
    for name, lam in [('time_dependent', lam_td), ('time_averaged', lam_avg)]:
        W = w * lam
        res.setdefault(name, {})[kind] = dict(
            Lambda=float(W.sum()), z_mean=float((W * z).sum() / W.sum()),
            z_median=wq(z, W, 0.5), z_p90=wq(z, W, 0.9),
            frac_z_lt_0p3=float(W[z < 0.3].sum() / W.sum()), frac_z_lt_1=float(W[z < 1.0].sum() / W.sum()),
            n_eff=float(W.sum() ** 2 / np.sum(W ** 2)))
for name in res:
    res[name]['TDE_over_EMRI'] = res[name]['TDE']['Lambda'] / res[name]['EMRI']['Lambda']
res['boost_TDE_over_EMRI_avg_vs_td'] = res['time_averaged']['TDE_over_EMRI'] / res['time_dependent']['TDE_over_EMRI']
np.savez_compressed(out_path.replace('.json', '.npz'), **dump)
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
