
import numpy as np
import seaborn as sns
import matplotlib
import h5py
import corner
import matplotlib.lines as mlines

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':28})
matplotlib.rc('text', usetex=True)

pal = sns.color_palette(palette="colorblind").as_hex()


default_kwargs = dict(
    bins=32,
    smooth=0.9,
    truth_color='black',
    # quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    max_n_ticks = 4,
    show_titles=True,
    label_kwargs=dict(fontsize=26),
    labelpad=0.2,
    title_kwargs={"fontsize": 18}
)

true_x_MIX = {
        "log10_M": {"lam_A": -2.5, "lam_B": -1.3},
        "log10_mu": {"lam_A": -2.5, "lam_B": -3.0},
        "a": {"mu_A": 0.65, "sigma_A": 0.03, "mu_B": 0.3, "sigma_B": 0.01},
        "e0": {},
        "Y0": {},
        "dist": {"lam": 3},
        "qS": {},
        "phiS": {},
        "qK": {},
        "phiK": {},
        "Phi_phi0": {},
        "Phi_theta0": {},
        "Phi_r0": {},
        "T" : {},
        }

# Ground truths for vertical lines
truths = [
            true_x_MIX['log10_M']['lam_A'],
            true_x_MIX['log10_mu']['lam_A'], 
            true_x_MIX['a']['mu_A'],
            true_x_MIX['a']['sigma_A'],
            true_x_MIX['log10_M']['lam_B'],
            true_x_MIX['log10_mu']['lam_B'], 
            true_x_MIX['a']['mu_B'],
            true_x_MIX['a']['sigma_B'],
            0.7
        ]

# Load posterior samples

output = '/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/_pop_MIX_B_B_NO_SF/1E2_events/inference'

hf = h5py.File(f'{output}/result.hdf5', 'r')


A_lambda_M = np.array(hf.get('posterior_samples')['A_lambda_M'])
A_lambda_mu = np.array(hf.get('posterior_samples')['A_lambda_mu'])
A_mu_a = np.array(hf.get('posterior_samples')['A_mu_a'])
A_sigma_a = np.array(hf.get('posterior_samples')['A_sigma_a'])

B_lambda_M = np.array(hf.get('posterior_samples')['B_lambda_M'])
B_lambda_mu = np.array(hf.get('posterior_samples')['B_lambda_mu'])
B_mu_a = np.array(hf.get('posterior_samples')['B_mu_a'])
B_sigma_a = np.array(hf.get('posterior_samples')['B_sigma_a'])

weight = np.array(hf.get('posterior_samples')['weight'])

samples_1 = np.vstack([A_lambda_M, A_lambda_mu, A_mu_a, A_sigma_a,
                    B_lambda_M, B_lambda_mu, B_mu_a, B_sigma_a,
                    weight]).T
# =======================================
output = '/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/_pop_MIX_B_B_NO_SF/1E3_events/inference'

hf = h5py.File(f'{output}/result.hdf5', 'r')

A_lambda_M = np.array(hf.get('posterior_samples')['A_lambda_M'])
A_lambda_mu = np.array(hf.get('posterior_samples')['A_lambda_mu'])
A_mu_a = np.array(hf.get('posterior_samples')['A_mu_a'])
A_sigma_a = np.array(hf.get('posterior_samples')['A_sigma_a'])

B_lambda_M = np.array(hf.get('posterior_samples')['B_lambda_M'])
B_lambda_mu = np.array(hf.get('posterior_samples')['B_lambda_mu'])
B_mu_a = np.array(hf.get('posterior_samples')['B_mu_a'])
B_sigma_a = np.array(hf.get('posterior_samples')['B_sigma_a'])

weight = np.array(hf.get('posterior_samples')['weight'])

samples_2 = np.vstack([A_lambda_M, A_lambda_mu, A_mu_a, A_sigma_a,
                    B_lambda_M, B_lambda_mu, B_mu_a, B_sigma_a,
                    weight]).T

# =======================================

output = '/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/_pop_MIX_B_B_NO_SF/1E4_events/inference'

hf = h5py.File(f'{output}/result.hdf5', 'r')

A_lambda_M = np.array(hf.get('posterior_samples')['A_lambda_M'])
A_lambda_mu = np.array(hf.get('posterior_samples')['A_lambda_mu'])
A_mu_a = np.array(hf.get('posterior_samples')['A_mu_a'])
A_sigma_a = np.array(hf.get('posterior_samples')['A_sigma_a'])

B_lambda_M = np.array(hf.get('posterior_samples')['B_lambda_M'])
B_lambda_mu = np.array(hf.get('posterior_samples')['B_lambda_mu'])
B_mu_a = np.array(hf.get('posterior_samples')['B_mu_a'])
B_sigma_a = np.array(hf.get('posterior_samples')['B_sigma_a'])

weight = np.array(hf.get('posterior_samples')['weight'])

samples_3 = np.vstack([A_lambda_M, A_lambda_mu, A_mu_a, A_sigma_a,
                    B_lambda_M, B_lambda_mu, B_mu_a, B_sigma_a,
                    weight]).T

# =======================================

labels = ["$\\Lambda^A_M$", "$\\Lambda^A_{\\mu}$", "$\\mu^A_a$", "$\\sigma^A_a$",
            "$\\Lambda^B_M$", "$\\Lambda^B_{\\mu}$", "$\\mu^B_a$", "$\\sigma^B_a$",
            "weight"]

fig = corner.corner(
    samples_1, color=pal[3], truths=truths, labels=labels, hist_kwargs=dict(density=True,color=pal[3], linewidth=2),contour_kwargs=dict(linewidths=2), **default_kwargs
)

fig = corner.corner(
    samples_2, fig=fig, color=pal[2], truths=truths, labels=labels, hist_kwargs=dict(density=True, color=pal[2], linewidth=2), contour_kwargs=dict(linewidths=2), **default_kwargs
)

fig = corner.corner(
    samples_3, fig=fig, color=pal[1], truths=truths, labels=labels, hist_kwargs=dict(density=True, color=pal[1], linewidth=2), contour_kwargs=dict(linewidths=2), **default_kwargs
)

E2 = mlines.Line2D([],[], color=pal[3], linewidth=2,label='$10^2$')
E3 = mlines.Line2D([],[], color=pal[2], linewidth=2,label='$10^3$')
E4 = mlines.Line2D([],[], color=pal[1], linewidth=2,label='$10^4$')

fig.legend(handles=[E2, E3, E4], loc=(0.8,0.75), fontsize=28, frameon=True, title='Injections')

fig.savefig(f'/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/1D-posteriors/pop_MIX_B_B.png',dpi=500, bbox_inches='tight')
fig.savefig(f'/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/1D-posteriors/pop_MIX_B_B.pdf',dpi=500, bbox_inches='tight')
