import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
N_TARGET = 8000        # target number of galaxies
BOX_SIZE = 1000.0      # Mpc (comoving)
GRID = 128             # grid size (GRID^3)
BIAS = 1.5             # galaxy bias
SEED = 42

rng = np.random.default_rng(SEED)

# ===============================
# SIMPLE POWER SPECTRUM P(k)
# ===============================
def Pk(k):
    """Toy power spectrum ~ k^n exp[-(k/kc)^2]"""
    n = -1.8
    kc = 0.2
    return np.where(k > 0, k**n * np.exp(-(k/kc)**2), 0.0)

# ===============================
# BUILD K-GRID
# ===============================
k = np.fft.fftfreq(GRID, d=BOX_SIZE / GRID) * 2 * np.pi
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

# ===============================
# GAUSSIAN RANDOM FIELD
# ===============================
amp = np.sqrt(Pk(k_mag) / 2.0)
noise = rng.normal(size=k_mag.shape) + 1j * rng.normal(size=k_mag.shape)
delta_k = amp * noise
delta_k[0, 0, 0] = 0.0  # remove DC mode

delta_x = np.fft.ifftn(delta_k).real
delta_x *= BIAS

# ===============================
# LOGNORMAL TRANSFORM
# ===============================
sigma2 = np.var(delta_x)
rho = np.exp(delta_x - sigma2 / 2.0)   # mean ~ 1

# ===============================
# POISSON SAMPLE GALAXIES
# ===============================
cell_vol = (BOX_SIZE / GRID)**3
nbar = N_TARGET / BOX_SIZE**3

lam = nbar * rho * cell_vol
N_cell = rng.poisson(lam)

coords = np.linspace(0, BOX_SIZE, GRID, endpoint=False)
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

xs, ys, zs = [], [], []
for i in range(GRID):
    for j in range(GRID):
        for k in range(GRID):
            if N_cell[i, j, k] > 0:
                n = N_cell[i, j, k]
                xs.append(X[i, j, k] + rng.random(n) * BOX_SIZE / GRID)
                ys.append(Y[i, j, k] + rng.random(n) * BOX_SIZE / GRID)
                zs.append(Z[i, j, k] + rng.random(n) * BOX_SIZE / GRID)

x = np.concatenate(xs)
y = np.concatenate(ys)
z = np.concatenate(zs)

print(f"Generated {len(x)} galaxies")

# ===============================
# PROJECT TO SKY
# ===============================
r = np.sqrt(x**2 + y**2 + z**2)

ra  = np.degrees(np.arctan2(y, x)) % 360
dec = np.degrees(np.arcsin(z / r))

# mock redshift (linear is fine for testing)
z_obs = r / BOX_SIZE * 2.0

# ===============================
# UNIFORM COMPARISON SAMPLE
# ===============================
ra_u = rng.uniform(0, 360, len(ra))
dec_u = np.degrees(np.arcsin(rng.uniform(-1, 1, len(ra))))

# ===============================
# PLOTS
# ===============================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(ra_u, dec_u, s=1, alpha=0.5)
plt.title("Uniform sky (no clustering)")
plt.xlabel("RA [deg]")
plt.ylabel("Dec [deg]")

plt.subplot(1, 2, 2)
plt.scatter(ra, dec, s=1, alpha=0.5)
plt.title("Lognormal clustered sky")
plt.xlabel("RA [deg]")

plt.tight_layout()
plt.show()