# --- Torch-only version of the pieces in your snippet (minimal changes) ---

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch

from cosmology import CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction, LastMajorMerger
from galaxy import Galaxy
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution

# Type alias now torch-based
ArrayLike = Union[float, torch.Tensor]


# -------------------------------
# Torch bilinear interpolator
# -------------------------------
class TorchRectBilinearInterpolator:
    """
    A small, torch-only replacement for scipy RegularGridInterpolator on a 2D rectilinear grid.
    Supports vectorized queries, returns 0 outside bounds (fill_value=0.0).
    """
    def __init__(self, z_grid: torch.Tensor, th_grid: torch.Tensor, values: torch.Tensor):
        # Expect shapes: z_grid (Nz,), th_grid (Nm,), values (Nz,Nm)
        assert z_grid.ndim == 1 and th_grid.ndim == 1 and values.ndim == 2
        assert values.shape == (z_grid.numel(), th_grid.numel())
        self.zg = z_grid
        self.tg = th_grid
        self.val = values
        self.device = values.device
        self.dtype = values.dtype
        self.zmin = z_grid[0]
        self.zmax = z_grid[-1]
        self.tmin = th_grid[0]
        self.tmax = th_grid[-1]

    @torch.no_grad()
    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (..., 2) tensor with columns [z, th]
        returns: (...) interpolated values (same leading shape)
        """
        if not torch.is_tensor(pts):
            pts = torch.as_tensor(pts, device=self.device, dtype=self.dtype)
        else:
            pts = pts.to(device=self.device, dtype=self.dtype)

        z = pts[..., 0]
        th = pts[..., 1]

        # Mask for out-of-bounds → 0
        oob = (z < self.zmin) | (z > self.zmax) | (th < self.tmin) | (th > self.tmax)

        # Find right indices
        # clamp to valid interior so i0>=0 and i1<=last
        iz1 = torch.searchsorted(self.zg, z, right=True).clamp(1, self.zg.numel() - 1)
        iz0 = iz1 - 1
        it1 = torch.searchsorted(self.tg, th, right=True).clamp(1, self.tg.numel() - 1)
        it0 = it1 - 1

        z0 = self.zg[iz0]
        z1 = self.zg[iz1]
        t0 = self.tg[it0]
        t1 = self.tg[it1]

        # Avoid division-by-zero if grid has repeated points (shouldn't happen)
        dz = torch.where((z1 > z0), (z1 - z0), torch.ones_like(z1))
        dt = torch.where((t1 > t0), (t1 - t0), torch.ones_like(t1))

        uz = (z - z0) / dz
        ut = (th - t0) / dt

        f00 = self.val[iz0, it0]
        f10 = self.val[iz1, it0]
        f01 = self.val[iz0, it1]
        f11 = self.val[iz1, it1]

        # Bilinear mix
        v0 = f00 * (1 - uz) + f10 * uz
        v1 = f01 * (1 - uz) + f11 * uz
        out = v0 * (1 - ut) + v1 * ut

        # Apply 0 fill outside bounds
        out = torch.where(oob, torch.zeros_like(out), out)
        return out


# -------------------------------
# Torch population cache
# -------------------------------
@dataclass
class PopulationCache:
    Nz: int
    Nm: int
    seed: Optional[int] = None
    rng: torch.Generator = field(init=False)

    # cached tensors for EMRI and TDE populations
    Ntot_emri: Optional[torch.Tensor] = None        # (Nz,Nm)
    masses_emri: Optional[torch.Tensor] = None      # (Nz,Nm,S)
    Ntot_tde: Optional[torch.Tensor] = None         # (Nz,Nm)
    masses_tde: Optional[torch.Tensor] = None       # (Nz,Nm,S)

    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        self.rng = torch.Generator(device=self.device)
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

    def _to_tensor(self, x: ArrayLike, shape=None) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(device=self.device, dtype=self.dtype)
        else:
            t = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if shape is not None:
            t = t.reshape(shape)
        return t

    def _broadcast_N(self, N: ArrayLike) -> torch.Tensor:
        N = self._to_tensor(N)
        if N.ndim == 0:
            return torch.full((self.Nz, self.Nm), float(N), device=self.device, dtype=self.dtype)
        if N.shape == (self.Nm,):
            return N.unsqueeze(0).expand(self.Nz, self.Nm)
        if N.shape == (self.Nz,):
            return N.unsqueeze(1).expand(self.Nz, self.Nm)
        if N.shape == (self.Nz, self.Nm):
            return N
        raise ValueError(f"Ntot has incompatible shape {tuple(N.shape)}; expected scalar, (Nm,), (Nz,), or (Nz,Nm).")

    def _broadcast_masses(self, M: Optional[ArrayLike], n_species: int, mass_range: Tuple[float, float]) -> torch.Tensor:
        S = max(1, int(n_species))
        if M is None:
            low, high = mass_range
            return (low + (high - low) * torch.rand((self.Nz, self.Nm, S), generator=self.rng, device=self.device, dtype=self.dtype))
        M = self._to_tensor(M)
        if M.ndim == 0:
            return torch.full((self.Nz, self.Nm, 1), float(M), device=self.device, dtype=self.dtype)
        if M.ndim == 1:
            if M.numel() != S:
                # more flexible: tile what is provided
                return M.view(1, 1, -1).expand(self.Nz, self.Nm, M.numel())
            return M.view(1, 1, S).expand(self.Nz, self.Nm, S)
        if M.ndim == 2:
            if M.shape[0] == self.Nm:
                return M.unsqueeze(0).expand(self.Nz, -1, -1)
            if M.shape[0] == self.Nz:
                return M.unsqueeze(1).expand(-1, self.Nm, -1)
            raise ValueError(f"2-D masses must be (Nm,S) or (Nz,S); got {tuple(M.shape)}.")
        if M.ndim == 3:
            if M.shape[:2] != (self.Nz, self.Nm):
                raise ValueError(f"3-D masses must start with (Nz,Nm,S); got {tuple(M.shape)}.")
            return M
        raise ValueError(f"component_masses has incompatible ndim={M.ndim}.")

    def set_emri(self, Ntot: ArrayLike = 1.0e5, component_masses: Optional[ArrayLike] = None,
                 n_species: int = 1, mass_range: Tuple[float, float] = (10.0, 10.0)):
        self.Ntot_emri = self._broadcast_N(Ntot)
        self.masses_emri = self._broadcast_masses(component_masses, n_species, mass_range)
        return self

    def set_tde(self, Ntot: ArrayLike = 1.0e5, component_masses: Optional[ArrayLike] = None,
                n_species: int = 1, mass_range: Tuple[float, float] = (1.0, 1.0)):
        self.Ntot_tde = self._broadcast_N(Ntot)
        self.masses_tde = self._broadcast_masses(component_masses, n_species, mass_range)
        return self

    # Flatten getters return torch tensors
    def get_emri_flat(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.Ntot_emri is None or self.masses_emri is None:
            self.set_emri()
        N_flat = self.Ntot_emri.reshape(-1)                        # (N_flat,)
        M_flat = self.masses_emri.reshape(self.Nz * self.Nm, -1)   # (N_flat,S)
        mbar_flat = M_flat.mean(dim=1)                             # (N_flat,)
        return N_flat, M_flat, mbar_flat

    def get_tde_flat(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.Ntot_tde is None or self.masses_tde is None:
            self.set_tde()
        N_flat = self.Ntot_tde.reshape(-1)
        M_flat = self.masses_tde.reshape(self.Nz * self.Nm, -1)
        mbar_flat = M_flat.mean(dim=1)
        return N_flat, M_flat, mbar_flat

    def reseed(self, seed: Optional[int] = None):
        self.seed = seed
        self.__post_init__()  # reset generator with new seed


# -------------------------------
# Torch Distribution2D
# -------------------------------
class Distribution2D:
    """
    Torch-first base: builds grid and offers helpers (mesh/flatten/reshape),
    cosmology terms, MBH MF, EMRI rate, interpolation + sampling.
    """
    def __init__(self, limits_z, limits_theta, npoints=200, grid_spacing="linear", device="cpu", dtype=torch.float64):
        self.device = device
        self.dtype = dtype

        if grid_spacing == "linear":
            self.z_grid = torch.linspace(limits_z[0], limits_z[1], npoints, device=device, dtype=dtype)
            self.theta_grid = torch.linspace(limits_theta[0], limits_theta[1], npoints, device=device, dtype=dtype)
        else:
            self.z_grid = torch.logspace(
                torch.log10(torch.tensor(limits_z[0], device=device, dtype=dtype)),
                torch.log10(torch.tensor(limits_z[1], device=device, dtype=dtype)),
                npoints, device=device, dtype=dtype
            )
            self.theta_grid = torch.logspace(
                torch.log10(torch.tensor(limits_theta[0], device=device, dtype=dtype)),
                torch.log10(torch.tensor(limits_theta[1], device=device, dtype=dtype)),
                npoints, device=device, dtype=dtype
            )

        # Meshgrid (Nz,Nm)
        self.Z, self.TH = torch.meshgrid(self.z_grid, self.theta_grid, indexing='ij')
        self.Nz, self.Nm = self.Z.shape
        self.N_flat = self.Nz * self.Nm

        # External models (likely NumPy internally; we convert outputs back to torch)
        self.cosmo = CosmologyModel()
        self._gsmf = GalaxyStellarMassFunction()
        self._mbhmf = MBHMassFunction(gsmf=self._gsmf)

        self.T_obs_det = 4.0

        # Torch population cache
        self.population_cache = PopulationCache(self.Nz, self.Nm, seed=None, device=self.device, dtype=self.dtype)

        # Interpolator cache
        self.pdf_grid: Optional[torch.Tensor] = None
        self.interp: Optional[TorchRectBilinearInterpolator] = None

    # ----- population setters -----
    def set_emri_population(self, Ntot=1.0e5, component_masses=None, n_species=1, mass_range=(10.0, 10.0), seed=None):
        if seed is not None:
            self.population_cache.reseed(seed)
        self.population_cache.set_emri(Ntot=Ntot, component_masses=component_masses, n_species=n_species, mass_range=mass_range)
        return self

    def set_tde_population(self, Ntot=1.0e5, component_masses=None, n_species=1, mass_range=(1.0, 1.0), seed=None):
        if seed is not None:
            self.population_cache.reseed(seed)
        self.population_cache.set_tde(Ntot=Ntot, component_masses=component_masses, n_species=n_species, mass_range=mass_range)
        return self

    # ----- grid helpers -----
    def mesh(self):
        return self.Z, self.TH

    def flatten(self, A2: ArrayLike) -> torch.Tensor:
        return torch.as_tensor(A2, device=self.device, dtype=self.dtype).reshape(-1)

    def unflatten(self, A1: ArrayLike) -> torch.Tensor:
        return torch.as_tensor(A1, device=self.device, dtype=self.dtype).reshape(self.Nz, self.Nm)

    # ----- cosmology / selection -----
    def dVc_dz_on_grid(self) -> torch.Tensor:
        # Boundary conversion: external cosmo likely returns numpy
        dVc_z_np = self.cosmo.dVc_dz(self.z_grid.detach().cpu().numpy())  # (Nz,)
        dVc_z = torch.as_tensor(dVc_z_np, device=self.device, dtype=self.dtype)
        return dVc_z[:, None]  # (Nz,1)

    def T_obs_on_grid(self) -> torch.Tensor:
        Z, _ = self.mesh()
        return torch.as_tensor(self.T_obs_det, device=self.device, dtype=self.dtype) / (1.0 + Z)

    def phi_MBH_on_grid(self, n_points_mass=512) -> torch.Tensor:
        # Boundary conversion: MBH MF likely numpy
        Z_np = self.Z.detach().cpu().numpy()
        TH_np = self.TH.detach().cpu().numpy()
        phi_np = self._mbhmf.evaluate_on_mesh(Z_np, TH_np, n_points_mass=n_points_mass)  # (Nz,Nm) numpy
        return torch.as_tensor(phi_np, device=self.device, dtype=self.dtype)

    # ----- EMRI/TDE rate -----
    def emri_rate_on_grid(self, gamma=1.5, kind='EMRI', kvir=1.0, mbar=None,
                          Ntot=None, component_masses=None, unit='Gyr') -> torch.Tensor:
        # Prepare flat inputs (torch)
        z_flat = self.Z.reshape(-1)
        lgMBH_flat = self.TH.reshape(-1)
        lgMgal_flat = torch.as_tensor(
            Galaxy.lgMgal_from_lgMBH(lgMBH_flat.detach().cpu().numpy()),
            device=self.device, dtype=self.dtype
        )

        # Domain objects (likely numpy-heavy inside)
        gal = Galaxy(lgMgal_flat.detach().cpu().numpy(), z_flat.detach().cpu().numpy())
        nsc = NSC(gal, lgMBH_flat.detach().cpu().numpy())
        prof = DehnenProfile(nsc, gamma)
        relax = RelaxationModel(nsc, prof)
        rate = RateModel(nsc)
        evol = CuspEvolution(nsc, relax, rate, LastMajorMerger(self.cosmo))

        # ---- population inputs
        if (Ntot is None) or (component_masses is None):
            Ntot_flat, masses_flat, mbar_flat_from_masses = self.population_cache.get_emri_flat()
        else:
            # Normalize explicit overrides (torch)
            Ntot_flat = torch.as_tensor(Ntot, device=self.device, dtype=self.dtype).reshape(-1)
            if Ntot_flat.numel() == 1:
                Ntot_flat = Ntot_flat.expand(self.N_flat)
            masses = torch.as_tensor(component_masses, device=self.device, dtype=self.dtype)
            if masses.ndim == 0:
                masses_flat = masses.view(1, 1).expand(self.N_flat, 1)
            elif masses.ndim == 1:
                masses_flat = masses.view(1, -1).expand(self.N_flat, -1)
            elif masses.ndim == 2 and masses.shape[0] == self.N_flat:
                masses_flat = masses
            else:
                raise ValueError("component_masses must be scalar, (S,), or (N_flat,S).")
            mbar_flat_from_masses = masses_flat.mean(dim=1)

        # mbar precedence
        if mbar is None:
            mbar_vec = mbar_flat_from_masses
        else:
            mbar_arr = torch.as_tensor(mbar, device=self.device, dtype=self.dtype)
            if mbar_arr.ndim == 0:
                mbar_vec = mbar_arr.expand(self.N_flat)
            else:
                if mbar_arr.numel() != self.N_flat:
                    raise ValueError("mbar array must be length N_flat when provided.")
                mbar_vec = mbar_arr

        # ---- call evolution (expects numpy); convert in/out once
        tau, T_c, t_EMRI, Gamma_hat = evol.evaluate_tau(
            Ntot=Ntot_flat.detach().cpu().numpy(),
            component_masses=masses_flat.detach().cpu().numpy(),
            kvir=kvir, kind=kind, mbar=mbar_vec.detach().cpu().numpy(), unit=unit
        )

        tau_t = torch.as_tensor(tau, device=self.device, dtype=self.dtype)
        Gamma_hat_t = torch.as_tensor(Gamma_hat, device=self.device, dtype=self.dtype)

        if kind.upper() == 'EMRI':
            universal_rate_np = rate.universal_EMRI_rate(tau)
        else:
            universal_rate_np = rate.universal_TDE_rate(tau)
        universal_rate_t = torch.as_tensor(universal_rate_np, device=self.device, dtype=self.dtype)

        Gamma_flat = Gamma_hat_t * universal_rate_t  # (N_flat,)
        return Gamma_flat.reshape(self.Nz, self.Nm)

    # ----- interpolation & evaluation -----
    def interpolate(self, pdf_2d_normalized: torch.Tensor):
        pdf_2d_normalized = torch.as_tensor(pdf_2d_normalized, device=self.device, dtype=self.dtype)
        self.pdf_grid = pdf_2d_normalized
        self.interp = TorchRectBilinearInterpolator(self.z_grid, self.theta_grid, pdf_2d_normalized)

    def evaluate_at_z_theta(self, z, theta) -> torch.Tensor:
        if self.interp is None:
            raise RuntimeError("Call .pdf(...) once before evaluating.")
        zt = torch.stack([
            torch.as_tensor(z, device=self.device, dtype=self.dtype).reshape(-1),
            torch.as_tensor(theta, device=self.device, dtype=self.dtype).reshape(-1)
        ], dim=1)
        return self.interp(zt)

    # ----- CDF & sampling -----
    def cdf(self, **pdf_kwargs) -> torch.Tensor:
        if self.pdf_grid is None:
            self.pdf_grid = self.pdf(**pdf_kwargs)
        flat = self.pdf_grid.reshape(-1)
        c = torch.cumsum(flat, dim=0)
        total = c[-1]
        if (not torch.isfinite(total)) or (total <= 0):
            raise RuntimeError("PDF integrates to zero; cannot sample.")
        return c / total

    @torch.no_grad()
    def draw_samples(self, size: int, **pdf_kwargs):
        cdf = self.cdf(**pdf_kwargs)  # (N_flat,)
        u = torch.rand(size, device=self.device, dtype=self.dtype)
        idx = torch.searchsorted(cdf, u)
        iz = idx // self.Nm
        it = idx % self.Nm
        z_s = self.z_grid[iz]
        th_s = self.theta_grid[it]
        return z_s, th_s

    def pdf(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError


# -------------------------------
# Example derived distribution
# -------------------------------
class dN_dlgMBH_dz(Distribution2D):
    def __init__(self, limits_z, limits_theta, npoints=200, grid_spacing='linear', device="cpu", dtype=torch.float64):
        super().__init__(limits_z, limits_theta, npoints, grid_spacing, device, dtype)

    @torch.no_grad()
    def pdf(self, gamma=1.5, kind='EMRI', kvir=1.0, mbar=None, n_points_mass=512) -> torch.Tensor:
        dVc  = self.dVc_dz_on_grid()                       # (Nz,1) torch
        phi  = self.phi_MBH_on_grid(n_points_mass)         # (Nz,Nm) torch
        Gamma = self.emri_rate_on_grid(gamma=gamma, kind=kind, kvir=kvir, mbar=mbar)  # (Nz,Nm) torch
        Tobs = self.T_obs_on_grid()                        # (Nz,Nm) torch

        pdf = dVc * phi * Gamma * Tobs                    # implicit broadcast
        total = torch.sum(pdf)

        if (not torch.isfinite(total)) or (total <= 0):
            raise RuntimeError("Computed PDF is zero everywhere. Check inputs/units.")

        pdf = pdf / total
        self.interpolate(pdf)
        return pdf


# -------------------------------
# Example construction
# -------------------------------
distribution = Distribution2D((0.01, 10.0), (4.0, 8.5), npoints=10, device='cpu', dtype=torch.float64)

