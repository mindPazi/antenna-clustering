"""
Funzioni per calcolo array antenna con clustering
Allineato al notebook clustering_comparison.ipynb
"""

import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import Delaunay
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# OPT: Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        xp = cp  # Use CuPy for array operations
    else:
        xp = np
        GPU_AVAILABLE = False
except ImportError:
    xp = np
    GPU_AVAILABLE = False


@dataclass
class LatticeConfig:
    """Configurazione lattice array"""
    Nz: int  # Number of rows
    Ny: int  # Number of columns
    dist_z: float  # antenna distance on z axis [times lambda]
    dist_y: float  # antenna distance on y axis [times lambda]
    lattice_type: int = 1  # 1=Rectangular


@dataclass
class SystemConfig:
    """Parametri sistema"""
    freq: float  # [Hz]
    lambda_: float = field(init=False)
    beta: float = field(init=False)
    azi0: float = 0.0
    ele0: float = 0.0
    dele: float = 0.5
    dazi: float = 0.5

    def __post_init__(self):
        self.lambda_ = 3e8 / self.freq
        self.beta = 2 * np.pi / self.lambda_


@dataclass
class MaskConfig:
    """Parametri maschera SLL"""
    elem: float = 30.0
    azim: float = 60.0
    SLL_level: float = 20.0
    SLLin: float = 15.0


@dataclass
class ElementPatternConfig:
    """Configurazione pattern elemento"""
    P: int = 1
    Gel: float = 5.0
    load_file: int = 0


class AntennaArray:
    """Classe per array di antenne con clustering"""

    def __init__(self, lattice: LatticeConfig, system: SystemConfig,
                 mask: MaskConfig, eef_config: Optional[ElementPatternConfig] = None):
        self.lattice = lattice
        self.system = system
        self.mask = mask
        self.eef_config = eef_config or ElementPatternConfig()
        self.Nel = lattice.Nz * lattice.Ny

        self._compute_lattice_vectors()
        self._generate_lattice()
        self._generate_polar_coordinates()
        self._generate_element_pattern()
        self._generate_mask()

    def _compute_lattice_vectors(self):
        lambda_ = self.system.lambda_
        dz = self.lattice.dist_z * lambda_
        dy = self.lattice.dist_y * lambda_
        self.x1 = np.array([dy, 0.0])
        self.x2 = np.array([0.0, dz])

    def _generate_lattice(self):
        Nz = self.lattice.Nz
        Ny = self.lattice.Ny

        if Nz % 2 == 1:
            M = np.arange(-(Nz - 1) / 2, (Nz - 1) / 2 + 1)
        else:
            M = np.arange(-Nz / 2 + 1, Nz / 2 + 1)

        if Ny % 2 == 1:
            N = np.arange(-(Ny - 1) / 2, (Ny - 1) / 2 + 1)
        else:
            N = np.arange(-Ny / 2 + 1, Ny / 2 + 1)

        self.NN, self.MM = np.meshgrid(N, M)
        dz = self.x2[1]
        dy = self.x1[0]
        DELTA = max(self.x2[0], self.x1[1])

        self.Y = self.NN * dy
        self.Z = self.MM * dz
        self.Y[1::2, :] = self.Y[1::2, :] + DELTA

        self.Dz = np.max(self.Z) - np.min(self.Z)
        self.Dy = np.max(self.Y) - np.min(self.Y)
        self.Dy_total = self.Dy + self.x1[0]
        self.Dz_total = self.Dz + self.x2[1]
        self.ArrayMask = np.ones_like(self.Y)

    def _generate_polar_coordinates(self):
        beta = self.system.beta
        lambda_ = self.system.lambda_

        self.ele = np.arange(-90, 90 + self.system.dele, self.system.dele)
        self.azi = np.arange(-90, 90 + self.system.dazi, self.system.dazi)
        self.AZI, self.ELE = np.meshgrid(self.azi, self.ele)

        self.WWae = beta * np.cos(np.deg2rad(90 - self.ELE))
        self.Vvae = beta * np.sin(np.deg2rad(90 - self.ELE)) * np.sin(np.deg2rad(self.AZI))

        chi = 2
        self.Nw = int(np.floor(chi * 4 * self.Dz_total / lambda_))
        self.Nv = int(np.floor(chi * 4 * self.Dy_total / lambda_))

        ww = np.linspace(0, beta, self.Nw + 1)
        self.ww = np.concatenate([-np.flip(ww[1:]), ww])
        vv = np.linspace(0, beta, self.Nv + 1)
        self.vv = np.concatenate([-np.flip(vv[1:]), vv])

        self.WW, self.VV = np.meshgrid(self.ww, self.vv)

        WW_clipped = np.clip(self.WW / beta, -1, 1)
        self.ELEi = 90 - np.rad2deg(np.arccos(WW_clipped))

        denom = beta * np.sin(np.deg2rad(90 - self.ELEi))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.clip(self.VV / denom, -1, 1)
            self.AZIi = np.real(np.rad2deg(np.arcsin(ratio)))

        self.AZIi[self.Nv, :] = 0
        self.AZIi[self.Nv, 0] = 90
        self.AZIi[self.Nv, -1] = 90

    def _generate_element_pattern(self):
        P = self.eef_config.P
        Gel = self.eef_config.Gel

        if P == 0:
            self.Fel = np.ones_like(self.ELE)
        else:
            self.Fel = (np.cos(np.deg2rad(self.ELE * 0.9)) * np.cos(np.deg2rad(self.AZI * 0.9))) ** P

        if P == 0:
            self.Fel_VW = np.ones_like(self.ELEi)
        else:
            self.Fel_VW = (np.cos(np.deg2rad(self.ELEi * 0.9)) * np.cos(np.deg2rad(self.AZIi * 0.9))) ** P

        self.RPE = 20 * np.log10(np.abs(self.Fel) + 1e-10)
        self.RPE_ele_max = np.max(self.RPE) + Gel
        self.G_boresight = self.RPE_ele_max + 10 * np.log10(self.Nel)

    def _generate_mask(self):
        ele0 = self.system.ele0
        azi0 = self.system.azi0
        elem = self.mask.elem
        azim = self.mask.azim
        SLL_level = self.mask.SLL_level
        SLLin = self.mask.SLLin

        ele_cond = np.abs(self.ELE - ele0) <= elem
        azi_cond = np.abs(self.AZI - azi0) <= azim

        self.in_fov_mask = ele_cond & azi_cond
        self.out_fov_mask = ~self.in_fov_mask

        self.Mask_EA = np.full_like(self.ELE, self.G_boresight - SLL_level, dtype=float)
        self.Mask_EA[self.in_fov_mask] = self.G_boresight - SLLin

    def index_to_position_cluster(self, Cluster: List[np.ndarray],
                                   ElementExc: Optional[np.ndarray] = None):
        if ElementExc is None:
            ElementExc = np.ones((self.lattice.Nz, self.lattice.Ny))

        Ntrans = len(Cluster)
        max_Lsub = max(c.shape[0] for c in Cluster)

        Yc = np.full((max_Lsub, Ntrans), np.nan)
        Zc = np.full((max_Lsub, Ntrans), np.nan)
        Ac = np.full((max_Lsub, Ntrans), np.nan)

        min_NN = int(np.min(self.NN))
        min_MM = int(np.min(self.MM))

        for kk, cluster in enumerate(Cluster):
            for l1 in range(cluster.shape[0]):
                Iy = int(cluster[l1, 0] - min_NN)
                Iz = int(cluster[l1, 1] - min_MM)
                Yc[l1, kk] = self.Y[Iz, Iy]
                Zc[l1, kk] = self.Z[Iz, Iy]
                Ac[l1, kk] = ElementExc[Iz, Iy]

        return Yc, Zc, Ac

    def coefficient_evaluation(self, Zc_m: np.ndarray, Yc_m: np.ndarray, Lsub: np.ndarray):
        beta = self.system.beta
        ele0 = self.system.ele0
        azi0 = self.system.azi0

        v0 = beta * np.sin(np.deg2rad(90 - ele0)) * np.sin(np.deg2rad(azi0))
        w0 = beta * np.cos(np.deg2rad(90 - ele0))
        Phase_m = np.exp(-1j * (w0 * Zc_m + v0 * Yc_m))
        Amplit_m = 1.0 / Lsub
        c0 = Amplit_m * Phase_m
        return c0

    def kernel1_rpe(self, Lsub: np.ndarray, Ac: np.ndarray,
                    Yc: np.ndarray, Zc: np.ndarray, c0: np.ndarray):
        Ntrans = len(Lsub)

        # OPT: Cache flattened arrays
        if not hasattr(self, '_VV_flat'):
            self._VV_flat = self.VV.flatten()
            self._WW_flat = self.WW.flatten()
            self._Fel_VW_flat = self.Fel_VW.flatten()

        VV_flat = self._VV_flat
        WW_flat = self._WW_flat
        Fel_VW_flat = self._Fel_VW_flat
        Npoints = len(VV_flat)

        # OPT: Vectorized kernel computation
        all_Y = []
        all_Z = []
        cluster_indices = []

        for kk in range(Ntrans):
            Lsub_k = int(Lsub[kk])
            for jj in range(Lsub_k):
                if not np.isnan(Yc[jj, kk]) and not np.isnan(Zc[jj, kk]):
                    all_Y.append(Yc[jj, kk])
                    all_Z.append(Zc[jj, kk])
                    cluster_indices.append(kk)

        all_Y_np = np.array(all_Y)
        all_Z_np = np.array(all_Z)
        cluster_indices_np = np.array(cluster_indices)

        # OPT: GPU acceleration - transfer to GPU if available
        if GPU_AVAILABLE:
            all_Y_gpu = xp.asarray(all_Y_np)
            all_Z_gpu = xp.asarray(all_Z_np)
            VV_flat_gpu = xp.asarray(VV_flat)
            WW_flat_gpu = xp.asarray(WW_flat)
            Fel_VW_flat_gpu = xp.asarray(Fel_VW_flat)

            # Compute phases on GPU
            phases = xp.exp(1j * (xp.outer(VV_flat_gpu, all_Y_gpu) + xp.outer(WW_flat_gpu, all_Z_gpu)))
            phases = phases * Fel_VW_flat_gpu[:, xp.newaxis]

            # Transfer back to CPU for add.at (not available in CuPy)
            phases_np = cp.asnumpy(phases)
        else:
            # CPU computation
            phases_np = np.exp(1j * (np.outer(VV_flat, all_Y_np) + np.outer(WW_flat, all_Z_np)))
            phases_np = phases_np * Fel_VW_flat[:, np.newaxis]

        # OPT: Sum contributions using np.add.at
        KerFF_sub = np.zeros((Npoints, Ntrans), dtype=complex)
        np.add.at(KerFF_sub.T, cluster_indices_np, phases_np.T)

        FF = KerFF_sub @ c0.T
        FF_norm = FF / (np.max(np.abs(FF)) + 1e-10)
        FF_norm_2D = FF_norm.reshape(self.VV.shape)
        FF_norm_dB = 20 * np.log10(np.abs(FF_norm_2D) + 1e-10)

        # OPT: Cache interpolation setup - use LinearNDInterpolator for much faster repeated interpolation
        if not hasattr(self, '_interp_points'):
            self._interp_points = np.column_stack([WW_flat, VV_flat])
            self._interp_xi = np.column_stack([self.WWae.flatten(), self.Vvae.flatten()])
            # OPT: Pre-compute Delaunay triangulation (expensive, but done only once)
            self._delaunay = Delaunay(self._interp_points)

        # OPT: Use cached triangulation for fast interpolation
        values = FF_norm_dB.flatten()
        interpolator = LinearNDInterpolator(self._delaunay, values, fill_value=-100)
        FF_I_dB_flat = interpolator(self._interp_xi)
        FF_I_dB = FF_I_dB_flat.reshape(self.WWae.shape)

        Nel_active = np.sum(Lsub)
        FF_I_dB = FF_I_dB + self.RPE_ele_max + 10 * np.log10(Nel_active)

        return FF_norm_dB, FF_I_dB, KerFF_sub, np.abs(FF_norm_2D) ** 2

    def compute_cost_function(self, FF_I_dB: np.ndarray) -> int:
        Constr = FF_I_dB - self.Mask_EA
        Cm = np.sum(Constr > 0)
        return int(Cm)

    def compute_sll(self, FF_I_dB: np.ndarray, G_boresight: float = None):
        # Calcola G_boresight se non fornito
        if G_boresight is None:
            G_boresight = np.max(FF_I_dB)

        # SLL out-of-FoV (relativo a G_boresight -> NEGATIVO)
        sll_out_values = FF_I_dB[self.out_fov_mask]
        sll_out = (np.max(sll_out_values) - G_boresight) if len(sll_out_values) > 0 else -100

        # SLL in-FoV (secondo massimo, relativo a G_boresight -> NEGATIVO)
        sll_in_values = FF_I_dB[self.in_fov_mask]
        if len(sll_in_values) > 0:
            max_val = np.max(sll_in_values)
            second_max = np.max(sll_in_values[sll_in_values < max_val - 0.1]) if np.any(sll_in_values < max_val - 0.1) else max_val
            sll_in = second_max - G_boresight
        else:
            sll_in = -100

        return sll_in, sll_out

    def evaluate_clustering(self, Cluster: List[np.ndarray],
                            ElementExc: Optional[np.ndarray] = None) -> Dict:
        if ElementExc is None:
            ElementExc = np.ones((self.lattice.Nz, self.lattice.Ny))

        Yc, Zc, Ac = self.index_to_position_cluster(Cluster, ElementExc)
        Ntrans = len(Cluster)

        Lsub = np.array([c.shape[0] for c in Cluster])
        Zc_m = np.array([np.nanmean(Zc[:Lsub[k], k]) for k in range(Ntrans)])
        Yc_m = np.array([np.nanmean(Yc[:Lsub[k], k]) for k in range(Ntrans)])

        c0 = self.coefficient_evaluation(Zc_m, Yc_m, Lsub)
        FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm = self.kernel1_rpe(Lsub, Ac, Yc, Zc, c0)

        Cm = self.compute_cost_function(FF_I_dB)

        # Calcola G_boresight prima di compute_sll
        G_boresight = self.RPE_ele_max + 10 * np.log10(np.sum(Lsub))
        sll_in, sll_out = self.compute_sll(FF_I_dB, G_boresight)

        max_idx = np.unravel_index(np.argmax(FF_I_dB), FF_I_dB.shape)
        theta_max = self.ele[max_idx[0]]
        phi_max = self.azi[max_idx[1]]

        Iele = np.argmin(np.abs(self.ele - self.system.ele0))
        Iazi = np.argmin(np.abs(self.azi - self.system.azi0))

        SL_maxpointing = G_boresight - FF_I_dB[max_idx]
        SL_theta_phi = G_boresight - FF_I_dB[Iele, Iazi]

        return {
            "Yc": Yc, "Zc": Zc, "Ac": Ac, "Yc_m": Yc_m, "Zc_m": Zc_m,
            "Lsub": Lsub, "Ntrans": Ntrans, "c0": c0,
            "FF_norm_dB": FF_norm_dB, "FF_I_dB": FF_I_dB, "KerFF_sub": KerFF_sub,
            "Cm": Cm, "sll_in": sll_in, "sll_out": sll_out,
            "theta_max": theta_max, "phi_max": phi_max,
            "SL_maxpointing": SL_maxpointing, "SL_theta_phi": SL_theta_phi,
            "G_boresight": G_boresight,
        }
