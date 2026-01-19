"""
Funzioni per calcolo reale Side Lobe Level (SLL) da array clustering
Tradotto da MATLAB "Irregular Antenna Clustering Tool"
"""

import numpy as np
from scipy.interpolate import griddata
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class LatticeConfig:
    """Configurazione lattice array"""

    Nz: int  # righe
    Ny: int  # colonne
    dist_z: float  # distanza z [lambda]
    dist_y: float  # distanza y [lambda]
    lattice_type: int = 1  # 1=rectangular, 2=square, 3=triangular


@dataclass
class SystemConfig:
    """Parametri sistema"""

    freq: float  # Hz
    lambda_: float  # wavelength [m]
    beta: float  # wave number
    azi0: float = 0.0  # steering azimuth [deg]
    ele0: float = 0.0  # steering elevation [deg]
    dele: float = 0.5  # angular resolution [deg]
    dazi: float = 0.5  # angular resolution [deg]


@dataclass
class MaskConfig:
    """Parametri maschera SLL"""

    elem: float = 30.0  # half FoV elevation [deg]
    azim: float = 60.0  # half FoV azimuth [deg]
    SLL_level: float = 20.0  # SLL outside FoV [dB]
    SLLin: float = 15.0  # SLL inside FoV [dB]


class AntennaArray:
    """Classe per array di antenne con clustering"""

    def __init__(self, lattice: LatticeConfig, system: SystemConfig, mask: MaskConfig):
        self.lattice = lattice
        self.system = system
        self.mask = mask
        self.Nel = lattice.Nz * lattice.Ny

        # Genera lattice
        self._generate_lattice()

        # Genera coordinate polari
        self._generate_polar_coordinates()

        # Genera pattern elemento
        self._generate_element_pattern()

        # Genera maschera
        self._generate_mask()

    def _generate_lattice(self):
        """Genera coordinate lattice array (come GenerateLattice.m)"""
        Nz, Ny = self.lattice.Nz, self.lattice.Ny
        lambda_ = self.system.lambda_

        # Vettori base
        x1 = np.array([lambda_ * self.lattice.dist_y, 0])
        x2 = np.array([0, lambda_ * self.lattice.dist_z])

        # Grid di indici
        nn = np.arange(-(Ny - 1) / 2, (Ny - 1) / 2 + 1)
        mm = np.arange(-(Nz - 1) / 2, (Nz - 1) / 2 + 1)
        NN, MM = np.meshgrid(nn, mm)

        # Coordinate fisiche [m]
        self.Y = NN * x1[0] + MM * x2[0]
        self.Z = NN * x1[1] + MM * x2[1]
        self.NN = NN
        self.MM = MM

        # Dimensioni array
        self.Dy = np.max(self.Y) - np.min(self.Y)
        self.Dz = np.max(self.Z) - np.min(self.Z)

    def _generate_polar_coordinates(self):
        """Genera coordinate polari per sampling (come PolarCoordinate_SteeringAngle)"""
        beta = self.system.beta

        # Sampling angolare per plot
        self.ele = np.arange(-90, 90 + self.system.dele, self.system.dele)
        self.azi = np.arange(-90, 90 + self.system.dazi, self.system.dazi)
        AZI, ELE = np.meshgrid(self.azi, self.ele)

        # Coordinate spettrali visibili
        self.WWae = beta * np.cos(np.deg2rad(90 - ELE))
        self.Wvae = beta * np.sin(np.deg2rad(90 - ELE)) * np.sin(np.deg2rad(AZI))

        # Nyquist spectral sampling (per ottimizzazione)
        chi = 2  # sampling factor
        Nw = int(np.floor(chi * 4 * self.Dz / self.system.lambda_))
        Nv = int(np.floor(chi * 4 * self.Dy / self.system.lambda_))

        ww = np.linspace(0, beta, Nw + 1)
        ww = np.concatenate([-np.flip(ww[1:]), ww])
        vv = np.linspace(0, beta, Nv + 1)
        vv = np.concatenate([-np.flip(vv[1:]), vv])

        WW, VV = np.meshgrid(ww, vv)

        # Coordinate equivalenti ele/azi
        ELEi = 90 - np.rad2deg(np.arccos(np.clip(WW / beta, -1, 1)))
        with np.errstate(invalid="ignore"):
            AZIi = np.real(
                np.rad2deg(
                    np.arcsin(
                        np.clip(VV / (beta * np.sin(np.deg2rad(90 - ELEi))), -1, 1)
                    )
                )
            )
        AZIi[Nv, 1:-1] = 0
        AZIi[Nv, [0, -1]] = 90

        self.Nw, self.Nv = Nw, Nv
        self.WW, self.VV = WW, VV
        self.ww, self.vv = ww, vv
        self.ELEi, self.AZIi = ELEi, AZIi
        self.AZI, self.ELE = AZI, ELE

    def _generate_element_pattern(self, P: int = 1, Gel: float = 5.0):
        """
        Genera pattern elemento singolo (come ElementPattern_v2d0.m)
        P=0: isotropico, P=1: coseno
        """
        # Pattern su griglia ANGOLARE (per plot)
        if P == 0:
            Fel_angular = np.ones_like(self.WWae)
        elif P == 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                Fel_angular = np.maximum(0, np.cos(np.deg2rad(self.ELE))) ** P
                Fel_angular = np.nan_to_num(Fel_angular)

        # RPE in dB
        RPE = 20 * np.log10(np.abs(Fel_angular) + 1e-10)
        RPE = RPE - np.max(RPE) + Gel

        # Pattern su griglia SPETTRALE (per calcolo array factor)
        if P == 0:
            Fel_VW = np.ones_like(self.WW)
        elif P == 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                Fel_VW = np.maximum(0, np.cos(np.deg2rad(self.ELEi))) ** P
                Fel_VW = np.nan_to_num(Fel_VW)

        self.Fel_angular = Fel_angular  # per plot
        self.Fel_VW = Fel_VW  # per calcolo array factor
        self.RPE = RPE
        self.RPE_ele_max = np.max(RPE)
        self.G_boresight = self.RPE_ele_max + 10 * np.log10(self.Nel)

    def _generate_mask(self):
        """Genera maschera SLL (come mask_design_v2d0.m)"""
        # Trova indici in/out FoV
        ele_cond = np.abs(self.ELE - self.system.ele0) <= self.mask.elem
        azi_cond = np.abs(self.AZI - self.system.azi0) <= self.mask.azim

        in_fov = ele_cond & azi_cond
        out_fov = ~in_fov

        # Maschera: G_boresight - SLL_level
        Mask_EA = np.full_like(self.ELE, self.G_boresight - self.mask.SLL_level)
        Mask_EA[in_fov] = self.G_boresight - self.mask.SLLin

        self.Mask_EA = Mask_EA
        self.in_fov_mask = in_fov
        self.out_fov_mask = out_fov

    def cluster_to_positions(
        self,
        cluster_genes: np.ndarray,
        cluster_type: np.ndarray = np.array([[0, 0], [0, 1]]),
    ) -> Dict:
        """
        Converte genes cluster in posizioni fisiche
        cluster_genes: array binario (1=cluster attivo, 0=inattivo)
        cluster_type: [[0,0], [0,1]] = vertical 2x1
        """
        # Genera tutti i possibili cluster positions
        possible_clusters = self._generate_all_clusters(cluster_type)

        # Seleziona solo quelli attivi
        active_indices = np.where(cluster_genes == 1)[0]
        if len(active_indices) == 0:
            return None

        # Limita al numero di cluster genes
        active_indices = active_indices[
            : min(len(active_indices), len(possible_clusters))
        ]

        clusters = [possible_clusters[i] for i in active_indices]

        # Calcola posizioni e phase center
        Yc_all = []
        Zc_all = []
        Lsub = []

        for cluster in clusters:
            # Coordinate elementi nel cluster
            y_coords = self.Y.flatten()[cluster]
            z_coords = self.Z.flatten()[cluster]

            Yc_all.append(y_coords)
            Zc_all.append(z_coords)
            Lsub.append(len(cluster))

        # Phase centers (media coordinate)
        Yc_m = np.array([np.mean(yc) for yc in Yc_all])
        Zc_m = np.array([np.mean(zc) for zc in Zc_all])

        return {
            "clusters": clusters,
            "Yc": Yc_all,
            "Zc": Zc_all,
            "Yc_m": Yc_m,
            "Zc_m": Zc_m,
            "Lsub": np.array(Lsub),
            "Ntrans": len(clusters),
        }

    def _generate_all_clusters(self, cluster_type: np.ndarray) -> List[np.ndarray]:
        """Genera tutte le posizioni possibili per un dato tipo di cluster"""
        clusters = []

        # Per cluster 2x1 verticale: [[0,0], [0,1]]
        if np.array_equal(cluster_type, np.array([[0, 0], [0, 1]])):
            # Cluster verticali di 2 elementi
            for j in range(self.lattice.Ny):
                for i in range(0, self.lattice.Nz - 1, 2):
                    idx1 = i * self.lattice.Ny + j
                    idx2 = (i + 1) * self.lattice.Ny + j
                    clusters.append(np.array([idx1, idx2]))

        return clusters

    def calculate_beamforming_weights(self, cluster_info: Dict) -> np.ndarray:
        """
        Calcola coefficienti beamforming (come coefficient_evaluation.m)
        """
        beta = self.system.beta
        azi0_rad = np.deg2rad(self.system.azi0)
        ele0_rad = np.deg2rad(self.system.ele0)

        # Vettore d'onda direzione steering
        v0 = beta * np.sin(np.pi / 2 - ele0_rad) * np.sin(azi0_rad)
        w0 = beta * np.cos(np.pi / 2 - ele0_rad)

        # Coefficienti: phase shift per puntare in direzione (azi0, ele0)
        Yc_m = cluster_info["Yc_m"]
        Zc_m = cluster_info["Zc_m"]
        Lsub = cluster_info["Lsub"]

        # Amplitude: normalizzata per cluster size
        Amplit = np.sqrt(1.0 / Lsub)

        # Phase: steering phase
        Phase = np.exp(-1j * (w0 * Zc_m + v0 * Yc_m))

        # Coefficienti complessi
        c0 = Amplit * Phase

        return c0

    def calculate_array_pattern(self, cluster_info: Dict, c0: np.ndarray) -> Dict:
        """
        Calcola pattern array totale (come Kernel1_RPE.m)
        """
        Ntrans = cluster_info["Ntrans"]
        Lsub = cluster_info["Lsub"]

        # Array factor su grid spettrale (WW, VV)
        AF = np.zeros_like(self.WW, dtype=complex)

        for k in range(Ntrans):
            # Coordinate elementi nel k-esimo cluster
            Yc = cluster_info["Yc"][k]
            Zc = cluster_info["Zc"][k]

            # Contributo cluster k
            cluster_af = np.zeros_like(self.WW, dtype=complex)
            for idx in range(len(Yc)):
                phase_shift = np.exp(1j * (self.WW * Zc[idx] + self.VV * Yc[idx]))
                cluster_af += phase_shift

            # Moltiplica per pattern elemento e coefficiente
            AF += c0[k] * cluster_af * self.Fel_VW

        # Pattern normalizzato
        FF_norm = np.abs(AF) ** 2

        # Normalizzazione
        Nel_active = np.sum(Lsub)
        FF_norm = FF_norm / (np.max(FF_norm) + 1e-10) * Nel_active

        # dB
        FF_norm_dB = 10 * np.log10(FF_norm + 1e-10)

        # Riferito a G_boresight
        FF_I_dB_spectral = FF_norm_dB - 10 * np.log10(Nel_active) + self.RPE_ele_max

        # Interpola su griglia angolare per valutazione SLL
        # Flatten grids
        ww_flat = self.WW.flatten()
        vv_flat = self.VV.flatten()
        ff_flat = FF_I_dB_spectral.flatten()

        # Coordinate angolari target
        wwae_flat = self.WWae.flatten()
        wvae_flat = self.Wvae.flatten()

        # Interpola
        FF_I_dB_angular = griddata(
            (ww_flat, vv_flat),
            ff_flat,
            (wwae_flat, wvae_flat),
            method="linear",
            fill_value=-100,
        )
        FF_I_dB_angular = FF_I_dB_angular.reshape(self.WWae.shape)

        return {
            "FF_norm_dB": FF_norm_dB,
            "FF_I_dB": FF_I_dB_angular,  # su griglia angolare per SLL
            "AF": AF,
        }

    def evaluate_sll(self, cluster_genes: np.ndarray) -> Dict:
        """
        Valuta SLL per una configurazione di clustering
        QUESTA È LA FUNZIONE CHIAVE PER IL GA
        """
        # Converti genes in cluster positions
        cluster_info = self.cluster_to_positions(cluster_genes)

        if cluster_info is None:
            return {
                "sll_out": 0.0,
                "sll_in": 0.0,
                "n_clusters": 0,
                "fitness": -1000.0,
                "valid": False,
            }

        # Calcola coefficienti beamforming
        c0 = self.calculate_beamforming_weights(cluster_info)

        # Calcola pattern array
        pattern = self.calculate_array_pattern(cluster_info, c0)
        FF_I_dB = pattern["FF_I_dB"]  # già su griglia angolare

        # Trova massimo
        max_idx = np.unravel_index(np.argmax(FF_I_dB), FF_I_dB.shape)

        # SLL in/out FoV
        sll_out_values = FF_I_dB[self.out_fov_mask]
        sll_in_values = FF_I_dB[self.in_fov_mask]

        sll_out = np.max(sll_out_values) if len(sll_out_values) > 0 else -100
        sll_in = np.max(sll_in_values) if len(sll_in_values) > 0 else -100

        # Scan loss
        G_max = np.max(FF_I_dB)
        scan_loss = self.G_boresight - G_max

        # Fitness: vogliamo SLL basso, pochi cluster
        n_clusters = cluster_info["Ntrans"]
        hardware_penalty = (n_clusters / self.Nel) * 3

        fitness = -abs(sll_out) - hardware_penalty

        return {
            "sll_out": sll_out,
            "sll_in": sll_in,
            "n_clusters": n_clusters,
            "scan_loss": scan_loss,
            "fitness": fitness,
            "valid": True,
            "pattern": FF_I_dB,
        }


def test_antenna_physics():
    """Test rapido"""
    # Configurazione 16x16
    lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.6, dist_y=0.53)

    freq = 29.5e9
    lambda_ = 3e8 / freq
    beta = 2 * np.pi / lambda_
    system = SystemConfig(freq=freq, lambda_=lambda_, beta=beta, azi0=0, ele0=0)

    mask = MaskConfig(elem=30, azim=60, SLL_level=20, SLLin=15)

    # Crea array
    print("Inizializzando array...")
    array = AntennaArray(lattice, system, mask)

    # Test clustering random
    n_possible_clusters = (lattice.Nz // 2) * lattice.Ny
    cluster_genes = np.random.randint(0, 2, size=n_possible_clusters)
    cluster_genes[:64] = 1  # attiva primi 64 cluster

    # Valuta SLL
    print("Valutando SLL...")
    result = array.evaluate_sll(cluster_genes)

    print("\nTest Antenna Physics:")
    print(f"  SLL out: {result['sll_out']:.2f} dB")
    print(f"  SLL in: {result['sll_in']:.2f} dB")
    print(f"  N clusters: {result['n_clusters']}")
    print(f"  Fitness: {result['fitness']:.2f}")
    print(f"  Valid: {result['valid']}")

    return array, result


if __name__ == "__main__":
    array, result = test_antenna_physics()
