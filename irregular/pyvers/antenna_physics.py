"""
Funzioni per calcolo array antenna con clustering - Fedele al MATLAB
Tradotto da MATLAB "Irregular Antenna Clustering Tool"
Versione corretta - 100% fedele al codice MATLAB originale
"""

import numpy as np
from scipy.interpolate import interp2d, griddata
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, field


@dataclass
class LatticeConfig:
    """Configurazione lattice array - come Input_Conf.m"""

    Nz: int  # Number of rows
    Ny: int  # Number of columns
    dist_z: float  # antenna distance on z axis [times lambda]
    dist_y: float  # antenna distance on y axis [times lambda]
    lattice_type: int = 1  # 1=Rectangular, 2=Squared, 3=Triangular Equilateral,
    # 4=Triangular NON-equilateral, 5=Hexagonal


@dataclass
class SystemConfig:
    """Parametri sistema - come Input_Conf.m"""

    freq: float  # [Hz]
    lambda_: float = field(init=False)  # wavelength [m]
    beta: float = field(init=False)  # wave number
    azi0: float = 0.0  # [deg] azimuth steering angle
    ele0: float = 0.0  # [deg] elevation steering angle
    dele: float = 0.5  # angle resolution [deg]
    dazi: float = 0.5  # angle resolution [deg]

    def __post_init__(self):
        self.lambda_ = 3e8 / self.freq
        self.beta = 2 * np.pi / self.lambda_


@dataclass
class MaskConfig:
    """Parametri maschera SLL - come Input_Conf.m"""

    elem: float = 30.0  # [deg] half-FoV width elevation plane
    azim: float = 60.0  # [deg] half-FoV width azimuthal plane
    SLL_level: float = 20.0  # [dB] SLL level outside the FoV
    SLLin: float = 15.0  # [dB] SLL level inside the FoV


@dataclass
class ElementPatternConfig:
    """Configurazione pattern elemento - come Input_Conf.m"""

    P: int = 1  # 0=isotropic, 1=cosine
    Gel: float = 5.0  # Maximum antenna Gain [dB]
    load_file: int = 0  # 0=generate, 1=load from HFSS
    rpe_folder: str = ""
    rpe_file_name: str = "RPE_element.csv"


class AntennaArray:
    """
    Classe per array di antenne con clustering
    Implementazione fedele al MATLAB
    """

    def __init__(
        self,
        lattice: LatticeConfig,
        system: SystemConfig,
        mask: MaskConfig,
        eef_config: Optional[ElementPatternConfig] = None,
    ):
        self.lattice = lattice
        self.system = system
        self.mask = mask
        self.eef_config = eef_config or ElementPatternConfig()
        self.Nel = lattice.Nz * lattice.Ny

        # Genera vettori base lattice
        self._compute_lattice_vectors()

        # Genera lattice (come GenerateLattice.m)
        self._generate_lattice()

        # Genera coordinate polari (come PolarCoordinate_SteeringAngle e Main_code.m)
        self._generate_polar_coordinates()

        # Genera pattern elemento (come ElementPattern.m)
        self._generate_element_pattern()

        # Genera maschera (come mask_design_v2d0.m)
        self._generate_mask()

    def _compute_lattice_vectors(self):
        """Calcola vettori base lattice come in Main_code.m"""
        lambda_ = self.system.lambda_
        dz = self.lattice.dist_z * lambda_
        dy = self.lattice.dist_y * lambda_

        # Vettori base per lattice rettangolare (default)
        # x1 = [dy, 0], x2 = [0, dz]
        self.x1 = np.array([dy, 0.0])
        self.x2 = np.array([0.0, dz])

        # Per lattice triangolare/esagonale, modificare x2
        if self.lattice.lattice_type == 3:  # Triangular Equilateral (45 deg)
            self.x2 = np.array([dy, dz])
        elif self.lattice.lattice_type == 5:  # Hexagonal (30 deg)
            self.x2 = np.array([dy / 2, dz])

    def _generate_lattice(self):
        """
        Genera coordinate lattice array
        FEDELE a GenerateLattice.m righe 42-81
        """
        Nz = self.lattice.Nz
        Ny = self.lattice.Ny

        # Generate array indexes - ESATTAMENTE come MATLAB
        # if (rem(Nz,2)) M=-(Nz-1)/2:(Nz-1)/2; else M=-Nz/2+1:Nz/2; end
        if Nz % 2 == 1:  # dispari
            M = np.arange(-(Nz - 1) / 2, (Nz - 1) / 2 + 1)
        else:  # pari
            M = np.arange(-Nz / 2 + 1, Nz / 2 + 1)

        if Ny % 2 == 1:  # dispari
            N = np.arange(-(Ny - 1) / 2, (Ny - 1) / 2 + 1)
        else:  # pari
            N = np.arange(-Ny / 2 + 1, Ny / 2 + 1)

        # [NN,MM]=meshgrid(N,M);
        self.NN, self.MM = np.meshgrid(N, M)

        # dz=x2(2); dy=x1(1); DELTA=max(x2(1),x1(2));
        dz = self.x2[1]
        dy = self.x1[0]
        DELTA = max(self.x2[0], self.x1[1])

        # Y=NN*dy; Z=MM*dz;
        self.Y = self.NN * dy
        self.Z = self.MM * dz

        # Y(2:2:end,:)=Y(2:2:end,:)+DELTA; % Offset per righe pari (lattice triangolare)
        # In Python: righe con indice 1, 3, 5, ... (0-indexed = righe pari in MATLAB 1-indexed)
        self.Y[1::2, :] = self.Y[1::2, :] + DELTA

        # DZ=(max(Z(:))-min(Z(:))); DY=(max(Y(:))-min(Y(:)));
        self.Dz = np.max(self.Z) - np.min(self.Z)
        self.Dy = np.max(self.Y) - np.min(self.Y)

        # Aggiungi dimensione elemento come in PostProcessing
        # Dy=Dy+x1(1); Dz=Dz+x2(2);
        self.Dy_total = self.Dy + self.x1[0]
        self.Dz_total = self.Dz + self.x2[1]

        # Aperture truncation mask (elliptical) - default: no truncation
        self.ArrayMask = np.ones_like(self.Y)

    def _generate_polar_coordinates(self):
        """
        Genera coordinate polari per sampling
        FEDELE a PostProcessing_singlesolution.m righe 104-128 e Main_code.m righe 63-72
        """
        beta = self.system.beta
        lambda_ = self.system.lambda_

        # AZIMUT AND ELEVATION SAMPLING (for plots)
        # ele=-90:dele:90; azi=-90:dazi:90;
        self.ele = np.arange(-90, 90 + self.system.dele, self.system.dele)
        self.azi = np.arange(-90, 90 + self.system.dazi, self.system.dazi)

        # [AZI,ELE]=meshgrid(azi,ele);
        self.AZI, self.ELE = np.meshgrid(self.azi, self.ele)

        # WWae=beta*cosd(90-ELE); Vvae=beta*sind(90-ELE).*sind(AZI);
        self.WWae = beta * np.cos(np.deg2rad(90 - self.ELE))
        self.Vvae = beta * np.sin(np.deg2rad(90 - self.ELE)) * np.sin(
            np.deg2rad(self.AZI)
        )

        # Per Main_code.m semplice: WW e VV sono gli stessi di WWae e Vvae
        # Ma per PostProcessing usiamo Nyquist spectral sampling

        # Nyquist SPECTRAL SAMPLING FOR OPTIMIZATION
        # chi=2; Nw=floor(chi*4*Dz/lambda); Nv=floor(chi*4*Dy/lambda);
        chi = 2
        self.Nw = int(np.floor(chi * 4 * self.Dz_total / lambda_))
        self.Nv = int(np.floor(chi * 4 * self.Dy_total / lambda_))

        # ww=linspace(0,beta,Nw+1); ww=[-fliplr(ww(2:end)), ww];
        ww = np.linspace(0, beta, self.Nw + 1)
        self.ww = np.concatenate([-np.flip(ww[1:]), ww])

        # vv=linspace(0,beta,Nv+1); vv=[-fliplr(vv(2:end)), vv];
        vv = np.linspace(0, beta, self.Nv + 1)
        self.vv = np.concatenate([-np.flip(vv[1:]), vv])

        # [WW,VV]=meshgrid(ww,vv);
        self.WW, self.VV = np.meshgrid(self.ww, self.vv)

        # ELEi=90-acosd(WW./beta);
        # Clip per evitare valori fuori [-1, 1]
        WW_clipped = np.clip(self.WW / beta, -1, 1)
        self.ELEi = 90 - np.rad2deg(np.arccos(WW_clipped))

        # AZIi=real(asind(VV./(beta*sind(90-ELEi))));
        denom = beta * np.sin(np.deg2rad(90 - self.ELEi))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.clip(self.VV / denom, -1, 1)
            self.AZIi = np.real(np.rad2deg(np.arcsin(ratio)))

        # AZIi(Nv+1,1:2*Nw+1)=0; AZIi(Nv+1,[1,2*Nw+1])=90;
        # In MATLAB indices are 1-based, in Python 0-based
        # Nv+1 in MATLAB -> Nv in Python (middle row)
        self.AZIi[self.Nv, :] = 0
        self.AZIi[self.Nv, 0] = 90
        self.AZIi[self.Nv, -1] = 90

    def _generate_element_pattern(self):
        """
        Genera pattern elemento singolo
        FEDELE a ElementPattern.m riga 38:
        Fel=((cosd(ELE*0.9).*cosd(AZI*0.9)).^P);
        """
        P = self.eef_config.P
        Gel = self.eef_config.Gel

        if self.eef_config.load_file == 1:
            # Carica da file HFSS
            self._load_element_pattern_from_file()
        else:
            # Pattern analitico - ESATTAMENTE come ElementPattern.m riga 38
            # Fel=((cosd(ELE*0.9).*cosd(AZI*0.9)).^P);
            if P == 0:
                self.Fel = np.ones_like(self.ELE)
            else:
                self.Fel = (
                    np.cos(np.deg2rad(self.ELE * 0.9))
                    * np.cos(np.deg2rad(self.AZI * 0.9))
                ) ** P

        # Calcola Fel_VW sulla griglia spettrale (ELEi, AZIi)
        if P == 0:
            self.Fel_VW = np.ones_like(self.ELEi)
        else:
            self.Fel_VW = (
                np.cos(np.deg2rad(self.ELEi * 0.9))
                * np.cos(np.deg2rad(self.AZIi * 0.9))
            ) ** P

        # RPE in dB
        self.RPE = 20 * np.log10(np.abs(self.Fel) + 1e-10)
        self.RPE_ele_max = np.max(self.RPE) + Gel

        # G_boresight per normalizzazione
        self.G_boresight = self.RPE_ele_max + 10 * np.log10(self.Nel)

    def _load_element_pattern_from_file(self):
        """Carica pattern elemento da file HFSS - come ElementPattern.m righe 20-36"""
        import os

        file_path = os.path.join(
            self.eef_config.rpe_folder, self.eef_config.rpe_file_name
        )

        try:
            with open(file_path, "r") as fid:
                lines = fid.readlines()

            angsp = 361
            RPE_elem = []
            for line in lines[1:]:  # skip header
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    RPE_elem.append(float(parts[2]))

            RPE = np.array(RPE_elem).reshape(angsp, angsp).T
            self.Fel = 10 ** (RPE / 20)

        except FileNotFoundError:
            print(f"Warning: RPE file not found at {file_path}, using cosine pattern")
            P = self.eef_config.P
            self.Fel = (
                np.cos(np.deg2rad(self.ELE * 0.9))
                * np.cos(np.deg2rad(self.AZI * 0.9))
            ) ** P

    def _generate_mask(self):
        """
        Genera maschera SLL
        FEDELE a mask_design_v2d0.m
        """
        ele0 = self.system.ele0
        azi0 = self.system.azi0
        elem = self.mask.elem
        azim = self.mask.azim
        SLL_level = self.mask.SLL_level
        SLLin = self.mask.SLLin

        # Trova indici in/out FoV sulla griglia angolare
        # Inside FoV: |ele - ele0| <= elem AND |azi - azi0| <= azim
        ele_cond = np.abs(self.ELE - ele0) <= elem
        azi_cond = np.abs(self.AZI - azi0) <= azim

        self.in_fov_mask = ele_cond & azi_cond
        self.out_fov_mask = ~self.in_fov_mask

        # Indici lineari per in/out FoV
        self.Isll_in = np.where(self.in_fov_mask.flatten())[0]
        self.Isll_out = np.where(self.out_fov_mask.flatten())[0]

        # Mask_EA: maschera in dB sulla griglia angolare
        # Fuori FoV: G_boresight - SLL_level
        # Dentro FoV: G_boresight - SLLin
        self.Mask_EA = np.full_like(self.ELE, self.G_boresight - SLL_level, dtype=float)
        self.Mask_EA[self.in_fov_mask] = self.G_boresight - SLLin

        # Genera anche mask per griglia spettrale
        ele_cond_vw = np.abs(self.ELEi - ele0) <= elem
        azi_cond_vw = np.abs(self.AZIi - azi0) <= azim
        self.in_fov_mask_vw = ele_cond_vw & azi_cond_vw
        self.out_fov_mask_vw = ~self.in_fov_mask_vw

    def generate_subarray_set(
        self, B: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Genera set di subarray
        FEDELE a SubArraySet_Generation.m

        B: matrice Lsub x 2 con posizioni indice del cluster
           es. B=[[0,0],[0,1]] per cluster verticale 2x1
        """
        B = np.atleast_2d(B)

        A = np.sum(B, axis=0)

        M = self.MM.flatten()
        N = self.NN.flatten()

        if A[0] == 0:  # vertical cluster
            step_M = B.shape[0]
            step_N = 1
        elif A[1] == 0:  # horizontal cluster
            step_N = B.shape[0]
            step_M = 1
        else:  # generic cluster
            step_M = 1
            step_N = 1

        S = []

        min_M, max_M = int(np.min(M)), int(np.max(M))
        min_N, max_N = int(np.min(N)), int(np.max(N))

        for kk in range(min_M, max_M + 1, step_M):
            for hh in range(min_N, max_N + 1, step_N):
                Bshift = B.copy()
                Bshift[:, 0] = B[:, 0] + hh
                Bshift[:, 1] = B[:, 1] + kk

                # Check bounds
                check = not np.any(
                    (Bshift[:, 0] > max_N)
                    | (Bshift[:, 0] < min_N)
                    | (Bshift[:, 1] > max_M)
                    | (Bshift[:, 1] < min_M)
                )

                if check:
                    S.append(Bshift)

        Nsub = len(S)
        return S, Nsub

    def index_to_position_cluster(
        self,
        Cluster: List[np.ndarray],
        ElementExc: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converte indici cluster in posizioni fisiche
        FEDELE a Index2Position_cluster_v2d0.m e Main_code.m righe 83-97
        """
        if ElementExc is None:
            ElementExc = np.ones((self.lattice.Nz, self.lattice.Ny))

        Ntrans = len(Cluster)

        # Trova dimensione massima cluster
        max_Lsub = max(c.shape[0] for c in Cluster)

        # Inizializza arrays
        Yc = np.full((max_Lsub, Ntrans), np.nan)
        Zc = np.full((max_Lsub, Ntrans), np.nan)
        Ac = np.full((max_Lsub, Ntrans), np.nan)

        min_NN = int(np.min(self.NN))
        min_MM = int(np.min(self.MM))

        for kk, cluster in enumerate(Cluster):
            for l1 in range(cluster.shape[0]):
                # Iy=Cluster(l1,2*kk+1)-min(NN(:))+1;
                # Iz=Cluster(l1,2*kk+2)-min(MM(:))+1;
                Iy = int(cluster[l1, 0] - min_NN)
                Iz = int(cluster[l1, 1] - min_MM)

                # Yc(l1,kk+1)=Y(Iz,Iy); Zc(l1,kk+1)=Z(Iz,Iy);
                Yc[l1, kk] = self.Y[Iz, Iy]
                Zc[l1, kk] = self.Z[Iz, Iy]
                Ac[l1, kk] = ElementExc[Iz, Iy]

        return Yc, Zc, Ac

    def coefficient_evaluation(
        self,
        Zc_m: np.ndarray,
        Yc_m: np.ndarray,
        Lsub: np.ndarray,
    ) -> np.ndarray:
        """
        Calcola coefficienti beamforming
        FEDELE a Main_code.m righe 106-110:

        v0=beta*sind(90-ele0)*sind(azi0);
        w0=beta*cosd(90-ele0);
        Phase_m=exp(-1i*(w0*Zc_m+v0*Yc_m));
        Amplit_m = ones(1,Ntrans)./Lsub;
        c0=Amplit_m.*Phase_m;
        """
        beta = self.system.beta
        ele0 = self.system.ele0
        azi0 = self.system.azi0

        # v0=beta*sind(90-ele0)*sind(azi0);
        v0 = beta * np.sin(np.deg2rad(90 - ele0)) * np.sin(np.deg2rad(azi0))

        # w0=beta*cosd(90-ele0);
        w0 = beta * np.cos(np.deg2rad(90 - ele0))

        # Phase_m=exp(-1i*(w0*Zc_m+v0*Yc_m));
        Phase_m = np.exp(-1j * (w0 * Zc_m + v0 * Yc_m))

        # Amplit_m = ones(1,Ntrans)./Lsub;
        Amplit_m = 1.0 / Lsub

        # c0=Amplit_m.*Phase_m;
        c0 = Amplit_m * Phase_m

        return c0

    def kernel1_rpe(
        self,
        Lsub: np.ndarray,
        Ac: np.ndarray,
        Yc: np.ndarray,
        Zc: np.ndarray,
        c0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcola pattern array tramite kernel
        FEDELE a Kernel1_RPE e Main_code.m righe 129-140:

        KerFF_sub=zeros(Nw*Nv,Ntrans);
        for kk=1:Ntrans
            for jj=1:Lsub(kk)
                KerFF_sub(:,kk)=KerFF_sub(:,kk)+exp(1i*(VV(:)*Yc(jj,kk)+WW(:)*Zc(jj,kk))).*Fel_VW(:);
            end
        end
        FF=KerFF_sub*c0.';
        FF_norm=FF./max((FF(:)));
        """
        Ntrans = len(Lsub)

        # OPT: Cache flattened arrays as class attributes to avoid repeated flattening
        if not hasattr(self, '_VV_flat'):
            self._VV_flat = self.VV.flatten()
            self._WW_flat = self.WW.flatten()
            self._Fel_VW_flat = self.Fel_VW.flatten()

        VV_flat = self._VV_flat
        WW_flat = self._WW_flat
        Fel_VW_flat = self._Fel_VW_flat

        Npoints = len(VV_flat)

        # OPT: Vectorized kernel computation - replaces nested loops with matrix operations
        # Build all valid (Y, Z) coordinates into arrays for vectorized exp computation
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

        # OPT: Convert to numpy arrays for vectorized computation
        all_Y = np.array(all_Y)
        all_Z = np.array(all_Z)
        cluster_indices = np.array(cluster_indices)

        # OPT: Compute all phases at once using broadcasting: (Npoints, N_elements)
        # phase[i,j] = exp(1j * (VV[i] * Y[j] + WW[i] * Z[j]))
        phases = np.exp(1j * (np.outer(VV_flat, all_Y) + np.outer(WW_flat, all_Z)))

        # OPT: Multiply by element pattern (broadcasting along columns)
        phases = phases * Fel_VW_flat[:, np.newaxis]

        # OPT: Sum contributions to each cluster using np.add.at for efficiency
        KerFF_sub = np.zeros((Npoints, Ntrans), dtype=complex)
        np.add.at(KerFF_sub.T, cluster_indices, phases.T)
        KerFF_sub = KerFF_sub  # OPT: Transpose handled implicitly by add.at on transposed array

        # FF=KerFF_sub*c0.';
        FF = KerFF_sub @ c0.T

        # FF_norm=FF./max((FF(:)));
        FF_norm = FF / (np.max(np.abs(FF)) + 1e-10)

        # Reshape to 2D
        # FF_norm_2D=reshape(FF_norm,Nv,Nw); (MATLAB shape)
        # In numpy, VV ha shape (2*Nv+1, 2*Nw+1), quindi:
        FF_norm_2D = FF_norm.reshape(self.VV.shape)

        # Fopt_dB=20*log10(abs(FF_norm_2D));
        FF_norm_dB = 20 * np.log10(np.abs(FF_norm_2D) + 1e-10)

        # OPT: Cache interpolation setup - avoid recomputing static grid points
        if not hasattr(self, '_interp_points'):
            self._interp_points = np.column_stack([WW_flat, VV_flat])
            self._interp_xi = np.column_stack([self.WWae.flatten(), self.Vvae.flatten()])

        # Interpola su griglia angolare per valutazione finale
        values = FF_norm_dB.flatten()
        FF_I_dB_flat = griddata(self._interp_points, values, self._interp_xi, method="linear", fill_value=-100)
        FF_I_dB = FF_I_dB_flat.reshape(self.WWae.shape)

        # Normalizza rispetto a G_boresight
        # FF_I_dB già è normalizzato, ma dobbiamo aggiungere il guadagno
        # G_boresight=max(max(RPE))+10*log10(Nel);
        Nel_active = np.sum(Lsub)
        FF_I_dB = FF_I_dB + self.RPE_ele_max + 10 * np.log10(Nel_active)

        return FF_norm_dB, FF_I_dB, KerFF_sub, np.abs(FF_norm_2D) ** 2

    def compute_sll(
        self, FF_I_dB: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calcola SLL in/out FoV
        FEDELE a SLL_in_out function
        """
        # SLL fuori FoV
        sll_out_values = FF_I_dB[self.out_fov_mask]
        sll_out = np.max(sll_out_values) if len(sll_out_values) > 0 else -100

        # SLL dentro FoV (escludendo il main beam)
        sll_in_values = FF_I_dB[self.in_fov_mask]
        # Trova il massimo (main beam) ed escludilo
        if len(sll_in_values) > 0:
            max_val = np.max(sll_in_values)
            # SLL è il secondo massimo o il max delle regioni laterali
            sll_in = np.max(sll_in_values[sll_in_values < max_val - 0.1]) if np.any(sll_in_values < max_val - 0.1) else max_val
        else:
            sll_in = -100

        return sll_in, sll_out

    def compute_cost_function(
        self, FF_I_dB: np.ndarray
    ) -> int:
        """
        Calcola cost function come in MATLAB
        FEDELE a Generation_code.m:
        Constr=FF_I_dB-MASK.Mask_EA;
        Cm=sum(sum(Constr>0));
        """
        Constr = FF_I_dB - self.Mask_EA
        Cm = np.sum(Constr > 0)
        return int(Cm)

    def evaluate_clustering(
        self,
        Cluster: List[np.ndarray],
        ElementExc: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Valuta una configurazione di clustering completa
        Combina tutte le funzioni per calcolare il pattern e le metriche
        """
        if ElementExc is None:
            ElementExc = np.ones((self.lattice.Nz, self.lattice.Ny))

        # Converti indici in posizioni
        Yc, Zc, Ac = self.index_to_position_cluster(Cluster, ElementExc)

        Ntrans = len(Cluster)

        # Calcola Lsub, Zc_m, Yc_m
        Lsub = np.array([c.shape[0] for c in Cluster])
        Zc_m = np.array([np.nanmean(Zc[:Lsub[k], k]) for k in range(Ntrans)])
        Yc_m = np.array([np.nanmean(Yc[:Lsub[k], k]) for k in range(Ntrans)])

        # Calcola coefficienti
        c0 = self.coefficient_evaluation(Zc_m, Yc_m, Lsub)

        # Calcola pattern
        FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm = self.kernel1_rpe(
            Lsub, Ac, Yc, Zc, c0
        )

        # Calcola cost function
        Cm = self.compute_cost_function(FF_I_dB)

        # Calcola SLL
        sll_in, sll_out = self.compute_sll(FF_I_dB)

        # Trova massimo
        max_idx = np.unravel_index(np.argmax(FF_I_dB), FF_I_dB.shape)
        theta_max = self.ele[max_idx[0]]
        phi_max = self.azi[max_idx[1]]

        # Indici steering angle
        Iele = np.argmin(np.abs(self.ele - self.system.ele0))
        Iazi = np.argmin(np.abs(self.azi - self.system.azi0))

        # Scan loss
        G_boresight = self.RPE_ele_max + 10 * np.log10(np.sum(Lsub))
        SL_maxpointing = G_boresight - FF_I_dB[max_idx]
        SL_theta_phi = G_boresight - FF_I_dB[Iele, Iazi]

        return {
            "Yc": Yc,
            "Zc": Zc,
            "Ac": Ac,
            "Yc_m": Yc_m,
            "Zc_m": Zc_m,
            "Lsub": Lsub,
            "Ntrans": Ntrans,
            "c0": c0,
            "FF_norm_dB": FF_norm_dB,
            "FF_I_dB": FF_I_dB,
            "KerFF_sub": KerFF_sub,
            "Cm": Cm,
            "sll_in": sll_in,
            "sll_out": sll_out,
            "theta_max": theta_max,
            "phi_max": phi_max,
            "SL_maxpointing": SL_maxpointing,
            "SL_theta_phi": SL_theta_phi,
            "G_boresight": G_boresight,
        }


def test_antenna_physics():
    """Test rapido - confronto con MATLAB"""
    # Configurazione come Main_code.m
    lattice = LatticeConfig(Nz=8, Ny=8, dist_z=0.7, dist_y=0.5, lattice_type=1)
    system = SystemConfig(freq=29e9, azi0=0, ele0=10)
    mask = MaskConfig(elem=30, azim=60, SLL_level=20, SLLin=15)
    eef = ElementPatternConfig(P=1, Gel=5, load_file=0)

    # Crea array
    print("Inizializzando array...")
    array = AntennaArray(lattice, system, mask, eef)

    # Test clustering: B=[0,0;0,1] vertical 2x1
    B = np.array([[0, 0], [0, 1]])
    S, Nsub = array.generate_subarray_set(B)
    print(f"Generati {Nsub} cluster di tipo 2x1 verticale")

    # Valuta clustering
    print("Valutando clustering...")
    result = array.evaluate_clustering(S)

    print("\n=== Test Antenna Physics (fedele a MATLAB) ===")
    print(f"  Ntrans (num cluster): {result['Ntrans']}")
    print(f"  Nel totali: {np.sum(result['Lsub'])}")
    print(f"  Cost function Cm: {result['Cm']}")
    print(f"  SLL out FoV: {result['sll_out']:.2f} dB")
    print(f"  SLL in FoV: {result['sll_in']:.2f} dB")
    print(f"  Max pointing: theta={result['theta_max']:.1f}, phi={result['phi_max']:.1f}")
    print(f"  Scan loss max: {result['SL_maxpointing']:.2f} dB")
    print(f"  Scan loss steering: {result['SL_theta_phi']:.2f} dB")
    print(f"  G_boresight: {result['G_boresight']:.2f} dBi")

    return array, result


if __name__ == "__main__":
    array, result = test_antenna_physics()
