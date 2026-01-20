"""
Funzioni di plotting per risultati clustering antenna
Fedele ai plot MATLAB in PostProcessing_singlesolution.m
Con analisi lobi (SLL, HPBW) come in regular/opt/grid_search
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from typing import Dict, List, Optional, Tuple

from antenna_physics import AntennaArray


# ============================================================================
# LOBE ANALYSIS FUNCTIONS (come in regular/opt/grid_search/regular_arrays.py)
# ============================================================================

def extract_lobe_metrics(FF_I_dB: np.ndarray, azi: np.ndarray, ele: np.ndarray,
                         azi0: float, ele0: float, G_boresight: float = None) -> Dict:
    """
    Extract lobe performance metrics from radiation pattern
    Similar to regular/opt/grid_search/regular_arrays.py extract_metrics
    
    INPUT:
    FF_I_dB: 2D radiation pattern [dB] shape (Nele, Nazi)
    azi: azimuth angles [deg]
    ele: elevation angles [deg]
    azi0, ele0: steering angles [deg]
    G_boresight: boresight gain [dB] (optional, computed if not provided)

    OUTPUT:
    metrics: dict with SLL, HPBW_ele, HPBW_azi, main_lobe_gain, etc.
    """
    # Find steering angle indices
    Iele = np.argmin(np.abs(ele - ele0))
    Iazi = np.argmin(np.abs(azi - azi0))

    # 1. GAIN: maximum of the pattern (dB)
    main_lobe_gain = np.max(FF_I_dB)
    if G_boresight is None:
        G_boresight = main_lobe_gain

    # 2. SLL: Side Lobe Level - Elevation cut
    ele_cut = FF_I_dB[:, Iazi]
    
    # Find main lobe region (within ±10 deg from steering angle)
    main_lobe_mask = np.abs(ele - ele0) <= 10
    ele_cut_sidelobe = ele_cut.copy()
    ele_cut_sidelobe[main_lobe_mask] = -np.inf
    
    sll_ele_dB = np.max(ele_cut_sidelobe) if np.any(~main_lobe_mask) else -100
    sll_ele_relative = sll_ele_dB - G_boresight  # Relative SLL (NEGATIVE)

    # 3. SLL: Side Lobe Level - Azimuth cut
    azi_cut = FF_I_dB[Iele, :]
    main_lobe_mask_azi = np.abs(azi - azi0) <= 10
    azi_cut_sidelobe = azi_cut.copy()
    azi_cut_sidelobe[main_lobe_mask_azi] = -np.inf
    
    sll_azi_dB = np.max(azi_cut_sidelobe) if np.any(~main_lobe_mask_azi) else -100
    sll_azi_relative = sll_azi_dB - G_boresight

    # 4. HPBW (Half Power Beam Width) - elevation plane
    half_power_dB = main_lobe_gain - 3
    above_halfpower = ele_cut >= half_power_dB

    if np.any(above_halfpower):
        edges = np.where(np.diff(above_halfpower.astype(int)) != 0)[0]
        if len(edges) >= 2:
            hpbw_ele = ele[edges[-1]] - ele[edges[0]]
        else:
            hpbw_ele = np.nan
    else:
        hpbw_ele = np.nan

    # 5. HPBW - azimuth plane
    above_halfpower_azi = azi_cut >= half_power_dB
    if np.any(above_halfpower_azi):
        edges_azi = np.where(np.diff(above_halfpower_azi.astype(int)) != 0)[0]
        if len(edges_azi) >= 2:
            hpbw_azi = azi[edges_azi[-1]] - azi[edges_azi[0]]
        else:
            hpbw_azi = np.nan
    else:
        hpbw_azi = np.nan

    # 6. Find all local maxima (lobes) in elevation cut
    from scipy.signal import find_peaks
    peaks_ele, props_ele = find_peaks(ele_cut, height=-50, distance=5)
    peaks_azi, props_azi = find_peaks(azi_cut, height=-50, distance=5)

    return {
        "main_lobe_gain": main_lobe_gain,
        "G_boresight": G_boresight,
        "sll_ele_dB": sll_ele_dB,
        "sll_ele_relative": sll_ele_relative,
        "sll_azi_dB": sll_azi_dB,
        "sll_azi_relative": sll_azi_relative,
        "hpbw_ele": hpbw_ele,
        "hpbw_azi": hpbw_azi,
        "n_lobes_ele": len(peaks_ele),
        "n_lobes_azi": len(peaks_azi),
        "peaks_ele_idx": peaks_ele,
        "peaks_azi_idx": peaks_azi,
        "ele_cut": ele_cut,
        "azi_cut": azi_cut,
    }


def plot_lobe_analysis(array: AntennaArray, FF_I_dB: np.ndarray, 
                       G_boresight: float = None,
                       title: str = "Lobe Analysis",
                       save_path: Optional[str] = None):
    """
    Plot lobe analysis similar to regular folder analysis.py
    Shows SLL, HPBW, main lobe and side lobes clearly
    """
    ele0 = array.system.ele0
    azi0 = array.system.azi0
    ele = array.ele
    azi = array.azi
    
    metrics = extract_lobe_metrics(FF_I_dB, azi, ele, azi0, ele0, G_boresight)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Elevation Cut with Lobes
    ax1 = fig.add_subplot(gs[0, 0])
    ele_cut = metrics['ele_cut']
    ax1.plot(ele, ele_cut, 'b-', linewidth=2, label='Pattern')
    ax1.axhline(y=metrics['main_lobe_gain'] - 3, color='orange', linestyle='--', 
                linewidth=1.5, label=f'-3dB ({metrics["main_lobe_gain"]-3:.1f})')
    
    # Mark peaks
    for pk_idx in metrics['peaks_ele_idx']:
        ax1.plot(ele[pk_idx], ele_cut[pk_idx], 'ro', markersize=8)
    
    ax1.axvline(x=ele0, color='g', linestyle=':', alpha=0.7, label=f'Steering θ={ele0}°')
    ax1.set_xlabel('Elevation θ [deg]', fontweight='bold')
    ax1.set_ylabel('Gain [dBi]', fontweight='bold')
    ax1.set_title(f'Elevation Cut (φ={azi0}°)\nHPBW={metrics["hpbw_ele"]:.1f}°, SLL={metrics["sll_ele_relative"]:.1f}dB', 
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-90, 90])
    
    # 2. Azimuth Cut with Lobes
    ax2 = fig.add_subplot(gs[0, 1])
    azi_cut = metrics['azi_cut']
    ax2.plot(azi, azi_cut, 'r-', linewidth=2, label='Pattern')
    ax2.axhline(y=metrics['main_lobe_gain'] - 3, color='orange', linestyle='--', 
                linewidth=1.5, label=f'-3dB')
    
    for pk_idx in metrics['peaks_azi_idx']:
        ax2.plot(azi[pk_idx], azi_cut[pk_idx], 'bo', markersize=8)
    
    ax2.axvline(x=azi0, color='g', linestyle=':', alpha=0.7, label=f'Steering φ={azi0}°')
    ax2.set_xlabel('Azimuth φ [deg]', fontweight='bold')
    ax2.set_ylabel('Gain [dBi]', fontweight='bold')
    ax2.set_title(f'Azimuth Cut (θ={ele0}°)\nHPBW={metrics["hpbw_azi"]:.1f}°, SLL={metrics["sll_azi_relative"]:.1f}dB', 
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-90, 90])
    
    # 3. 2D Contour Pattern
    ax3 = fig.add_subplot(gs[0, 2])
    F_plot = FF_I_dB.copy()
    F_plot[FF_I_dB < -30] = -30
    cf = ax3.contourf(azi, ele, F_plot, levels=20, cmap='jet')
    cbar = plt.colorbar(cf, ax=ax3)
    cbar.set_label('Gain [dBi]')
    ax3.plot(azi0, ele0, 'w*', markersize=15, markeredgecolor='black', label='Steering')
    ax3.set_xlabel('Azimuth φ [deg]', fontweight='bold')
    ax3.set_ylabel('Elevation θ [deg]', fontweight='bold')
    ax3.set_title('2D Radiation Pattern', fontweight='bold')
    ax3.legend()
    
    # 4. Summary Box
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    summary_text = f"""
    ══════════════════════════════
           LOBE ANALYSIS
    ══════════════════════════════
    
    Main Lobe Gain:    {metrics['main_lobe_gain']:.2f} dBi
    G_boresight:       {metrics['G_boresight']:.2f} dBi
    
    ── ELEVATION PLANE ──
    SLL (relative):    {metrics['sll_ele_relative']:.2f} dB
    HPBW:              {metrics['hpbw_ele']:.1f}°
    N° Side Lobes:     {metrics['n_lobes_ele']}
    
    ── AZIMUTH PLANE ──
    SLL (relative):    {metrics['sll_azi_relative']:.2f} dB  
    HPBW:              {metrics['hpbw_azi']:.1f}°
    N° Side Lobes:     {metrics['n_lobes_azi']}
    
    Steering: θ={ele0}°, φ={azi0}°
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # 5. SLL Comparison Bar Chart
    ax5 = fig.add_subplot(gs[1, 1])
    sll_data = [metrics['sll_ele_relative'], metrics['sll_azi_relative']]
    bars = ax5.bar(['Elevation', 'Azimuth'], sll_data, color=['steelblue', 'coral'])
    ax5.axhline(y=-20, color='red', linestyle='--', label='Target SLL=-20dB')
    ax5.axhline(y=-15, color='orange', linestyle='--', label='Target SLL=-15dB')
    ax5.set_ylabel('SLL [dB] (relative)', fontweight='bold')
    ax5.set_title('Side Lobe Level Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, sll_data):
        ax5.text(bar.get_x() + bar.get_width()/2, val - 1, f'{val:.1f}dB', 
                ha='center', va='top', fontweight='bold', color='white')
    
    # 6. HPBW Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    hpbw_data = [metrics['hpbw_ele'], metrics['hpbw_azi']]
    bars = ax6.bar(['Elevation', 'Azimuth'], hpbw_data, color=['steelblue', 'coral'])
    ax6.set_ylabel('HPBW [deg]', fontweight='bold')
    ax6.set_title('Half Power Beam Width', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, hpbw_data):
        if not np.isnan(val):
            ax6.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}°', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato in {save_path}")
    
    plt.show()
    
    return metrics


def plot_subarray_map(
    Yc: np.ndarray,
    Zc: np.ndarray,
    Yc_m: np.ndarray,
    Zc_m: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot mappa subarray - come PostProcessing_singlesolution.m righe 251-276

    figure
    for ih=1:size(Yc,2)
        RGBcolor=rand(1,3);
        subplot(1,2,1)
        plot(Yc(:,ih),Zc(:,ih),'sq','MarkerEdgeColor',RGBcolor,...);
        subplot(1,2,2)
        plot(Yc_m(ih),Zc_m(ih),'sq','MarkerEdgeColor',RGBcolor,...);
    end
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    Ntrans = Yc.shape[1]

    for ih in range(Ntrans):
        RGBcolor = np.random.rand(3)

        # Plot elementi cluster
        yc_col = Yc[:, ih]
        zc_col = Zc[:, ih]
        valid = ~np.isnan(yc_col)

        ax1.scatter(
            yc_col[valid],
            zc_col[valid],
            s=60,
            c=[RGBcolor],
            marker="s",
            edgecolors=RGBcolor,
        )

        # Plot phase center
        ax2.scatter(
            Yc_m[ih],
            Zc_m[ih],
            s=60,
            c=[RGBcolor],
            marker="s",
            edgecolors=RGBcolor,
        )

    ax1.set_xlabel("y [m]")
    ax1.set_ylabel("z [m]")
    ax1.set_title("Antenna Sub-arrays Cluster Element")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("z [m]")
    ax2.set_title("Antenna Sub-arrays Phase Center")
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def plot_phase_centers(
    Yc_m: np.ndarray,
    Zc_m: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot solo phase centers in nero - come PostProcessing_singlesolution.m righe 278-287
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        Yc_m,
        Zc_m,
        s=60,
        c="black",
        marker="s",
        edgecolors="black",
    )

    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Antenna Sub-arrays Phase Center")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_rpe_cuts(
    array: AntennaArray,
    FF_I_dB: np.ndarray,
    ele0: float,
    azi0: float,
    label_line: str = "Irregular clustering",
    color_line: str = "r",
    save_path: Optional[str] = None,
):
    """
    Plot tagli RPE nei piani cardinali - come PostProcessing_singlesolution.m righe 335-358

    figure
    subplot(1,2,1)
    plot(ele,FF_I_dB(:,Iazi),color_line,'Linewidth',2)
    plot(ele,(Mask_EA(:,Iazi)),'g','Linewidth',2)
    subplot(1,2,2)
    plot(azi,FF_I_dB(Iele,:),color_line,'Linewidth',2)
    plot(azi,(Mask_EA(Iele,:)),'g','Linewidth',2)
    """
    ele = array.ele
    azi = array.azi
    Mask_EA = array.Mask_EA

    Iele = np.argmin(np.abs(ele - ele0))
    Iazi = np.argmin(np.abs(azi - azi0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Piano verticale (elevation)
    ax1.plot(ele, FF_I_dB[:, Iazi], color_line, linewidth=2, label=f"{label_line} phi={azi0} [deg]")
    ax1.plot(ele, Mask_EA[:, Iazi], "g", linewidth=2, label="Mask")
    ax1.set_xlim([-90, 90])
    ax1.set_ylim([-30, np.max(Mask_EA[:, Iazi]) + 0.5])
    ax1.set_xlabel(r"$\theta$ [deg]")
    ax1.set_ylabel(r"RPE R($\theta$,$\phi$) [dB]")
    ax1.set_title("Vertical plane")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Piano orizzontale (azimuth)
    ax2.plot(azi, FF_I_dB[Iele, :], color_line, linewidth=2, label=f"{label_line} theta={ele0} [deg]")
    ax2.plot(azi, Mask_EA[Iele, :], "g", linewidth=2, label="Mask")
    ax2.set_xlim([-90, 90])
    ax2.set_ylim([-30, np.max(Mask_EA[Iele, :]) + 0.5])
    ax2.set_xlabel(r"$\phi$ [deg]")
    ax2.set_ylabel(r"RPE R($\theta$,$\phi$) [dB]")
    ax2.set_title("Horizontal plane")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def plot_rpe_contour(
    array: AntennaArray,
    FF_I_dB: np.ndarray,
    ele0: float,
    azi0: float,
    label_line: str = "Irregular clustering",
    save_path: Optional[str] = None,
):
    """
    Plot contour 2D del radiation pattern - come PostProcessing_singlesolution.m righe 360-368

    F_plot=FF_I_dB;
    F_plot(FF_I_dB<-30)=-30;
    figure
    contourf(azi, ele, F_plot)
    """
    ele = array.ele
    azi = array.azi

    # Clip values below -30 dB
    F_plot = FF_I_dB.copy()
    F_plot[FF_I_dB < -30] = -30

    fig, ax = plt.subplots(figsize=(10, 8))

    cf = ax.contourf(azi, ele, F_plot, levels=30, cmap="jet")
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Realized Gain [dBi]")

    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_title(f"{label_line} - Radiation Pattern [theta={ele0}, phi={azi0}]")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def plot_rpe_surface(
    array: AntennaArray,
    FF_I_dB: np.ndarray,
    ele0: float,
    azi0: float,
    label_line: str = "Irregular clustering",
    save_path: Optional[str] = None,
):
    """
    Plot 3D surface del radiation pattern - come PostProcessing_singlesolution.m righe 376-382

    figure
    surf(azi(1:4:361), ele(1:4:361), F_plot(1:4:361,1:4:361))
    """
    ele = array.ele
    azi = array.azi

    # Clip values below -30 dB
    F_plot = FF_I_dB.copy()
    F_plot[FF_I_dB < -30] = -30

    # Subsample for visualization (ogni 4 punti)
    step = 4
    azi_sub = azi[::step]
    ele_sub = ele[::step]
    F_sub = F_plot[::step, ::step]

    AZI_sub, ELE_sub = np.meshgrid(azi_sub, ele_sub)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(AZI_sub, ELE_sub, F_sub, cmap="jet", alpha=0.8)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5)
    cbar.set_label("Realized Gain [dBi]")

    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_zlabel("Gain [dBi]")
    ax.set_title(f"{label_line} - Radiation Pattern [theta={ele0}, phi={azi0}]")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def plot_full_analysis(
    array: AntennaArray,
    result: Dict,
    label_line: str = "Irregular clustering",
    save_path: Optional[str] = None,
):
    """
    Plot completo di analisi - combina subarray map, RPE cuts, e statistiche
    Come figura composita in PostProcessing_singlesolution.m righe 289-333
    """
    Yc = result["Yc"]
    Zc = result["Zc"]
    Yc_m = result["Yc_m"]
    Zc_m = result["Zc_m"]
    Lsub = result["Lsub"]
    FF_I_dB = result["FF_I_dB"]

    ele0 = array.system.ele0
    azi0 = array.system.azi0
    ele = array.ele
    azi = array.azi
    Mask_EA = array.Mask_EA

    Iele = np.argmin(np.abs(ele - ele0))
    Iazi = np.argmin(np.abs(azi - azi0))

    # Trova max
    max_idx = np.unravel_index(np.argmax(FF_I_dB), FF_I_dB.shape)
    Iele_max = max_idx[0]
    Iazi_max = max_idx[1]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Subarray map
    ax1 = fig.add_subplot(gs[0, 0])
    Ntrans = Yc.shape[1]
    for ih in range(Ntrans):
        RGBcolor = np.random.rand(3)
        yc_col = Yc[:, ih]
        zc_col = Zc[:, ih]
        valid = ~np.isnan(yc_col)
        ax1.scatter(yc_col[valid], zc_col[valid], s=40, c=[RGBcolor], marker="s")

    ax1.set_xlabel("y [m]")
    ax1.set_ylabel("z [m]")
    ax1.set_title("Antenna Sub-arrays")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # 2. Cluster size histogram
    ax2 = fig.add_subplot(gs[0, 1])
    unique_sizes, counts = np.unique(Lsub.astype(int), return_counts=True)
    ax2.bar(unique_sizes, counts)
    ax2.set_xlabel("Cluster size")
    ax2.set_ylabel("Number of clusters")
    ax2.set_title(f"Total elements: {int(np.sum(Lsub))}")
    ax2.grid(True, alpha=0.3)

    # 3. Vertical plane cut
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ele, FF_I_dB[:, Iazi], "b", linewidth=2, label="RPE")
    ax3.plot(ele, Mask_EA[:, Iazi], "g", linewidth=2, label="Mask")
    ax3.plot(ele, FF_I_dB[:, Iazi_max], "r", linewidth=2, label="RPE_max")
    ax3.set_xlim([-90, 90])
    ax3.set_ylim([-30, np.max(Mask_EA[:, Iazi]) + 0.5])
    ax3.set_xlabel(r"$\theta$ [deg]")
    ax3.set_ylabel(r"RPE R($\theta$,$\phi$) [dB]")
    ax3.set_title("Vertical plane")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Horizontal plane cut
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(azi, FF_I_dB[Iele, :], "b", linewidth=2, label="RPE")
    ax4.plot(azi, Mask_EA[Iele, :], "g", linewidth=2, label="Mask")
    ax4.plot(azi, FF_I_dB[Iele_max, :], "r", linewidth=2, label="RPE_max")
    ax4.set_xlim([-90, 90])
    ax4.set_ylim([-30, np.max(Mask_EA[Iele, :]) + 0.5])
    ax4.set_xlabel(r"$\phi$ [deg]")
    ax4.set_ylabel(r"RPE R($\theta$,$\phi$) [dB]")
    ax4.set_title("Horizontal plane")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"{label_line} - Analysis", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def plot_monte_carlo_statistics(
    all_Cm: List[int],
    all_Ntrans: List[int],
    all_Nel: List[int],
    save_path: Optional[str] = None,
):
    """
    Plot statistiche Monte Carlo - come Generation_code.m righe 264-296

    figure
    subplot(1,3,1)
    plot(simulation(:,end-2))  % cost function
    subplot(1,3,2)
    plot(simulation(:,end-1))  % number of clusters
    subplot(1,3,3)
    plot(simulation(:,end))    % number of elements
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Cost function
    axes[0].plot(all_Cm)
    axes[0].set_ylabel("Cost function")
    axes[0].set_xlabel("Iteration")
    axes[0].set_title("Cost Function Evolution")
    axes[0].grid(True, alpha=0.3)

    # Number of clusters
    axes[1].plot(all_Ntrans)
    axes[1].set_ylabel("Number of clusters")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Cluster Count Evolution")
    axes[1].grid(True, alpha=0.3)

    # Number of elements
    axes[2].plot(all_Nel)
    axes[2].set_ylabel("Number of elements")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("Element Count Evolution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def plot_cost_vs_clusters(
    all_Cm: List[int],
    all_Ntrans: List[int],
    save_path: Optional[str] = None,
):
    """
    Plot cost function vs numero cluster - come Generation_code.m righe 279-290

    figure
    plot(Narray, fcost, 'x'); grid
    xlabel('N clusters'); ylabel('N points exceeding mask')
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filtra valori infiniti
    valid = [i for i, cm in enumerate(all_Cm) if cm != float("inf")]
    Ntrans_valid = [all_Ntrans[i] for i in valid]
    Cm_valid = [all_Cm[i] for i in valid]

    ax.scatter(Ntrans_valid, Cm_valid, marker="x", alpha=0.5)

    # Calcola media e std per ogni numero di cluster
    unique_Ntrans = sorted(set(Ntrans_valid))
    means = []
    stds = []

    for n in unique_Ntrans:
        cms = [Cm_valid[i] for i, nt in enumerate(Ntrans_valid) if nt == n]
        if cms:
            means.append(np.mean(cms))
            stds.append(np.std(cms))
        else:
            means.append(0)
            stds.append(0)

    ax.errorbar(unique_Ntrans, means, yerr=stds, fmt="r-", linewidth=2, capsize=3)

    ax.set_xlabel("N. clusters")
    ax.set_ylabel("N. points exceeding mask")
    ax.set_title("Cost Function vs Number of Clusters")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def plot_cluster_histogram(
    all_Ntrans: List[int],
    save_path: Optional[str] = None,
):
    """
    Istogramma numero cluster - come Generation_code.m righe 292-295

    figure
    hist(all_Ntrans); grid
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(all_Ntrans, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of TRx chains")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Number of Clusters")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()


def cdf_plot(y_vector: np.ndarray, save_path: Optional[str] = None):
    """
    Plot CDF - come cdf_plot.m

    [x_cdf y_cdf] = cdf_plot(y_vector)
    """
    # Ordina i valori
    y_sorted = np.sort(y_vector.flatten())
    y_sorted = y_sorted[~np.isnan(y_sorted)]

    # Calcola CDF
    n = len(y_sorted)
    x_cdf = y_sorted
    y_cdf = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_cdf, y_cdf, linewidth=2)
    ax.set_xlabel("Value [dB]")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution Function")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvato in {save_path}")

    plt.show()

    return x_cdf, y_cdf


def plot_from_json(filename: str = "mc_results.json"):
    """
    Plot risultati da file JSON salvato
    """
    with open(filename, "r") as f:
        data = json.load(f)

    stats = data["statistics"]

    # Plot statistiche Monte Carlo
    plot_monte_carlo_statistics(
        stats["all_Cm"],
        stats["all_Ntrans"],
        stats["all_Nel"],
    )

    # Plot cost vs clusters
    plot_cost_vs_clusters(
        stats["all_Cm"],
        stats["all_Ntrans"],
    )

    # Plot histogram
    plot_cluster_histogram(stats["all_Ntrans"])


def plot_sll_hpbw_comparison(results_list: List[Dict], labels: List[str],
                              array: AntennaArray,
                              save_path: Optional[str] = None):
    """
    Confronta SLL e HPBW tra diverse soluzioni (come regular analysis.py)
    
    INPUT:
    results_list: lista di result dict con FF_I_dB
    labels: nomi per ogni soluzione
    array: AntennaArray per estrarre metriche
    """
    from scipy.signal import find_peaks
    
    ele = array.ele
    azi = array.azi
    ele0 = array.system.ele0
    azi0 = array.system.azi0
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    all_metrics = []
    
    # 1. Elevation cuts comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (result, label) in enumerate(zip(results_list, labels)):
        FF_I_dB = result['FF_I_dB']
        Iazi = np.argmin(np.abs(azi - azi0))
        ele_cut = FF_I_dB[:, Iazi]
        ax1.plot(ele, ele_cut, color=colors[i], linewidth=2, label=label)
        
        metrics = extract_lobe_metrics(FF_I_dB, azi, ele, azi0, ele0, result.get('G_boresight'))
        all_metrics.append(metrics)
    
    ax1.axhline(y=all_metrics[0]['main_lobe_gain'] - 3, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Elevation θ [deg]', fontweight='bold')
    ax1.set_ylabel('Gain [dBi]', fontweight='bold')
    ax1.set_title('Elevation Cuts Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-90, 90])
    
    # 2. Azimuth cuts comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (result, label) in enumerate(zip(results_list, labels)):
        FF_I_dB = result['FF_I_dB']
        Iele = np.argmin(np.abs(ele - ele0))
        azi_cut = FF_I_dB[Iele, :]
        ax2.plot(azi, azi_cut, color=colors[i], linewidth=2, label=label)
    
    ax2.set_xlabel('Azimuth φ [deg]', fontweight='bold')
    ax2.set_ylabel('Gain [dBi]', fontweight='bold')
    ax2.set_title('Azimuth Cuts Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-90, 90])
    
    # 3. SLL Comparison Bar Chart
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(labels))
    width = 0.35
    sll_ele = [m['sll_ele_relative'] for m in all_metrics]
    sll_azi = [m['sll_azi_relative'] for m in all_metrics]
    
    bars1 = ax3.bar(x - width/2, sll_ele, width, label='SLL Elevation', color='steelblue')
    bars2 = ax3.bar(x + width/2, sll_azi, width, label='SLL Azimuth', color='coral')
    
    ax3.axhline(y=-20, color='red', linestyle='--', label='Target -20dB')
    ax3.set_ylabel('SLL [dB] (relative)', fontweight='bold')
    ax3.set_title('Side Lobe Level Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. HPBW Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    hpbw_ele = [m['hpbw_ele'] for m in all_metrics]
    hpbw_azi = [m['hpbw_azi'] for m in all_metrics]
    
    bars1 = ax4.bar(x - width/2, hpbw_ele, width, label='HPBW Elevation', color='steelblue')
    bars2 = ax4.bar(x + width/2, hpbw_azi, width, label='HPBW Azimuth', color='coral')
    
    ax4.set_ylabel('HPBW [deg]', fontweight='bold')
    ax4.set_title('Half Power Beam Width Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. N Lobes Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    n_lobes_ele = [m['n_lobes_ele'] for m in all_metrics]
    n_lobes_azi = [m['n_lobes_azi'] for m in all_metrics]
    
    bars1 = ax5.bar(x - width/2, n_lobes_ele, width, label='Lobes Elevation', color='steelblue')
    bars2 = ax5.bar(x + width/2, n_lobes_azi, width, label='Lobes Azimuth', color='coral')
    
    ax5.set_ylabel('Number of Lobes', fontweight='bold')
    ax5.set_title('Lobe Count Comparison', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    table_data = []
    headers = ['Metric'] + labels
    table_data.append(['Main Lobe [dBi]'] + [f"{m['main_lobe_gain']:.1f}" for m in all_metrics])
    table_data.append(['SLL Ele [dB]'] + [f"{m['sll_ele_relative']:.1f}" for m in all_metrics])
    table_data.append(['SLL Azi [dB]'] + [f"{m['sll_azi_relative']:.1f}" for m in all_metrics])
    table_data.append(['HPBW Ele [°]'] + [f"{m['hpbw_ele']:.1f}" for m in all_metrics])
    table_data.append(['HPBW Azi [°]'] + [f"{m['hpbw_azi']:.1f}" for m in all_metrics])
    table_data.append(['N Lobes Ele'] + [str(m['n_lobes_ele']) for m in all_metrics])
    table_data.append(['N Lobes Azi'] + [str(m['n_lobes_azi']) for m in all_metrics])
    
    table = ax6.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax6.set_title('Performance Summary', fontweight='bold', y=0.85)
    
    plt.suptitle('Lobe Analysis Comparison', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvato in {save_path}")
    
    plt.show()
    
    return all_metrics


if __name__ == "__main__":
    # Test: plot da file JSON se esiste
    try:
        plot_from_json("mc_results.json")
    except FileNotFoundError:
        print("File mc_results.json non trovato. Esegui prima antenna_clustering_GA.py")
