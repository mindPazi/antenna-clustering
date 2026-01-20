"""
Funzioni di plotting per risultati clustering antenna
Fedele ai plot MATLAB in PostProcessing_singlesolution.m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from typing import Dict, List, Optional

from antenna_physics import AntennaArray


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


if __name__ == "__main__":
    # Test: plot da file JSON se esiste
    try:
        plot_from_json("mc_results.json")
    except FileNotFoundError:
        print("File mc_results.json non trovato. Esegui prima antenna_clustering_GA.py")
