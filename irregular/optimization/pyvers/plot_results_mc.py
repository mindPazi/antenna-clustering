"""
Plotting functions for antenna clustering results
Aligned with clustering_comparison.ipynb notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
from scipy.signal import find_peaks

from antenna_physics import AntennaArray


def plot_mc_results(results_original, results_optimized):
    """Plot Monte Carlo results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Monte Carlo Results', fontsize=14, fontweight='bold')

    # Cost function evolution
    axes[0].plot(results_original['all_Cm'], 'b-', alpha=0.7, label='Original')
    axes[0].plot(results_optimized['all_Cm'], 'r-', alpha=0.7, label='Optimized')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Cost Function (Cm)')
    axes[0].set_title('Cost Function Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram
    cm_orig = [c for c in results_original['all_Cm'] if c != float('inf') and c < 5000]
    cm_opt = [c for c in results_optimized['all_Cm'] if c != float('inf') and c < 5000]
    axes[1].hist(cm_orig, bins=30, alpha=0.5, label='Original', color='blue')
    axes[1].hist(cm_opt, bins=30, alpha=0.5, label='Optimized', color='red')
    axes[1].set_xlabel('Cost Function (Cm)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cost Function Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Clusters
    axes[2].plot(results_original['all_Ntrans'], 'b-', alpha=0.7, label='Original')
    axes[2].plot(results_optimized['all_Ntrans'], 'r-', alpha=0.7, label='Optimized')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Number of Clusters')
    axes[2].set_title('Number of Clusters')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def extract_lobe_metrics(FF_I_dB, azi, ele, azi0, ele0, G_boresight=None):
    """
    Extract lobe performance metrics from radiation pattern
    Similar to regular/opt/grid_search/regular_arrays.py

    Returns: SLL, HPBW, main_lobe_gain, etc.
    """
    Iele = np.argmin(np.abs(ele - ele0))
    Iazi = np.argmin(np.abs(azi - azi0))

    main_lobe_gain = np.max(FF_I_dB)
    if G_boresight is None:
        G_boresight = main_lobe_gain

    # SLL Elevation
    ele_cut = FF_I_dB[:, Iazi]
    main_lobe_mask = np.abs(ele - ele0) <= 10
    ele_cut_sidelobe = ele_cut.copy()
    ele_cut_sidelobe[main_lobe_mask] = -np.inf

    sll_ele_dB = np.max(ele_cut_sidelobe) if np.any(~main_lobe_mask) else -100
    sll_ele_relative = sll_ele_dB - G_boresight

    # SLL Azimuth
    azi_cut = FF_I_dB[Iele, :]
    main_lobe_mask_azi = np.abs(azi - azi0) <= 10
    azi_cut_sidelobe = azi_cut.copy()
    azi_cut_sidelobe[main_lobe_mask_azi] = -np.inf

    sll_azi_dB = np.max(azi_cut_sidelobe) if np.any(~main_lobe_mask_azi) else -100
    sll_azi_relative = sll_azi_dB - G_boresight

    # HPBW Elevation
    half_power_dB = main_lobe_gain - 3
    above_halfpower = ele_cut >= half_power_dB
    if np.any(above_halfpower):
        edges = np.where(np.diff(above_halfpower.astype(int)) != 0)[0]
        hpbw_ele = ele[edges[-1]] - ele[edges[0]] if len(edges) >= 2 else np.nan
    else:
        hpbw_ele = np.nan

    # HPBW Azimuth
    above_halfpower_azi = azi_cut >= half_power_dB
    if np.any(above_halfpower_azi):
        edges_azi = np.where(np.diff(above_halfpower_azi.astype(int)) != 0)[0]
        hpbw_azi = azi[edges_azi[-1]] - azi[edges_azi[0]] if len(edges_azi) >= 2 else np.nan
    else:
        hpbw_azi = np.nan

    # Count lobes
    peaks_ele, _ = find_peaks(ele_cut, height=-50, distance=5)
    peaks_azi, _ = find_peaks(azi_cut, height=-50, distance=5)

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


def plot_lobe_analysis(FF_I_dB, antenna_array, G_boresight=None,
                       title="Lobe Analysis", save_path=None):
    """
    Plot lobe analysis similar to regular folder
    Shows SLL, HPBW, main lobe and side lobes
    """
    ele0 = antenna_array.system.ele0
    azi0 = antenna_array.system.azi0
    ele = antenna_array.ele
    azi = antenna_array.azi

    metrics = extract_lobe_metrics(FF_I_dB, azi, ele, azi0, ele0, G_boresight)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Elevation Cut with Lobes
    ax1 = fig.add_subplot(gs[0, 0])
    ele_cut = metrics['ele_cut']
    ax1.plot(ele, ele_cut, 'b-', linewidth=2, label='Pattern')
    ax1.axhline(y=metrics['main_lobe_gain'] - 3, color='orange', linestyle='--',
                linewidth=1.5, label=f'-3dB')
    for pk_idx in metrics['peaks_ele_idx']:
        ax1.plot(ele[pk_idx], ele_cut[pk_idx], 'ro', markersize=8)
    ax1.axvline(x=ele0, color='g', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Elevation θ [deg]', fontweight='bold')
    ax1.set_ylabel('Gain [dBi]', fontweight='bold')
    ax1.set_title(f'Elevation Cut (φ={azi0}°)\nHPBW={metrics["hpbw_ele"]:.1f}°, SLL={metrics["sll_ele_relative"]:.1f}dB', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-90, 90])

    # 2. Azimuth Cut with Lobes
    ax2 = fig.add_subplot(gs[0, 1])
    azi_cut = metrics['azi_cut']
    ax2.plot(azi, azi_cut, 'r-', linewidth=2, label='Pattern')
    ax2.axhline(y=metrics['main_lobe_gain'] - 3, color='orange', linestyle='--', linewidth=1.5)
    for pk_idx in metrics['peaks_azi_idx']:
        ax2.plot(azi[pk_idx], azi_cut[pk_idx], 'bo', markersize=8)
    ax2.axvline(x=azi0, color='g', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Azimuth φ [deg]', fontweight='bold')
    ax2.set_ylabel('Gain [dBi]', fontweight='bold')
    ax2.set_title(f'Azimuth Cut (θ={ele0}°)\nHPBW={metrics["hpbw_azi"]:.1f}°, SLL={metrics["sll_azi_relative"]:.1f}dB', fontweight='bold')
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
    ax3.plot(azi0, ele0, 'w*', markersize=15, markeredgecolor='black')
    ax3.set_xlabel('Azimuth φ [deg]', fontweight='bold')
    ax3.set_ylabel('Elevation θ [deg]', fontweight='bold')
    ax3.set_title('2D Radiation Pattern', fontweight='bold')

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
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # 5. SLL Comparison Bar Chart
    ax5 = fig.add_subplot(gs[1, 1])
    sll_data = [metrics['sll_ele_relative'], metrics['sll_azi_relative']]
    bars = ax5.bar(['Elevation', 'Azimuth'], sll_data, color=['steelblue', 'coral'])
    ax5.axhline(y=-20, color='red', linestyle='--', label='Target -20dB')
    ax5.axhline(y=-15, color='orange', linestyle='--', label='Target -15dB')
    ax5.set_ylabel('SLL [dB] (relative)', fontweight='bold')
    ax5.set_title('Side Lobe Level Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
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
        print(f"Plot saved to {save_path}")

    plt.show()
    return metrics


def plot_lobe_comparison(results_list, labels, antenna_array, save_path=None):
    """
    Compare lobe analysis between multiple solutions
    """
    ele = antenna_array.ele
    azi = antenna_array.azi
    ele0 = antenna_array.system.ele0
    azi0 = antenna_array.system.azi0

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

    ax3.bar(x - width/2, sll_ele, width, label='SLL Elevation', color='steelblue')
    ax3.bar(x + width/2, sll_azi, width, label='SLL Azimuth', color='coral')
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

    ax4.bar(x - width/2, hpbw_ele, width, label='HPBW Elevation', color='steelblue')
    ax4.bar(x + width/2, hpbw_azi, width, label='HPBW Azimuth', color='coral')
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

    ax5.bar(x - width/2, n_lobes_ele, width, label='Lobes Ele', color='steelblue')
    ax5.bar(x + width/2, n_lobes_azi, width, label='Lobes Azi', color='coral')
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

    table = ax6.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax6.set_title('Performance Summary', fontweight='bold', y=0.85)

    plt.suptitle('Lobe Analysis Comparison', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    return all_metrics


def rows_to_clusters(selected_rows, optimizer):
    """Convert selected_rows to Cluster list using optimizer data"""
    clusters = []
    offset = 0
    for bb, S in enumerate(optimizer.S_all):
        Nsub = optimizer.N_all[bb]
        for idx in range(Nsub):
            if selected_rows[offset + idx] == 1:
                clusters.append(S[idx])
        offset += Nsub
    return clusters


def genes_to_clusters(genes, ga_optimizer):
    """Convert genes (GA) to Cluster list using GA optimizer data"""
    clusters = []
    offset = 0
    for bb, S in enumerate(ga_optimizer.S_all):
        Nsub = ga_optimizer.N_all[bb]
        for idx in range(Nsub):
            if genes[offset + idx] == 1:
                clusters.append(S[idx])
        offset += Nsub
    return clusters


def get_ff_i_db_mc(solution, antenna_array, optimizer):
    """Get FF_I_dB from a MC solution"""
    if 'FF_I_dB' in solution:
        return solution['FF_I_dB'], solution.get('G_boresight')

    if 'selected_rows' in solution:
        Cluster = rows_to_clusters(solution['selected_rows'], optimizer)
        if len(Cluster) > 0:
            result = antenna_array.evaluate_clustering(Cluster)
            return result['FF_I_dB'], result['G_boresight']

    return None, None


def get_ff_i_db_ga(solution, antenna_array, ga_opt):
    """Get FF_I_dB from a GA solution"""
    if 'FF_I_dB' in solution:
        return solution['FF_I_dB'], solution.get('G_boresight')

    if 'genes' in solution:
        Cluster = genes_to_clusters(solution['genes'], ga_opt)
        if len(Cluster) > 0:
            result = antenna_array.evaluate_clustering(Cluster)
            return result['FF_I_dB'], result['G_boresight']

    return None, None
