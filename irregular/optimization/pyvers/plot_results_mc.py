"""
Plotting functions for antenna clustering results
Aligned with clustering_comparison.ipynb notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

from antenna_physics import AntennaArray


# ============================================================
# LOBE ANALYSIS FUNCTIONS (Cell 30)
# ============================================================

def extract_lobe_metrics(FF_I_dB, azi, ele, azi0, ele0, G_boresight=None):
    """
    Extract lobe performance metrics from far-field pattern.
    """
    # Find boresight indices
    ele_idx = np.argmin(np.abs(ele - ele0))
    azi_idx = np.argmin(np.abs(azi - azi0))

    # Extract cuts
    ele_cut = FF_I_dB[:, azi_idx]  # Elevation cut at azimuth = azi0
    azi_cut = FF_I_dB[ele_idx, :]  # Azimuth cut at elevation = ele0

    # Main lobe gain
    main_lobe_gain = G_boresight if G_boresight else np.max(FF_I_dB)

    # HPBW (Half-Power Beam Width) - find -3dB points
    def find_hpbw(cut, angles):
        max_idx = np.argmax(cut)
        threshold = cut[max_idx] - 3

        # Find left -3dB point
        left_idx = max_idx
        for i in range(max_idx, -1, -1):
            if cut[i] < threshold:
                left_idx = i
                break

        # Find right -3dB point
        right_idx = max_idx
        for i in range(max_idx, len(cut)):
            if cut[i] < threshold:
                right_idx = i
                break

        return angles[right_idx] - angles[left_idx]

    hpbw_ele = find_hpbw(ele_cut, ele)
    hpbw_azi = find_hpbw(azi_cut, azi)

    # Side Lobe Level (relative to main lobe)
    def find_sll_relative(cut, angles):
        max_val = np.max(cut)
        max_idx = np.argmax(cut)

        # Find peaks excluding main lobe region
        peaks, _ = find_peaks(cut)

        # Filter peaks outside main lobe (-3dB region)
        threshold = max_val - 3
        side_peaks = [p for p in peaks if cut[p] < threshold]

        if side_peaks:
            max_side = max(cut[p] for p in side_peaks)
            return max_side  # Already relative (normalized pattern)
        return -30  # Default if no side lobes found

    sll_ele_relative = find_sll_relative(ele_cut, ele)
    sll_azi_relative = find_sll_relative(azi_cut, azi)

    # Count lobes
    peaks_ele, _ = find_peaks(ele_cut, height=-30)
    peaks_azi, _ = find_peaks(azi_cut, height=-30)

    return {
        'main_lobe_gain': main_lobe_gain,
        'hpbw_ele': hpbw_ele,
        'hpbw_azi': hpbw_azi,
        'sll_ele_relative': sll_ele_relative,
        'sll_azi_relative': sll_azi_relative,
        'n_lobes_ele': len(peaks_ele),
        'n_lobes_azi': len(peaks_azi),
        'ele_cut': ele_cut,
        'azi_cut': azi_cut,
    }


def plot_lobe_analysis(FF_I_dB, antenna_array, G_boresight=None,
                       title="Lobe Analysis", save_path=None):
    """
    Plot lobe analysis: elevation/azimuth cuts, 2D pattern, metrics table, polar plots.
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
    ax1.plot(ele, metrics['ele_cut'], 'b-', linewidth=2, label='Elevation Cut')
    ax1.axhline(y=-3, color='r', linestyle='--', alpha=0.7, label='-3dB (HPBW)')
    ax1.axhline(y=metrics['sll_ele_relative'], color='g', linestyle=':', alpha=0.7,
                label=f'SLL: {metrics["sll_ele_relative"]:.1f}dB')
    ax1.set_xlabel('Elevation [deg]')
    ax1.set_ylabel('Normalized Gain [dB]')
    ax1.set_title(f'Elevation Cut (azi={azi0}°)\nHPBW={metrics["hpbw_ele"]:.1f}°')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-25, 30])

    # 2. Azimuth Cut with Lobes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(azi, metrics['azi_cut'], 'b-', linewidth=2, label='Azimuth Cut')
    ax2.axhline(y=-3, color='r', linestyle='--', alpha=0.7, label='-3dB (HPBW)')
    ax2.axhline(y=metrics['sll_azi_relative'], color='g', linestyle=':', alpha=0.7,
                label=f'SLL: {metrics["sll_azi_relative"]:.1f}dB')
    ax2.set_xlabel('Azimuth [deg]')
    ax2.set_ylabel('Normalized Gain [dB]')
    ax2.set_title(f'Azimuth Cut (ele={ele0}°)\nHPBW={metrics["hpbw_azi"]:.1f}°')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-25, 30])

    # 3. 2D Pattern (contour)
    ax3 = fig.add_subplot(gs[0, 2])
    levels = np.arange(-40, 35, 3)
    contour = ax3.contourf(antenna_array.AZI, antenna_array.ELE, FF_I_dB,
                           levels=levels, cmap='jet', extend='both')
    plt.colorbar(contour, ax=ax3, label='dB')
    ax3.plot(azi0, ele0, 'w*', markersize=15, markeredgecolor='k')
    ax3.set_xlabel('Azimuth [deg]')
    ax3.set_ylabel('Elevation [deg]')
    ax3.set_title('2D Far-Field Pattern')

    # 4. Metrics Summary Table
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')

    table_data = [
        ['Main Lobe Gain', f'{metrics["main_lobe_gain"]:.2f} dBi'],
        ['HPBW Elevation', f'{metrics["hpbw_ele"]:.1f}°'],
        ['HPBW Azimuth', f'{metrics["hpbw_azi"]:.1f}°'],
        ['SLL Elevation', f'{metrics["sll_ele_relative"]:.1f} dB'],
        ['SLL Azimuth', f'{metrics["sll_azi_relative"]:.1f} dB'],
        ['Lobes (Ele)', f'{metrics["n_lobes_ele"]}'],
        ['Lobes (Azi)', f'{metrics["n_lobes_azi"]}'],
    ]

    table = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='center',
                      colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')

    # 5. Polar plot elevation
    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    theta_rad = np.deg2rad(ele)
    r = metrics['ele_cut'] + 40  # Shift to positive
    ax5.plot(theta_rad, r, 'b-', linewidth=2)
    ax5.set_theta_zero_location('N')
    ax5.set_title('Elevation Pattern (Polar)', y=1.1)

    # 6. Polar plot azimuth
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    theta_rad = np.deg2rad(azi)
    r = metrics['azi_cut'] + 40  # Shift to positive
    ax6.plot(theta_rad, r, 'b-', linewidth=2)
    ax6.set_theta_zero_location('N')
    ax6.set_title('Azimuth Pattern (Polar)', y=1.1)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return metrics


# ============================================================
# HELPER FUNCTIONS (Cell 31)
# ============================================================

def rows_to_clusters(selected_rows, optimizer):
    """Convert selected_rows (MC) to Cluster list, filling uncovered elements."""
    clusters = [optimizer._all_clusters_flat[i] for i in np.where(selected_rows == 1)[0]]
    return optimizer._fill_uncovered(clusters, set().union(*(optimizer._cluster_elements[i] for i in np.where(selected_rows == 1)[0])) if np.any(selected_rows == 1) else set())


def genes_to_clusters(genes, ga_opt):
    """Convert genes (GA) to Cluster list, filling uncovered elements."""
    clusters = [ga_opt.all_subarrays[i] for i in np.where(genes == 1)[0]]
    return ga_opt._fill_uncovered(clusters, genes)


def get_ff_from_mc(solution, antenna_array, mc_optimizer):
    """Get FF_I_dB from a Monte Carlo solution."""
    if 'selected_rows' in solution:
        Cluster = rows_to_clusters(solution['selected_rows'], mc_optimizer)
        if len(Cluster) > 0:
            result = antenna_array.evaluate_clustering(Cluster)
            return result['FF_I_dB'], result['G_boresight']
    return None, None


def get_ff_from_ga(solution, antenna_array, ga_opt):
    """Get FF_I_dB from a Genetic Algorithm solution."""
    if 'genes' in solution:
        Cluster = genes_to_clusters(solution['genes'], ga_opt)
        if len(Cluster) > 0:
            result = antenna_array.evaluate_clustering(Cluster)
            return result['FF_I_dB'], result['G_boresight']
    return None, None


# ============================================================
# CLUSTER LAYOUT PLOTTING (Cell 42)
# ============================================================

def plot_cluster_layout(clusters, antenna_array, title="Cluster Layout", ax=None):
    """
    Plot antenna array with clusters colored by group.
    Each cluster gets a unique color.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all antenna positions in gray (background)
    y_vec = antenna_array.Y.flatten()
    z_vec = antenna_array.Z.flatten()
    ax.scatter(y_vec, z_vec, c='lightgray', s=80, alpha=0.5, label='Unused elements')

    # Generate colors for clusters
    n_clusters = len(clusters)
    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, min(50, n_clusters)))

    # Plot each cluster with its color
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]
        for pos in cluster:
            n_idx, m_idx = pos[0], pos[1]
            # Find position in array
            y_idx = np.where(antenna_array.NN[0, :] == n_idx)[0]
            z_idx = np.where(antenna_array.MM[:, 0] == m_idx)[0]
            if len(y_idx) > 0 and len(z_idx) > 0:
                y = antenna_array.Y[z_idx[0], y_idx[0]]
                z = antenna_array.Z[z_idx[0], y_idx[0]]
                ax.scatter(y, z, c=[color], s=120, edgecolors='black', linewidths=0.5, zorder=10)

    ax.set_xlabel('Y position [m]')
    ax.set_ylabel('Z position [m]')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax
