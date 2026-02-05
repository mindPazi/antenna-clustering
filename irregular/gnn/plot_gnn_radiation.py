import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

# Add antenna physics utilities to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimization" / "pyvers"))

from antenna_physics import (
    LatticeConfig,
    SystemConfig,
    MaskConfig,
    ElementPatternConfig,
    AntennaArray,
)
from gnn import assignments_to_antenna_format


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
        peaks, _ = find_peaks(cut)
        threshold = max_val - 3
        side_peaks = [p for p in peaks if cut[p] < threshold]
        if side_peaks:
            return max(cut[p] for p in side_peaks)
        return -30

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
                       title="Lobe Analysis"):
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
                label=f"SLL: {metrics['sll_ele_relative']:.1f}dB")
    ax1.set_xlabel('Elevation [deg]')
    ax1.set_ylabel('Normalized Gain [dB]')
    ax1.set_title(f"Elevation Cut (azi={azi0} deg)\nHPBW={metrics['hpbw_ele']:.1f} deg")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-25, 30])

    # 2. Azimuth Cut with Lobes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(azi, metrics['azi_cut'], 'b-', linewidth=2, label='Azimuth Cut')
    ax2.axhline(y=-3, color='r', linestyle='--', alpha=0.7, label='-3dB (HPBW)')
    ax2.axhline(y=metrics['sll_azi_relative'], color='g', linestyle=':', alpha=0.7,
                label=f"SLL: {metrics['sll_azi_relative']:.1f}dB")
    ax2.set_xlabel('Azimuth [deg]')
    ax2.set_ylabel('Normalized Gain [dB]')
    ax2.set_title(f"Azimuth Cut (ele={ele0} deg)\nHPBW={metrics['hpbw_azi']:.1f} deg")
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
        ['Main Lobe Gain', f"{metrics['main_lobe_gain']:.2f} dBi"],
        ['HPBW Elevation', f"{metrics['hpbw_ele']:.1f} deg"],
        ['HPBW Azimuth', f"{metrics['hpbw_azi']:.1f} deg"],
        ['SLL Elevation', f"{metrics['sll_ele_relative']:.1f} dB"],
        ['SLL Azimuth', f"{metrics['sll_azi_relative']:.1f} dB"],
        ['Lobes (Ele)', f"{metrics['n_lobes_ele']}"],
        ['Lobes (Azi)', f"{metrics['n_lobes_azi']}"],
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
    plt.show()
    return metrics


def compute_radiation(cl, num_cl, grid_shape, title):
    clusters_antenna = assignments_to_antenna_format(cl, grid_shape=grid_shape)
    lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.7, dist_y=0.5, lattice_type=1)
    system = SystemConfig(freq=29.5e9, azi0=0, ele0=0, dele=0.5, dazi=0.5)
    mask = MaskConfig(elem=30, azim=60, SLL_level=20, SLLin=15)
    eef = ElementPatternConfig(P=1, Gel=5, load_file=0)
    array = AntennaArray(lattice, system, mask, eef)
    result_ff = array.evaluate_clustering(clusters_antenna)
    lobe_metrics = plot_lobe_analysis(result_ff['FF_I_dB'], array,
                                      G_boresight=result_ff['G_boresight'], title=title)
    lobe_metrics['sll_in'] = result_ff['sll_in']
    lobe_metrics['sll_out'] = result_ff['sll_out']
    lobe_metrics['Cm'] = result_ff['Cm']
    return lobe_metrics
