"""
Extract and plot intermediate steps from checkpoint JSON files.

Usage:
    python extract_and_plot.py <json_file> [--step <step_name>] [--list] [--save <output_dir>]

Examples:
    # List available steps in a checkpoint file
    python extract_and_plot.py gnn_case_1.json --list

    # Plot all steps from a GNN checkpoint
    python extract_and_plot.py gnn_case_1.json

    # Plot only the radiation step
    python extract_and_plot.py gnn_case_1.json --step radiation

    # Plot a clustering comparison checkpoint
    python extract_and_plot.py clustering_case_1.json --step radiation_mc

    # Save plots to a directory instead of showing them
    python extract_and_plot.py gnn_case_1.json --save ./plots
"""

import json
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_checkpoint(filepath):
    """Load a checkpoint JSON file."""
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    with open(filepath, 'r') as f:
        return json.load(f)


def list_steps(checkpoint):
    """Print available steps in a checkpoint."""
    print(f"Notebook: {checkpoint['notebook']}")
    print(f"Case: {checkpoint['case_name']}")
    print(f"Description: {checkpoint['description']}")
    print(f"Timestamp: {checkpoint['timestamp']}")
    print(f"\nAvailable steps:")
    for step_name, step_data in checkpoint['steps'].items():
        keys = list(step_data.keys()) if isinstance(step_data, dict) else []
        print(f"  - {step_name}: {', '.join(keys)}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_cluster_layout(step_data, config, title="Cluster Layout", save_path=None):
    """Plot cluster scatter and grid from training step data."""
    clusters = np.array(step_data['cluster_assignments'])
    rows = config['rows']
    cols = config['cols']
    dx = config['dx']
    dy = config['dy']
    num_clusters = config['num_clusters']

    # Generate positions
    pos_x = np.array([c * dx for r in range(rows) for c in range(cols)])
    pos_y = np.array([r * dy for r in range(rows) for c in range(cols)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_clusters, 10)))
    if num_clusters > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_clusters, 20)))
    if num_clusters > 20:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_clusters))

    sizes_array = np.bincount(clusters, minlength=num_clusters)
    for k in range(num_clusters):
        mask = clusters == k
        if mask.sum() > 0:
            axes[0].scatter(pos_x[mask], pos_y[mask],
                            c=[colors[k % len(colors)]], s=80,
                            label=f'C{k} ({sizes_array[k]})' if num_clusters <= 10 else None,
                            alpha=0.8, edgecolors='black', linewidth=0.5)

    axes[0].set_xlabel('X (wavelengths)')
    axes[0].set_ylabel('Y (wavelengths)')
    axes[0].set_title(f'{title} - Scatter (K={num_clusters})')
    if num_clusters <= 10:
        axes[0].legend(loc='upper right', fontsize=7)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    grid = clusters.reshape(rows, cols)
    im = axes[1].imshow(grid, cmap='tab10' if num_clusters <= 10 else 'nipy_spectral',
                        vmin=0, vmax=num_clusters - 1)
    axes[1].set_title(f'{title} - Grid')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im, ax=axes[1], label='Cluster')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_radiation(step_data, config=None, title="Radiation Pattern", save_path=None):
    """Plot radiation pattern from a radiation step."""
    ele_cut = np.array(step_data['ele_cut'])
    azi_cut = np.array(step_data['azi_cut'])

    has_2d = 'FF_I_dB' in step_data and step_data['FF_I_dB'] is not None
    has_angles = 'ele_angles' in step_data and 'azi_angles' in step_data

    if has_angles:
        ele = np.array(step_data['ele_angles'])
        azi = np.array(step_data['azi_angles'])
    else:
        ele = np.arange(len(ele_cut))
        azi = np.arange(len(azi_cut))

    ele0 = config.get('ele0', 0) if config else 0
    azi0 = config.get('azi0', 0) if config else 0

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Elevation cut
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ele, ele_cut, 'b-', linewidth=2, label='Elevation Cut')
    ax1.axhline(y=-3, color='r', linestyle='--', alpha=0.7, label='-3dB')
    sll_ele = step_data.get('sll_ele_relative')
    if sll_ele is not None:
        ax1.axhline(y=sll_ele, color='g', linestyle=':', alpha=0.7,
                    label=f'SLL: {sll_ele:.1f}dB')
    ax1.set_xlabel('Elevation [deg]')
    ax1.set_ylabel('Gain [dB]')
    hpbw_ele = step_data.get('hpbw_ele', '')
    ax1.set_title(f"Elevation Cut (azi={azi0} deg)\nHPBW={hpbw_ele}")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-25, 30])

    # Azimuth cut
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(azi, azi_cut, 'b-', linewidth=2, label='Azimuth Cut')
    ax2.axhline(y=-3, color='r', linestyle='--', alpha=0.7, label='-3dB')
    sll_azi = step_data.get('sll_azi_relative')
    if sll_azi is not None:
        ax2.axhline(y=sll_azi, color='g', linestyle=':', alpha=0.7,
                    label=f'SLL: {sll_azi:.1f}dB')
    ax2.set_xlabel('Azimuth [deg]')
    ax2.set_ylabel('Gain [dB]')
    hpbw_azi = step_data.get('hpbw_azi', '')
    ax2.set_title(f"Azimuth Cut (ele={ele0} deg)\nHPBW={hpbw_azi}")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-25, 30])

    # 2D contour
    ax3 = fig.add_subplot(gs[0, 2])
    if has_2d and has_angles:
        FF_I_dB = np.array(step_data['FF_I_dB'])
        AZI, ELE = np.meshgrid(azi, ele)
        levels = np.arange(-40, 35, 3)
        contour = ax3.contourf(AZI, ELE, FF_I_dB, levels=levels, cmap='jet', extend='both')
        plt.colorbar(contour, ax=ax3, label='dB')
        ax3.plot(azi0, ele0, 'w*', markersize=15, markeredgecolor='k')
        ax3.set_xlabel('Azimuth [deg]')
        ax3.set_ylabel('Elevation [deg]')
        ax3.set_title('2D Far-Field Pattern')
    else:
        ax3.text(0.5, 0.5, '2D data\nnot available', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12)
        ax3.set_title('2D Far-Field Pattern')

    # Metrics table
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    table_data = []
    for key, label in [
        ('main_lobe_gain', 'Main Lobe Gain [dBi]'),
        ('hpbw_ele', 'HPBW Elevation [deg]'),
        ('hpbw_azi', 'HPBW Azimuth [deg]'),
        ('sll_ele_relative', 'SLL Elevation [dB]'),
        ('sll_azi_relative', 'SLL Azimuth [dB]'),
        ('Cm', 'Cost Function (Cm)'),
        ('clustering_factor', 'Clustering Factor'),
        ('sll_out', 'SLL out FoV [dB]'),
        ('sll_in', 'SLL in FoV [dB]'),
        ('G_boresight', 'G boresight [dBi]'),
        ('n_lobes_ele', 'Lobes (Elevation)'),
        ('n_lobes_azi', 'Lobes (Azimuth)'),
    ]:
        val = step_data.get(key)
        if val is not None:
            if isinstance(val, float):
                table_data.append([label, f"{val:.2f}"])
            else:
                table_data.append([label, str(val)])
    if table_data:
        table = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'],
                          loc='center', cellLoc='center', colWidths=[0.6, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
    ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')

    # Polar elevation
    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    ax5.plot(np.deg2rad(ele), ele_cut + 40, 'b-', linewidth=2)
    ax5.set_theta_zero_location('N')
    ax5.set_title('Elevation (Polar)', y=1.1)

    # Polar azimuth
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    ax6.plot(np.deg2rad(azi), azi_cut + 40, 'b-', linewidth=2)
    ax6.set_theta_zero_location('N')
    ax6.set_title('Azimuth (Polar)', y=1.1)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_clustering_comparison(step_mc, step_ga, config, title="MC vs GA", save_path=None):
    """Plot MC vs GA radiation comparison (for clustering_comparison checkpoints)."""
    has_mc = step_mc is not None
    has_ga = step_ga is not None
    if not has_mc and not has_ga:
        print("No radiation data available for comparison.")
        return

    ele0 = config.get('ele0', 0)
    azi0 = config.get('azi0', 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ref = step_mc if has_mc else step_ga
    ele = np.array(ref['ele_angles'])
    azi = np.array(ref['azi_angles'])
    levels = np.arange(-40, 35, 3)

    # MC 2D pattern
    if has_mc and 'FF_I_dB' in step_mc:
        ff_mc = np.array(step_mc['FF_I_dB'])
        AZI, ELE = np.meshgrid(azi, ele)
        c1 = axes[0, 0].contourf(AZI, ELE, ff_mc, levels=levels, cmap='jet', extend='both')
        plt.colorbar(c1, ax=axes[0, 0], label='dB')
        axes[0, 0].plot(azi0, ele0, 'w*', markersize=12, markeredgecolor='k')
        axes[0, 0].set_title(f"Monte Carlo")
    axes[0, 0].set_xlabel('Azimuth [deg]')
    axes[0, 0].set_ylabel('Elevation [deg]')

    # GA 2D pattern
    if has_ga and 'FF_I_dB' in step_ga:
        ff_ga = np.array(step_ga['FF_I_dB'])
        AZI, ELE = np.meshgrid(azi, ele)
        c2 = axes[0, 1].contourf(AZI, ELE, ff_ga, levels=levels, cmap='jet', extend='both')
        plt.colorbar(c2, ax=axes[0, 1], label='dB')
        axes[0, 1].plot(azi0, ele0, 'w*', markersize=12, markeredgecolor='k')
        axes[0, 1].set_title(f"Genetic Algorithm")
    axes[0, 1].set_xlabel('Azimuth [deg]')
    axes[0, 1].set_ylabel('Elevation [deg]')

    # Elevation cut comparison
    if has_mc:
        axes[1, 0].plot(ele, np.array(step_mc['ele_cut']), 'b-', linewidth=2, label='MC')
    if has_ga:
        axes[1, 0].plot(ele, np.array(step_ga['ele_cut']), 'r--', linewidth=2, label='GA')
    axes[1, 0].axhline(y=-3, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].set_xlabel('Elevation [deg]')
    axes[1, 0].set_ylabel('Gain [dB]')
    axes[1, 0].set_title('Elevation Cut Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-25, 30])

    # Azimuth cut comparison
    if has_mc:
        axes[1, 1].plot(azi, np.array(step_mc['azi_cut']), 'b-', linewidth=2, label='MC')
    if has_ga:
        axes[1, 1].plot(azi, np.array(step_ga['azi_cut']), 'r--', linewidth=2, label='GA')
    axes[1, 1].axhline(y=-3, color='gray', linestyle=':', alpha=0.7)
    axes[1, 1].set_xlabel('Azimuth [deg]')
    axes[1, 1].set_ylabel('Gain [dB]')
    axes[1, 1].set_title('Azimuth Cut Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([-25, 30])

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_optimization_summary(steps, title="Optimization Summary", save_path=None):
    """Plot optimization summary: MC vs GA metrics comparison table + convergence."""
    mc = steps.get('mc_optimization', {})
    ga = steps.get('ga_optimization', {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Metrics comparison table
    axes[0].axis('off')
    table_data = []
    for label, mc_key, ga_key in [
        ('Best Cm', 'best_Cm', 'best_Cm'),
        ('N clusters', 'best_Ntrans', 'best_n_clusters'),
        ('SLL out [dB]', 'sll_out', 'sll_out'),
        ('SLL in [dB]', 'sll_in', 'sll_in'),
        ('Time [s]', 'elapsed_time', 'elapsed_time'),
    ]:
        mc_val = mc.get(mc_key, 'N/A')
        ga_val = ga.get(ga_key, 'N/A')
        if isinstance(mc_val, float):
            mc_val = f"{mc_val:.2f}"
        if isinstance(ga_val, float):
            ga_val = f"{ga_val:.2f}"
        table_data.append([label, str(mc_val), str(ga_val)])

    table = axes[0].table(cellText=table_data,
                          colLabels=['Metric', 'Monte Carlo', 'Genetic Alg.'],
                          loc='center', cellLoc='center', colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    axes[0].set_title('MC vs GA Comparison', fontsize=12, fontweight='bold')

    # GA convergence curve
    ga_history = ga.get('best_Cm_history', [])
    if ga_history:
        axes[1].plot(range(1, len(ga_history) + 1), ga_history, 'r-', linewidth=2)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Best Cm')
        axes[1].set_title('GA Convergence')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No convergence\ndata available',
                     transform=axes[1].transAxes, ha='center', va='center')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def plot_step(checkpoint, step_name, save_dir=None):
    """Plot a specific step from a checkpoint."""
    steps = checkpoint['steps']
    notebook = checkpoint['notebook']
    case = checkpoint['case_name']
    desc = checkpoint['description']

    if step_name not in steps:
        print(f"Step '{step_name}' not found. Available: {list(steps.keys())}")
        return

    data = steps[step_name]
    base_title = f"{notebook} / {case} - {desc}"
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{notebook}_{case}_{step_name}.png")

    config = steps.get('config', {})

    # Dispatch based on step name
    if step_name.startswith('training'):
        plot_cluster_layout(data, config, title=f"{base_title}\n{step_name}", save_path=save_path)

    elif step_name.startswith('radiation'):
        plot_radiation(data, config, title=f"{base_title}\n{step_name}", save_path=save_path)

    elif step_name == 'config':
        print(f"\n--- Configuration ({case}) ---")
        for k, v in data.items():
            print(f"  {k}: {v}")

    elif step_name in ('mc_optimization', 'ga_optimization'):
        print(f"\n--- {step_name} ({case}) ---")
        for k, v in data.items():
            if not isinstance(v, list):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: [{len(v)} values]")

    else:
        print(f"No specific plot handler for step '{step_name}'. Data keys: {list(data.keys())}")


def plot_all(checkpoint, save_dir=None):
    """Plot all available steps from a checkpoint."""
    steps = checkpoint['steps']
    notebook = checkpoint['notebook']
    case = checkpoint['case_name']
    desc = checkpoint['description']
    config = steps.get('config', {})
    base_title = f"{notebook} / {case} - {desc}"

    if notebook == 'gnn':
        # GNN: plot training (cluster layout) + radiation
        for sname in steps:
            if sname.startswith('training'):
                sp = os.path.join(save_dir, f"{notebook}_{case}_{sname}.png") if save_dir else None
                plot_cluster_layout(steps[sname], config,
                                    title=f"{base_title}\n{sname}", save_path=sp)
            elif sname.startswith('radiation'):
                sp = os.path.join(save_dir, f"{notebook}_{case}_{sname}.png") if save_dir else None
                plot_radiation(steps[sname], config,
                               title=f"{base_title}\n{sname}", save_path=sp)

    elif notebook == 'clustering_comparison':
        # Clustering: optimization summary + MC vs GA radiation comparison
        sp = os.path.join(save_dir, f"{notebook}_{case}_summary.png") if save_dir else None
        plot_optimization_summary(steps, title=f"{base_title}\nOptimization Summary",
                                  save_path=sp)

        mc_rad = steps.get('radiation_mc')
        ga_rad = steps.get('radiation_ga')
        if mc_rad or ga_rad:
            sp = os.path.join(save_dir, f"{notebook}_{case}_radiation.png") if save_dir else None
            plot_clustering_comparison(mc_rad, ga_rad, config,
                                       title=f"{base_title}\nMC vs GA Radiation",
                                       save_path=sp)

    # Print config
    if 'config' in steps:
        print(f"\n--- Configuration ---")
        for k, v in steps['config'].items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and plot intermediate steps from checkpoint JSON files.')
    parser.add_argument('json_file', help='Path to the checkpoint JSON file')
    parser.add_argument('--step', '-s', help='Specific step to plot (e.g., radiation, training)')
    parser.add_argument('--list', '-l', action='store_true', help='List available steps')
    parser.add_argument('--save', help='Save plots to this directory instead of showing')
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.json_file)

    if args.list:
        list_steps(checkpoint)
        return

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    if args.step:
        plot_step(checkpoint, args.step, save_dir=args.save)
    else:
        plot_all(checkpoint, save_dir=args.save)


if __name__ == '__main__':
    main()
