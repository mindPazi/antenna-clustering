import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# EXTENDED GRID SEARCH ANALYSIS
# ============================================================================


def analyze_extended_results(csv_path, output_dir="./analysis"):
    """
    Analyze extended grid search results with steering angles, array sizes, and patterns
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 90)
    print("EXTENDED GRID SEARCH ANALYSIS")
    print("=" * 90)

    # 1. GLOBAL STATISTICS
    print("\n✓ GLOBAL STATISTICS:")
    print("-" * 90)
    print(f"Total configurations: {len(df)}")
    print(f"Array sizes tested: {df['array_size'].unique()}")
    print(f"Pattern types tested: {df['pattern_type'].unique()}")
    print(
        f"Steering angles tested: {df[['azi0', 'ele0']].drop_duplicates().values.tolist()}"
    )
    print(f"\nSLL range: {df['sll_dB'].min():.2f} to {df['sll_dB'].max():.2f} dB")
    print(f"HPBW range: {df['hpbw_ele'].min():.1f}° to {df['hpbw_ele'].max():.1f}°")

    # 2. BEST DESIGNS BY PATTERN TYPE
    print("\n\n✓ BEST DESIGNS BY PATTERN TYPE (Minimum SLL):")
    print("-" * 90)
    for pattern_type in sorted(df["pattern_type"].unique()):
        df_pattern = df[df["pattern_type"] == pattern_type]
        best = df_pattern.nsmallest(5, "sll_dB")

        print(f"\n{pattern_type}:")
        for idx, row in best.iterrows():
            print(
                f"  SLL={row['sll_dB']:7.2f}dB | "
                f"Array={row['array_size']} | "
                f"Az={row['azi0']:3.0f}° El={row['ele0']:3.0f}° | "
                f"dy={row['dist_y']:.1f}λ dz={row['dist_z']:.1f}λ | "
                f"C={int(row['cluster_size'])}"
            )

    # 3. EFFECT OF PATTERN TYPE
    print("\n\n✓ EFFECT OF PATTERN TYPE:")
    print("-" * 90)
    for pattern_type in sorted(df["pattern_type"].unique()):
        df_pattern = df[df["pattern_type"] == pattern_type]
        print(f"\n{pattern_type}:")
        print(f"  Best SLL:     {df_pattern['sll_dB'].min():7.2f} dB")
        print(f"  Mean SLL:     {df_pattern['sll_dB'].mean():7.2f} dB")
        print(f"  Worst SLL:    {df_pattern['sll_dB'].max():7.2f} dB")
        print(f"  Best HPBW:    {df_pattern['hpbw_ele'].min():7.1f}°")

    # 6. Create comparison visualizations
    print("\n\nGenerating visualizations...")

    steering_pairs = [
        tuple(row) for row in df[["azi0", "ele0"]].drop_duplicates().values
    ]

    # ========== PLOT 1: SLL Comparison by Pattern Type (P=0 vs P=1) ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, pattern_type in enumerate(sorted(df["pattern_type"].unique())):
        df_pattern = df[df["pattern_type"] == pattern_type]

        data_by_steering = []
        for azi0, ele0 in sorted(steering_pairs):
            df_steer = df_pattern[
                (df_pattern["azi0"] == azi0) & (df_pattern["ele0"] == ele0)
            ]
            data_by_steering.append(df_steer["sll_dB"].min())

        axes[i].bar(
            range(len(data_by_steering)), data_by_steering, color="steelblue", alpha=0.7
        )
        axes[i].set_ylabel("Best SLL [dB]", fontweight="bold", fontsize=12)
        axes[i].set_title(f"Pattern: {pattern_type}", fontweight="bold", fontsize=13)
        axes[i].grid(True, alpha=0.3, axis="y")

        steering_labels_short = [
            f"({int(azi0)}°,{int(ele0)}°)" for azi0, ele0 in sorted(steering_pairs)
        ]
        axes[i].set_xticks(range(len(data_by_steering)))
        axes[i].set_xticklabels(steering_labels_short, rotation=45, fontsize=9)

    plt.suptitle(
        "Best SLL Performance by Pattern Type and Steering Angle",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_sll_by_pattern_type.png", dpi=150)
    plt.close()

    # ========== PLOT 2: SLL vs HPBW Trade-off (P=0 Isotropic) ==========
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = {"8x8": "#1f77b4", "16x16": "#ff7f0e"}
    df_isotropic = df[df["pattern_type"] == "Isotropic"]

    for idx, (azi0, ele0) in enumerate(sorted(steering_pairs)):
        ax = axes[idx]
        df_steer = df_isotropic[
            (df_isotropic["azi0"] == azi0) & (df_isotropic["ele0"] == ele0)
        ]

        for array_size in sorted(df["array_size"].unique()):
            df_array = df_steer[df_steer["array_size"] == array_size]
            ax.scatter(
                df_array["sll_dB"],
                df_array["hpbw_ele"],
                s=60,
                alpha=0.7,
                color=colors[array_size],
                label=f"{array_size}",
            )

        ax.set_xlabel("SLL [dB]", fontweight="bold", fontsize=10)
        ax.set_ylabel("HPBW [°]", fontweight="bold", fontsize=10)
        ax.set_title(f"Steering: Az={int(azi0)}°, El={int(ele0)}°", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_visible(False)

    plt.suptitle(
        "SLL vs HPBW Trade-off - ISOTROPIC Pattern (P=0)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2a_tradeoff_isotropic_p0.png", dpi=150)
    plt.close()

    # ========== PLOT 3: SLL vs HPBW Trade-off (P=1 Cosine) ==========
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    df_cosine = df[df["pattern_type"] == "Cosine"]

    for idx, (azi0, ele0) in enumerate(sorted(steering_pairs)):
        ax = axes[idx]
        df_steer = df_cosine[(df_cosine["azi0"] == azi0) & (df_cosine["ele0"] == ele0)]

        for array_size in sorted(df["array_size"].unique()):
            df_array = df_steer[df_steer["array_size"] == array_size]
            ax.scatter(
                df_array["sll_dB"],
                df_array["hpbw_ele"],
                s=60,
                alpha=0.7,
                color=colors[array_size],
                label=f"{array_size}",
            )

        ax.set_xlabel("SLL [dB]", fontweight="bold", fontsize=10)
        ax.set_ylabel("HPBW [°]", fontweight="bold", fontsize=10)
        ax.set_title(f"Steering: Az={int(azi0)}°, El={int(ele0)}°", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_visible(False)

    plt.suptitle(
        "SLL vs HPBW Trade-off - COSINE Pattern (P=1)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2b_tradeoff_cosine_p1.png", dpi=150)
    plt.close()

    # ========== PLOT 4: Direct Comparison P=0 vs P=1 ==========
    fig, ax = plt.subplots(figsize=(14, 8))

    for pattern_type in sorted(df["pattern_type"].unique()):
        df_pattern = df[df["pattern_type"] == pattern_type]
        marker = "o" if pattern_type == "Isotropic" else "s"
        alpha = 0.5 if pattern_type == "Isotropic" else 0.7

        ax.scatter(
            df_pattern["sll_dB"],
            df_pattern["hpbw_ele"],
            s=80,
            alpha=alpha,
            marker=marker,
            label=f"{pattern_type}",
        )

    ax.set_xlabel("SLL [dB]  (Lower = Better →)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "HPBW Elevation [°]  (Lower = Better ↓)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "Pattern Type Comparison: Isotropic (P=0) vs Cosine (P=1)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_comparison_p0_vs_p1.png", dpi=150)
    plt.close()

    # ========== PLOT 5: True Heatmap - SLL by Steering Angle and Array Size ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, pattern_type in enumerate(sorted(df["pattern_type"].unique())):
        df_pattern = df[df["pattern_type"] == pattern_type]

        # Create pivot table: rows=steering angle, cols=array size
        heatmap_data = []
        array_sizes = sorted(df["array_size"].unique())
        steering_angles_sorted = sorted(steering_pairs)

        for azi0, ele0 in steering_angles_sorted:
            row_data = []
            for array_size in array_sizes:
                df_combo = df_pattern[
                    (df_pattern["azi0"] == azi0)
                    & (df_pattern["ele0"] == ele0)
                    & (df_pattern["array_size"] == array_size)
                ]
                best_sll = df_combo["sll_dB"].min() if len(df_combo) > 0 else np.nan
                row_data.append(best_sll)
            heatmap_data.append(row_data)

        heatmap_data = np.array(heatmap_data)

        # Plot heatmap
        im = axes[i].imshow(
            heatmap_data, cmap="RdYlGn", aspect="auto", origin="lower", vmin=-25, vmax=0
        )
        axes[i].set_ylabel("Steering Angle", fontweight="bold")
        axes[i].set_xlabel("Array Size", fontweight="bold")
        axes[i].set_title(
            f"{pattern_type} (P={0 if pattern_type == 'Isotropic' else 1})",
            fontweight="bold",
            fontsize=12,
        )

        # Set ticks
        steering_labels = [
            f"({int(azi0)}°,{int(ele0)}°)" for azi0, ele0 in steering_angles_sorted
        ]
        axes[i].set_yticks(range(len(steering_angles_sorted)))
        axes[i].set_yticklabels(steering_labels, fontsize=9)
        axes[i].set_xticks(range(len(array_sizes)))
        axes[i].set_xticklabels(array_sizes, fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label("Best SLL [dB]", fontweight="bold")

        # Add values in cells
        for y in range(len(steering_angles_sorted)):
            for x in range(len(array_sizes)):
                value = heatmap_data[y, x]
                if not np.isnan(value):
                    text_color = "white" if value < -12 else "black"
                    axes[i].text(
                        x,
                        y,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight="bold",
                    )

    plt.suptitle(
        "SLL Heatmap: Steering Angle vs Array Size (by Pattern Type)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_heatmap_sll_pattern.png", dpi=150)
    plt.close()

    # Save improved CSV with analysis
    df_save = df.copy()
    df_save.to_csv(f"{output_dir}/grid_search_extended_analysis.csv", index=False)

    print("✓ Visualizations saved")

    return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    csv_path = "./results/grid_search_extended.csv"
    output_dir = "./analysis"

    df_extended = analyze_extended_results(csv_path, output_dir)

    print("\n" + "=" * 90)
    print("✓ Analysis complete!")
    print("=" * 90)
