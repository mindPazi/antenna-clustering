import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import time
import sys

sys.path.insert(0, "..")
from ElementPattern_optimized import ElementPattern
from GenerateLattice_optimized import GenerateLattice
from SubArraySet_Generation_optimized import SubArraySet_Generation

# ============================================================================
# EXTENDED GRID SEARCH WITH STEERING ANGLES AND ARRAY SIZES
# ============================================================================


def extract_metrics(Fopt_dB, azi, ele, azi0, ele0):
    """
    Extract performance metrics from radiation pattern

    INPUT:
    Fopt_dB: 2D radiation pattern [dB] shape (Nele, Nazi)
    azi: azimuth angles [deg]
    ele: elevation angles [deg]
    azi0, ele0: steering angles [deg]

    OUTPUT:
    metrics: dict with SLL, HPBW_ele, HPBW_azi
    """

    # Find steering angle indices
    Iele = np.argmin(np.abs(ele - ele0))
    Iazi = np.argmin(np.abs(azi - azi0))

    # 1. GAIN: maximum of the pattern (dB)
    gain_dB = np.max(Fopt_dB)

    # 2. SLL: Side Lobe Level
    # Extract elevation cut at steering azimuth
    ele_cut = Fopt_dB[:, Iazi]

    # Find main lobe region (within ±10 deg from steering angle)
    main_lobe_idx = np.abs(ele - ele0) <= 10
    ele_cut_sidelobe = ele_cut.copy()
    ele_cut_sidelobe[main_lobe_idx] = -np.inf  # Exclude main lobe

    sll_dB = np.max(ele_cut_sidelobe)

    # 3. HPBW (Half Power Beam Width) - elevation plane
    half_power_dB = gain_dB - 3

    # Find indices where pattern is above -3dB
    above_halfpower = ele_cut >= half_power_dB

    # Find edges of main lobe
    if np.any(above_halfpower):
        edges = np.where(np.diff(above_halfpower.astype(int)) != 0)[0]
        if len(edges) >= 2:
            hpbw_ele = ele[edges[-1]] - ele[edges[0]]
        else:
            hpbw_ele = np.nan
    else:
        hpbw_ele = np.nan

    # 4. HPBW - azimuth plane
    azi_cut = Fopt_dB[Iele, :]
    above_halfpower_azi = azi_cut >= half_power_dB

    if np.any(above_halfpower_azi):
        edges_azi = np.where(np.diff(above_halfpower_azi.astype(int)) != 0)[0]
        if len(edges_azi) >= 2:
            hpbw_azi = azi[edges_azi[-1]] - azi[edges_azi[0]]
        else:
            hpbw_azi = np.nan
    else:
        hpbw_azi = np.nan

    # 5. Merit: Gain - SLL trade-off
    merit = gain_dB - 0.5 * np.abs(sll_dB)

    metrics = {
        "gain_dB": gain_dB,
        "sll_dB": sll_dB,
        "hpbw_ele": hpbw_ele,
        "hpbw_azi": hpbw_azi,
        "merit": merit,
    }

    return metrics


def compute_radiation_pattern(
    dist_y, dist_z, cluster_config, Ny=16, Nz=16, azi0=0, ele0=10, f=29e9, P=1
):
    """
    Compute radiation pattern for given parameters

    INPUT:
    dist_y, dist_z: antenna spacing [times lambda]
    cluster_config: cluster configuration (B matrix)
    Ny, Nz: array dimensions
    azi0, ele0: steering angles [deg]
    f: frequency [Hz]
    P: element pattern type (0=isotropic, 1=cosine)

    OUTPUT:
    Fopt_dB: 2D pattern [dB]
    azi, ele: angle grids
    """

    # Wave parameters
    lambda_ = 3e8 / f
    beta = 2 * np.pi / lambda_

    # Lattice generation
    dz = dist_z * lambda_
    dy = dist_y * lambda_
    x1 = np.array([dy, 0])
    x2 = np.array([0, dz])

    Y, Z, NN, MM, Dy, Dz, ArrayMask, I = GenerateLattice(Ny, Nz, x1, x2)

    # Angle sampling
    dele = 1.0  # Coarser for speed
    dazi = 1.0
    ele = np.arange(-90, 90 + dele, dele)
    azi = np.arange(-90, 90 + dazi, dazi)
    AZI, ELE = np.meshgrid(azi, ele)

    WW = beta * np.cos(np.radians(90 - ELE))
    VV = beta * np.sin(np.radians(90 - ELE)) * np.sin(np.radians(AZI))
    Nw = WW.shape[1]
    Nv = VV.shape[0]

    # Element factor
    Fel = ElementPattern(P, ELE, AZI, 0, "")
    Fel_VW = Fel

    # Cluster setup
    B = cluster_config
    Cluster, Nsub = SubArraySet_Generation(B, NN.flatten(), MM.flatten())

    min_NN = np.min(NN)
    min_MM = np.min(MM)

    Iy_all = (Cluster[::2] - min_NN).astype(int)
    Iz_all = (Cluster[1::2] - min_MM).astype(int)

    Ntrans = Cluster.shape[1]
    Lsub_elements = B.shape[0]
    Iy_all = Iy_all.reshape(Lsub_elements, Ntrans)
    Iz_all = Iz_all.reshape(Lsub_elements, Ntrans)

    Yc = Y[Iz_all, Iy_all]
    Zc = Z[Iz_all, Iy_all]

    Lsub = np.full(Ntrans, Lsub_elements, dtype=int)
    Zc_m = np.mean(Zc, axis=0)
    Yc_m = np.mean(Yc, axis=0)

    # Excitations with steering
    v0 = beta * np.sin(np.radians(90 - ele0)) * np.sin(np.radians(azi0))
    w0 = beta * np.cos(np.radians(90 - ele0))
    Phase_m = np.exp(-1j * (w0 * Zc_m + v0 * Yc_m))
    Amplit_m = np.ones(Ntrans) / Lsub
    c0 = Amplit_m * Phase_m

    # Far field computation
    VV_flat = VV.flatten()[:, np.newaxis]
    WW_flat = WW.flatten()[:, np.newaxis]
    Fel_flat = Fel_VW.flatten()[:, np.newaxis]

    KerFF_sub = np.zeros((Nv * Nw, Ntrans), dtype=complex)
    for jj in range(Lsub_elements):
        phase_term = np.exp(1j * (VV_flat * Yc[jj, :] + WW_flat * Zc[jj, :]))
        KerFF_sub += phase_term * Fel_flat

    FF = KerFF_sub @ c0
    FF_norm = FF / np.max(np.abs(FF))
    FF_norm_2D = FF_norm.reshape(Nv, Nw)
    Fopt_dB = 20 * np.log10(np.abs(FF_norm_2D) + 1e-10)

    return Fopt_dB, azi, ele


def grid_search_extended(
    dist_y_range,
    dist_z_range,
    cluster_sizes,
    steering_angles,
    array_sizes,
    pattern_types,
    output_dir="./grid_search_extended_results",
):
    """
    Extended grid search with steering angles, array sizes, and element patterns

    INPUT:
    dist_y_range: list of dist_y values [times lambda]
    dist_z_range: list of dist_z values [times lambda]
    cluster_sizes: list of cluster sizes
    steering_angles: list of tuples [(azi0, ele0), ...]
    array_sizes: list of tuples [(Ny, Nz), ...]
    pattern_types: list of P values [0=isotropic, 1=cosine, ...]
    output_dir: output directory

    OUTPUT:
    results_df: pandas DataFrame
    """

    import os

    os.makedirs(output_dir, exist_ok=True)

    # Cluster configurations
    cluster_configs = {
        1: np.array([[0, 0]]),
        2: np.array([[0, 0], [0, 1]]),
        3: np.array([[0, 0], [0, 1], [0, 2]]),
        4: np.array([[0, 0], [0, 1], [0, 2], [0, 3]]),
    }

    # Pattern type names
    pattern_names = {0: "Isotropic", 1: "Cosine"}

    results = []
    total_configs = (
        len(dist_y_range)
        * len(dist_z_range)
        * len(cluster_sizes)
        * len(steering_angles)
        * len(array_sizes)
        * len(pattern_types)
    )
    config_count = 0

    print("=" * 90)
    print(f"EXTENDED GRID SEARCH: {total_configs} configurations to evaluate")
    print("=" * 90)
    print(f"Array sizes: {array_sizes}")
    print(f"Steering angles: {steering_angles}")
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Pattern types: {[pattern_names.get(p, f'P={p}') for p in pattern_types]}")
    print(f"Spacing: {len(dist_y_range)}x{len(dist_z_range)} combinations")
    print("=" * 90 + "\n")

    start_time = time.time()

    for array_size in array_sizes:
        Ny, Nz = array_size
        print(f"\n{'='*90}")
        print(f"ARRAY SIZE: {Ny}x{Nz}")
        print(f"{'='*90}")

        for pattern_type in pattern_types:
            pattern_name = pattern_names.get(pattern_type, f"P={pattern_type}")
            print(f"\nPATTERN: {pattern_name}")

            for azi0, ele0 in steering_angles:
                print(f"  STEERING: Az={azi0}°, El={ele0}°")

                for dist_y in dist_y_range:
                    for dist_z in dist_z_range:
                        for cluster_size in cluster_sizes:
                            config_count += 1
                            elapsed = time.time() - start_time
                            eta = (
                                elapsed / config_count * (total_configs - config_count)
                            )

                            if config_count % 100 == 0:
                                print(
                                    f"    [{config_count}/{total_configs}] ETA: {eta/60:.1f} min"
                                )

                            try:
                                B = cluster_configs[cluster_size]

                                # Compute pattern
                                Fopt_dB, azi, ele = compute_radiation_pattern(
                                    dist_y,
                                    dist_z,
                                    B,
                                    Ny=Ny,
                                    Nz=Nz,
                                    azi0=azi0,
                                    ele0=ele0,
                                    P=pattern_type,
                                )

                                # Extract metrics
                                metrics = extract_metrics(Fopt_dB, azi, ele, azi0, ele0)

                                # Store result
                                result_dict = {
                                    "array_size": f"{Ny}x{Nz}",
                                    "Ny": Ny,
                                    "Nz": Nz,
                                    "pattern_type": pattern_name,
                                    "P": pattern_type,
                                    "azi0": azi0,
                                    "ele0": ele0,
                                    "dist_y": dist_y,
                                    "dist_z": dist_z,
                                    "cluster_size": cluster_size,
                                    "gain_dB": metrics["gain_dB"],
                                    "sll_dB": metrics["sll_dB"],
                                    "hpbw_ele": metrics["hpbw_ele"],
                                    "hpbw_azi": metrics["hpbw_azi"],
                                    "merit": metrics["merit"],
                                }
                                results.append(result_dict)

                            except Exception as e:
                                print(
                                    f"      ✗ Error at {Ny}x{Nz}, {pattern_name}, Az={azi0}, El={ele0}, "
                                    f"dy={dist_y:.1f}, dz={dist_z:.1f}, C={cluster_size}: {e}"
                                )
                                continue

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_path = os.path.join(output_dir, "grid_search_extended.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*90}")
    print(f"Grid search completed in {total_time/60:.2f} minutes")
    print(f"Total configs: {len(results_df)}")
    print(f"{'='*90}\n")

    return results_df, output_dir


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # CONFIGURATION
    dist_y_range = np.arange(0.3, 1.0, 0.1)  # 0.3λ to 0.9λ
    dist_z_range = np.arange(0.3, 1.0, 0.1)  # 0.3λ to 0.9λ
    cluster_sizes = [1, 2, 3, 4]  # Different clusters

    # ARRAY SIZES TO TEST
    array_sizes = [(8, 8), (16, 16)]  # 8x8 and 16x16

    # STEERING ANGLES TO TEST
    steering_angles = [
        (0, 0),  # Broadside
        (0, 10),  # Your original
        (0, 30),  # Higher elevation
        (30, 0),  # Azimuth steering
        (30, 10),  # Combined steering
    ]

    # ELEMENT PATTERN TYPES TO TEST
    pattern_types = [0, 1]  # 0=Isotropic, 1=Cosine

    print("\n" + "=" * 90)
    print("EXTENDED GRID SEARCH - STEERING ANGLES + ARRAY SIZES + ELEMENT PATTERNS")
    print("=" * 90)
    print(f"dist_y range: {dist_y_range[0]:.1f}λ to {dist_y_range[-1]:.1f}λ")
    print(f"dist_z range: {dist_z_range[0]:.1f}λ to {dist_z_range[-1]:.1f}λ")
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Array sizes: {array_sizes}")
    print(f"Steering angles: {steering_angles}")
    print(f"Element patterns: Isotropic (P=0), Cosine (P=1)")
    print("=" * 90 + "\n")

    # Run grid search
    results_df, output_dir = grid_search_extended(
        dist_y_range,
        dist_z_range,
        cluster_sizes,
        steering_angles,
        array_sizes,
        pattern_types,
        output_dir="./results",
    )

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 90)
    print(results_df.describe())

    # Print best designs for each configuration
    print("\n\nBEST DESIGNS BY PATTERN TYPE:")
    print("-" * 90)

    for pattern_type in pattern_types:
        df_pattern = results_df[results_df["P"] == pattern_type]
        pattern_name = {0: "Isotropic", 1: "Cosine"}.get(
            pattern_type, f"P={pattern_type}"
        )

        print(f"\n{'*'*90}")
        print(f"PATTERN: {pattern_name}")
        print(f"{'*'*90}")

        best = df_pattern.nsmallest(5, "sll_dB")[
            [
                "array_size",
                "dist_y",
                "dist_z",
                "cluster_size",
                "azi0",
                "ele0",
                "sll_dB",
                "merit",
            ]
        ]
        print(best.to_string(index=False))

    print("\n" + "=" * 90)
    print("✓ Grid search complete!")
    print("=" * 90)
