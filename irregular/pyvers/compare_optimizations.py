"""
Comparison Script: Before vs After Optimizations
=================================================

This script compares the original Monte Carlo approach (use_optimizations=False)
with the optimized version (use_optimizations=True).

Run this script to see the performance difference.

Usage:
    python compare_optimizations.py

The script will run both versions with the same configuration and print:
- Execution time comparison
- Number of valid solutions found
- Best cost function (Cm) achieved
- Best SLL values achieved
"""

import numpy as np
import time
import json
from antenna_physics import (
    AntennaArray,
    LatticeConfig,
    SystemConfig,
    MaskConfig,
    ElementPatternConfig,
)
from antenna_clustering_MC import (
    IrregularClusteringMonteCarlo,
    ClusterConfig,
    SimulationConfig,
)


def run_comparison(Niter: int = 200, seed: int = 42):
    """
    Run comparison between original and optimized algorithms.

    Parameters:
    -----------
    Niter : int
        Number of iterations for each run (default: 200 for quick comparison)
    seed : int
        Random seed for reproducibility
    """

    print("=" * 70)
    print("COMPARISON: ORIGINAL vs OPTIMIZED CLUSTERING ALGORITHM")
    print("=" * 70)
    print()

    # Configuration (same as main.py)
    lattice = LatticeConfig(
        Nz=16,
        Ny=16,
        dist_z=0.6,
        dist_y=0.53,
        lattice_type=1,
    )

    system = SystemConfig(
        freq=29.5e9,
        azi0=0,
        ele0=0,
        dele=0.5,
        dazi=0.5,
    )

    mask = MaskConfig(
        elem=30,
        azim=60,
        SLL_level=20,
        SLLin=15,
    )

    eef = ElementPatternConfig(
        P=1,
        Gel=5,
        load_file=0,
    )

    cluster_config = ClusterConfig(
        Cluster_type=[np.array([[0, 0], [0, 1]])],  # 2x1 vertical cluster
        rotation_cluster=0,
    )

    sim_config = SimulationConfig(
        Niter=Niter,
        Cost_thr=1000,
    )

    print(f"Configuration:")
    print(f"  Array size: {lattice.Nz}x{lattice.Ny} = {lattice.Nz * lattice.Ny} elements")
    print(f"  Frequency: {system.freq/1e9:.1f} GHz")
    print(f"  Iterations per test: {Niter}")
    print(f"  Cost threshold: {sim_config.Cost_thr}")
    print()

    # Create array once (shared)
    print("=" * 70)
    print("STEP 1: Inizializzazione array antenna...")
    print("=" * 70)
    array = AntennaArray(lattice, system, mask, eef)
    print("  Array inizializzato con successo!")
    print()

    # ==================== RUN ORIGINAL (NO OPTIMIZATIONS) ====================
    print("=" * 70)
    print("STEP 2: RUNNING ORIGINAL ALGORITHM (use_optimizations=False)")
    print("        Questo usa solo selezione random con probabilità 50%")
    print("=" * 70)
    np.random.seed(seed)

    print("  Creazione optimizer...")
    optimizer_original = IrregularClusteringMonteCarlo(
        array, cluster_config, sim_config
    )
    print("  Optimizer creato. Avvio ottimizzazione...\n")

    start_original = time.time()
    results_original = optimizer_original.run(verbose=True, use_optimizations=False)
    time_original = time.time() - start_original
    print(f"\n  COMPLETATO in {time_original:.2f} secondi")

    # ==================== RUN OPTIMIZED ====================
    print()
    print("=" * 70)
    print("STEP 3: RUNNING OPTIMIZED ALGORITHM (use_optimizations=True)")
    print("        Questo usa greedy init + local search + adaptive sampling")
    print("=" * 70)
    np.random.seed(seed)  # Same seed for fair comparison

    print("  Creazione optimizer...")
    optimizer_optimized = IrregularClusteringMonteCarlo(
        array, cluster_config, sim_config
    )
    print("  Optimizer creato. Avvio ottimizzazione...\n")

    start_optimized = time.time()
    results_optimized = optimizer_optimized.run(verbose=True, use_optimizations=True)
    time_optimized = time.time() - start_optimized
    print(f"\n  COMPLETATO in {time_optimized:.2f} secondi")

    # ==================== COMPARISON SUMMARY ====================
    print()
    print("=" * 70)
    print("STEP 4: COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print("  Elaborazione risultati in corso...")
    print()

    # Helper to get best solution stats
    def get_best_stats(results):
        if results["n_valid_solutions"] == 0:
            return None
        best = min(results["simulation"], key=lambda x: x["Cm"])
        return best

    best_original = get_best_stats(results_original)
    best_optimized = get_best_stats(results_optimized)

    print(f"{'Metric':<30} {'Original':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-" * 75)

    # Execution time
    print(f"{'Execution time (s)':<30} {time_original:>15.2f} {time_optimized:>15.2f} {(time_original/time_optimized - 1)*100:>14.1f}%")

    # Valid solutions
    n_orig = results_original["n_valid_solutions"]
    n_opt = results_optimized["n_valid_solutions"]
    if n_orig > 0:
        pct_improvement = (n_opt / n_orig - 1) * 100
        print(f"{'Valid solutions found':<30} {n_orig:>15d} {n_opt:>15d} {pct_improvement:>14.1f}%")
    else:
        print(f"{'Valid solutions found':<30} {n_orig:>15d} {n_opt:>15d} {'N/A':>15}")

    # Best cost function
    if best_original and best_optimized:
        cm_orig = best_original["Cm"]
        cm_opt = best_optimized["Cm"]
        improvement = (cm_orig - cm_opt) / cm_orig * 100 if cm_orig > 0 else 0
        print(f"{'Best Cm (lower=better)':<30} {cm_orig:>15d} {cm_opt:>15d} {improvement:>14.1f}%")

        # SLL values
        sll_out_orig = best_original["sll_out"]
        sll_out_opt = best_optimized["sll_out"]
        print(f"{'Best SLL out FoV (dB)':<30} {sll_out_orig:>15.2f} {sll_out_opt:>15.2f}")

        sll_in_orig = best_original["sll_in"]
        sll_in_opt = best_optimized["sll_in"]
        print(f"{'Best SLL in FoV (dB)':<30} {sll_in_orig:>15.2f} {sll_in_opt:>15.2f}")

        # Number of clusters in best solution
        print(f"{'Best Ntrans (clusters)':<30} {best_original['Ntrans']:>15d} {best_optimized['Ntrans']:>15d}")
        print(f"{'Best Nel (elements)':<30} {best_original['Nel']:>15d} {best_optimized['Nel']:>15d}")
    else:
        print("Could not compare best solutions - one or both algorithms found no valid solutions")

    print()
    print("=" * 70)
    print()

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 70)
    if best_optimized and (not best_original or best_optimized["Cm"] < best_original["Cm"]):
        print("  The OPTIMIZED algorithm found a BETTER solution (lower Cm).")
    elif best_original and best_optimized and best_optimized["Cm"] == best_original["Cm"]:
        print("  Both algorithms found solutions with the SAME quality.")
    else:
        print("  The original algorithm found a better solution this time.")
        print("  (This can happen with random elements; try with more iterations)")

    if n_opt > n_orig:
        print(f"  The OPTIMIZED algorithm found {n_opt - n_orig} MORE valid solutions.")

    print()
    print("KEY OPTIMIZATIONS APPLIED:")
    print("  1. NumPy vectorization in kernel computation (OPT: comments in antenna_physics.py)")
    print("  2. Greedy initialization for better starting points (first 10 iterations)")
    print("  3. Local search refinement for promising solutions")
    print("  4. Adaptive probability sampling based on solution quality")
    print()

    # ==================== SAVE RESULTS TO JSON ====================
    def extract_stats(results):
        """Extract statistics lists from results for plotting."""
        all_Cm = [r["Cm"] for r in results["simulation"]]
        all_Ntrans = [r["Ntrans"] for r in results["simulation"]]
        all_Nel = [r["Nel"] for r in results["simulation"]]
        all_sll_out = [r["sll_out"] for r in results["simulation"]]
        all_sll_in = [r["sll_in"] for r in results["simulation"]]
        return {
            "all_Cm": all_Cm,
            "all_Ntrans": all_Ntrans,
            "all_Nel": all_Nel,
            "all_sll_out": all_sll_out,
            "all_sll_in": all_sll_in,
        }

    # Save original results
    json_data_original = {
        "algorithm": "original",
        "n_valid_solutions": results_original["n_valid_solutions"],
        "execution_time": time_original,
        "statistics": extract_stats(results_original),
    }
    with open("mc_results_original.json", "w") as f:
        json.dump(json_data_original, f, indent=2)
    print("Results saved to: mc_results_original.json")

    # Save optimized results
    json_data_optimized = {
        "algorithm": "optimized",
        "n_valid_solutions": results_optimized["n_valid_solutions"],
        "execution_time": time_optimized,
        "statistics": extract_stats(results_optimized),
    }
    with open("mc_results_optimized.json", "w") as f:
        json.dump(json_data_optimized, f, indent=2)
    print("Results saved to: mc_results_optimized.json")

    # Save combined comparison (compatible with plot_results.py)
    json_data_combined = {
        "statistics": extract_stats(results_optimized),  # Default to optimized for plotting
        "comparison": {
            "original": json_data_original,
            "optimized": json_data_optimized,
        }
    }
    with open("mc_results.json", "w") as f:
        json.dump(json_data_combined, f, indent=2)
    print("Combined results saved to: mc_results.json (compatible with plot_results.py)")
    print()

    return results_original, results_optimized


def main():
    """
    Main entry point for comparison.
    """
    print()
    print("ANTENNA CLUSTERING OPTIMIZATION - COMPARISON TEST")
    print("=" * 70)
    print()
    print("This script compares the clustering quality between:")
    print("  - ORIGINAL: Pure Monte Carlo random sampling")
    print("  - OPTIMIZED: Greedy init + Local search + Adaptive sampling")
    print()
    print("For a quick test, using 200 iterations per algorithm.")
    print("For production, increase Niter to 1000+ for better statistics.")
    print()

    # Run with fewer iterations for quick comparison
    results_orig, results_opt = run_comparison(Niter=200, seed=42)

    print("To run with more iterations:")
    print("  python -c \"from compare_optimizations import run_comparison; run_comparison(Niter=1000)\"")
    print()
    print("To plot the results:")
    print("  python plot_results.py")
    print()


if __name__ == "__main__":
    main()
