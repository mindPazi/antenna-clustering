"""
Antenna clustering optimization - Monte Carlo
Aligned with clustering_comparison.ipynb notebook
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict

from antenna_physics import (
    AntennaArray,
    LatticeConfig,
    SystemConfig,
    MaskConfig,
    ElementPatternConfig,
)


@dataclass
class SimulationConfig:
    """Simulation parameters"""
    Niter: int = 1000
    Cost_thr: int = 1000


@dataclass
class ClusterConfig:
    """Cluster configuration for free-form clustering.

    Parameters:
    - max_cluster_size: maximum number of elements in a cluster (1 to N).
    - min_cluster_size: minimum number of elements in a cluster (default 1).
    """
    max_cluster_size: int = 3
    min_cluster_size: int = 1


class FreeFormSubarraySetGeneration:
    """Generate ALL possible connected clusters (FREE FORM shapes).

    This class generates every possible connected cluster of elements
    from size min_size to max_size. Connectivity is 4-way (up/down/left/right).

    For a 16x16 array with max_size=4, this generates ~6000 unique clusters.
    """

    def __init__(self, lattice: LatticeConfig, NN: np.ndarray, MM: np.ndarray,
                 max_size: int = 4, min_size: int = 1):
        self.lattice = lattice
        self.NN = NN
        self.MM = MM
        self.max_size = max_size
        self.min_size = min_size

        # Get array bounds
        self.min_N = int(np.min(NN))
        self.max_N = int(np.max(NN))
        self.min_M = int(np.min(MM))
        self.max_M = int(np.max(MM))

        # Generate all connected clusters
        self.S, self.Nsub = self._generate_all_connected()

    def _get_neighbors(self, pos):
        """Get 4-connected neighbors of a position (n, m)"""
        n, m = pos
        neighbors = []
        # Up, Down, Left, Right
        for dn, dm in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nn, nm = n + dn, m + dm
            if self.min_N <= nn <= self.max_N and self.min_M <= nm <= self.max_M:
                neighbors.append((nn, nm))
        return neighbors

    def _generate_all_connected(self):
        """Generate all connected clusters using iterative expansion"""
        all_clusters = set()  # Use set of frozensets for deduplication

        # All valid positions in the array
        all_positions = []
        for n in range(self.min_N, self.max_N + 1):
            for m in range(self.min_M, self.max_M + 1):
                all_positions.append((n, m))

        # Start with single elements
        current_level = [frozenset([pos]) for pos in all_positions]

        # Add single-element clusters if min_size <= 1
        if self.min_size <= 1:
            all_clusters.update(current_level)

        # Iteratively expand to larger sizes
        for size in range(2, self.max_size + 1):
            next_level = set()

            for cluster in current_level:
                # Find all neighbors of the current cluster
                cluster_neighbors = set()
                for pos in cluster:
                    for neighbor in self._get_neighbors(pos):
                        if neighbor not in cluster:
                            cluster_neighbors.add(neighbor)

                # Try adding each neighbor
                for neighbor in cluster_neighbors:
                    new_cluster = frozenset(cluster | {neighbor})
                    if new_cluster not in next_level and new_cluster not in all_clusters:
                        next_level.add(new_cluster)

            # Add clusters of this size if >= min_size
            if size >= self.min_size:
                all_clusters.update(next_level)

            current_level = list(next_level)

        # Convert to list of numpy arrays (sorted for consistency)
        S = []
        for cluster_frozen in all_clusters:
            cluster_list = sorted(list(cluster_frozen))  # Sort for consistency
            cluster_array = np.array(cluster_list)
            S.append(cluster_array)

        # Sort clusters by size then by first element for reproducibility
        S.sort(key=lambda x: (x.shape[0], tuple(x[0])))

        return S, len(S)


print("FreeFormSubarraySetGeneration class defined!")


class IrregularClusteringMonteCarlo:
    """Clustering optimization with Monte Carlo approach.

    Uses 3-phase optimization:
    - Phase 1 (iter 1-10): Greedy initialization
    - Phase 2 (iter 11-50): Random sampling
    - Phase 3 (iter 51+): Adaptive sampling
    Plus local search refinement.
    """

    def __init__(self, array: AntennaArray, cluster_config: ClusterConfig,
                 sim_config: SimulationConfig):
        self.array = array
        self.cluster_config = cluster_config
        self.sim_config = sim_config

        # Generate all possible connected clusters (FREE FORM)
        print(f"[MC] Generating free-form clusters (size {cluster_config.min_cluster_size}-{cluster_config.max_cluster_size})...")
        gen = FreeFormSubarraySetGeneration(
            array.lattice, array.NN, array.MM,
            max_size=cluster_config.max_cluster_size,
            min_size=cluster_config.min_cluster_size
        )
        self.S_all = [gen.S]
        self.N_all = [gen.Nsub]
        self.L = [cluster.shape[0] for cluster in gen.S]
        print(f"[MC] Generated {gen.Nsub} free-form clusters")

        # Build flat list of all clusters for easy access
        self._all_clusters_flat = []
        for S in self.S_all:
            self._all_clusters_flat.extend(S)

        # Pre-compute cluster elements as sets (OPTIMIZATION)
        self._cluster_elements = [
            set(tuple(pos) for pos in cluster)
            for cluster in self._all_clusters_flat
        ]

        self.simulation = []
        self.all_Cm = []
        self.all_Ntrans = []

        # Adaptive probability tracking
        self.total_clusters = sum(self.N_all)
        self._cluster_scores = np.ones(self.total_clusters)
        self._selection_counts = np.ones(self.total_clusters)

    def _remove_overlaps(self, selected_rows: np.ndarray):
        """Remove overlapping clusters using pre-computed element sets."""
        indices = np.where(selected_rows == 1)[0]
        np.random.shuffle(indices)

        covered_elements = set()
        valid_rows = np.zeros_like(selected_rows)
        valid_clusters = []

        for idx in indices:
            cluster_elements = self._cluster_elements[idx]
            if not (cluster_elements & covered_elements):
                valid_rows[idx] = 1
                valid_clusters.append(self._all_clusters_flat[idx])
                covered_elements.update(cluster_elements)

        return valid_clusters, valid_rows

    def _select_random_clusters(self):
        """Random selection (50% probability) with overlap removal."""
        selected_rows = np.random.randint(0, 2, size=self.total_clusters)
        return self._remove_overlaps(selected_rows)

    def _select_adaptive_clusters(self):
        """Adaptive probability selection with overlap removal."""
        probs = self._cluster_scores / self._selection_counts
        probs = np.clip(probs, 0.1, 0.9)
        selected_rows = (np.random.random(self.total_clusters) < probs).astype(int)
        return self._remove_overlaps(selected_rows)

    def _update_adaptive_scores(self, selected_rows: np.ndarray, Cm: int):
        """Update adaptive scores based on solution quality."""
        normalized_cm = Cm / self.sim_config.Cost_thr
        reward = np.exp(-normalized_cm)
        indices = np.where(selected_rows == 1)[0]
        self._cluster_scores[indices] += reward
        self._selection_counts[indices] += 1

    def _greedy_initialization(self, max_clusters: int = None):
        """Greedy initialization - select non-overlapping clusters."""
        if max_clusters is None:
            max_clusters = self.total_clusters // 2

        covered_elements = set()
        selected_indices = []

        available_indices = list(range(self.total_clusters))
        np.random.shuffle(available_indices)

        for idx in available_indices:
            if len(selected_indices) >= max_clusters:
                break
            cluster_elements = self._cluster_elements[idx]
            if not (cluster_elements & covered_elements):
                selected_indices.append(idx)
                covered_elements.update(cluster_elements)

        selected_rows = np.zeros(self.total_clusters, dtype=int)
        selected_rows[selected_indices] = 1
        selected_clusters = [self._all_clusters_flat[i] for i in selected_indices]

        return selected_clusters, selected_rows

    def _rows_to_clusters(self, selected_rows: np.ndarray):
        """Convert selection array to cluster list."""
        return [self._all_clusters_flat[i] for i in np.where(selected_rows == 1)[0]]

    def _local_search(self, selected_rows: np.ndarray, current_Cm: int,
                      max_iterations: int = 10):
        """Local search - flip single bits and check for improvement."""
        best_rows = selected_rows.copy()
        best_Cm = current_Cm

        # Get currently covered elements
        covered = set()
        for idx in np.where(best_rows == 1)[0]:
            covered.update(self._cluster_elements[idx])

        for _ in range(max_iterations):
            improved = False
            indices_to_try = np.random.permutation(self.total_clusters)[:20]

            for idx in indices_to_try:
                candidate_rows = best_rows.copy()

                if candidate_rows[idx] == 1:
                    # Flip 1->0: remove cluster (always valid)
                    candidate_rows[idx] = 0
                else:
                    # Flip 0->1: add cluster (check overlap first)
                    cluster_elements = self._cluster_elements[idx]
                    if cluster_elements & covered:
                        continue  # Would overlap, skip

                    candidate_rows[idx] = 1

                clusters = self._rows_to_clusters(candidate_rows)
                if len(clusters) == 0:
                    continue

                result = self.array.evaluate_clustering(clusters)
                candidate_Cm = result["Cm"]

                if candidate_Cm < best_Cm:
                    best_rows = candidate_rows
                    best_Cm = candidate_Cm
                    # Update covered set
                    covered = set()
                    for i in np.where(best_rows == 1)[0]:
                        covered.update(self._cluster_elements[i])
                    improved = True
                    break

            if not improved:
                break

        return best_rows, best_Cm

    def run(self, verbose: bool = True) -> Dict:
        """Execute 3-phase optimization."""
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("MONTE CARLO CLUSTERING OPTIMIZATION")
            print(f"  Free-form clusters: size {self.cluster_config.min_cluster_size}-{self.cluster_config.max_cluster_size}")
            print(f"  3-Phase approach: Greedy -> Random -> Adaptive")
            print("=" * 60)
            print(f"Array: {self.array.lattice.Nz}x{self.array.lattice.Ny} = {self.array.Nel} elements")
            print(f"Available clusters: {self.total_clusters}")
            print(f"Iterations: {self.sim_config.Niter}")
            print("=" * 60)

        best_Cm_so_far = float("inf")

        for ij_cont in range(1, self.sim_config.Niter + 1):
            if verbose and ij_cont % 50 == 0:
                pct = (ij_cont / self.sim_config.Niter) * 100
                print(f"  [Progress: {ij_cont}/{self.sim_config.Niter} ({pct:.0f}%) | Best Cm: {best_Cm_so_far:.0f}]")

            # 3-Phase selection strategy
            if ij_cont <= 10:
                if verbose and ij_cont == 1:
                    print("  >> Phase 1: Greedy initialization (iter 1-10)")
                Cluster, selected_rows = self._greedy_initialization()
            elif ij_cont <= 50:
                if verbose and ij_cont == 11:
                    print("  >> Phase 2: Random sampling (iter 11-50)")
                Cluster, selected_rows = self._select_random_clusters()
            else:
                if verbose and ij_cont == 51:
                    print("  >> Phase 3: Adaptive sampling (iter 51+)")
                Cluster, selected_rows = self._select_adaptive_clusters()

            if len(Cluster) == 0:
                self.all_Cm.append(float("inf"))
                self.all_Ntrans.append(0)
                continue

            result = self.array.evaluate_clustering(Cluster)
            Cm = result["Cm"]
            Ntrans = result["Ntrans"]

            # Local search refinement (every 25 iterations if promising)
            if Cm < best_Cm_so_far * 0.8 and ij_cont % 25 == 0:
                old_Cm = Cm
                selected_rows, Cm = self._local_search(selected_rows, Cm, max_iterations=10)
                if Cm < old_Cm:
                    Cluster = self._rows_to_clusters(selected_rows)
                    if len(Cluster) > 0:
                        result = self.array.evaluate_clustering(Cluster)
                        Cm = result["Cm"]
                        Ntrans = result["Ntrans"]

            self._update_adaptive_scores(selected_rows, Cm)

            self.all_Cm.append(Cm)
            self.all_Ntrans.append(Ntrans)

            if Cm < best_Cm_so_far:
                best_Cm_so_far = Cm

            if Cm < self.sim_config.Cost_thr:
                solution = {
                    "selected_rows": selected_rows.copy(),
                    "Cm": Cm,
                    "Ntrans": Ntrans,
                    "sll_in": result["sll_in"],
                    "sll_out": result["sll_out"],
                    "iteration": ij_cont,
                }
                self.simulation.append(solution)

        elapsed_time = time.time() - start_time

        if verbose:
            print()
            print("=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Iterations: {self.sim_config.Niter}")
            print(f"Valid solutions: {len(self.simulation)}")
            print(f"Time: {elapsed_time:.2f} s")

            if self.simulation:
                best_sol = min(self.simulation, key=lambda x: x["Cm"])
                print(f"\nBEST SOLUTION:")
                print(f"  Cm: {best_sol['Cm']}")
                print(f"  Ntrans: {best_sol['Ntrans']}")
                print(f"  SLL out: {best_sol['sll_out']:.2f} dB")
                print(f"  SLL in: {best_sol['sll_in']:.2f} dB")

        return {
            "simulation": self.simulation,
            "all_Cm": self.all_Cm,
            "all_Ntrans": self.all_Ntrans,
            "elapsed_time": elapsed_time,
            "n_valid_solutions": len(self.simulation),
        }


def main():
    """Execute Monte Carlo optimization"""
    lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.6, dist_y=0.53, lattice_type=1)
    system = SystemConfig(freq=29.5e9, azi0=0, ele0=0, dele=0.5, dazi=0.5)
    mask = MaskConfig(elem=30, azim=60, SLL_level=20, SLLin=15)
    eef = ElementPatternConfig(P=1, Gel=5, load_file=0)
    cluster_config = ClusterConfig(max_cluster_size=3, min_cluster_size=1)
    sim_config = SimulationConfig(Niter=500, Cost_thr=1000)

    print("Initializing antenna array...")
    array = AntennaArray(lattice, system, mask, eef)

    optimizer = IrregularClusteringMonteCarlo(array, cluster_config, sim_config)
    results = optimizer.run(verbose=True)

    return optimizer


if __name__ == "__main__":
    optimizer = main()
