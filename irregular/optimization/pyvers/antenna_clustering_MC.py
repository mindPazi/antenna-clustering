"""
Antenna clustering optimization - Monte Carlo
Aligned with clustering_comparison.ipynb notebook
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict

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
    random_seed: int = None  # FIX: for reproducibility


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
    """Clustering optimization with Monte Carlo approach + optimizations"""

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

        self.simulation = []
        self.all_Cm = []
        self.all_Ntrans = []
        self.all_Nel = []

        # Adaptive probability tracking
        self.total_clusters = sum(self.N_all)
        self._cluster_scores = np.ones(self.total_clusters)
        self._selection_counts = np.ones(self.total_clusters)

    def _select_random_clusters(self):
        """Original random selection (50% probability)"""
        selected_clusters = []
        selected_rows = []

        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]
            selection = np.random.randint(0, 2, size=Nsub)
            selected_rows.append(selection)
            for idx in np.where(selection == 1)[0]:
                selected_clusters.append(S[idx])

        return selected_clusters, np.concatenate(selected_rows)

    def _select_adaptive_clusters(self):
        """OPT: Selection with adaptive probability"""
        selected_clusters = []
        selected_rows = []

        probs = self._cluster_scores / self._selection_counts
        probs = np.clip(probs, 0.1, 0.9)

        offset = 0
        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]
            cluster_probs = probs[offset : offset + Nsub]
            selection = (np.random.random(Nsub) < cluster_probs).astype(int)
            selected_rows.append(selection)
            for idx in np.where(selection == 1)[0]:
                selected_clusters.append(S[idx])
            offset += Nsub

        return selected_clusters, np.concatenate(selected_rows)

    def _update_adaptive_scores(self, selected_rows: np.ndarray, Cm: int):
        """OPT: Update adaptive scores.

        FIX: Use exponential scaling instead of cutoff.
        Before: reward = max(0, Cost_thr - Cm) -> always 0 if Cm > Cost_thr
        Now: reward = exp(-Cm/Cost_thr) -> always > 0, learns from bad solutions too
        """
        normalized_cm = Cm / self.sim_config.Cost_thr
        reward = np.exp(-normalized_cm)  # Range [0, 1], mai zero

        indices = np.where(selected_rows == 1)[0]
        self._cluster_scores[indices] += reward
        self._selection_counts[indices] += 1

    def _greedy_initialization(self, max_clusters: int = None):
        """OPT: Greedy initialization"""
        if max_clusters is None:
            max_clusters = self.total_clusters // 2

        all_clusters = []
        cluster_to_flat_idx = []
        offset = 0
        for bb, S in enumerate(self.S_all):
            for idx, cluster in enumerate(S):
                all_clusters.append(cluster)
                cluster_to_flat_idx.append(offset + idx)
            offset += self.N_all[bb]

        covered_elements = set()
        selected_flat_indices = []

        available_indices = list(range(len(all_clusters)))
        np.random.shuffle(available_indices)

        for idx in available_indices:
            if len(selected_flat_indices) >= max_clusters:
                break
            cluster = all_clusters[idx]
            cluster_elements = set(tuple(pos) for pos in cluster)
            overlap = cluster_elements & covered_elements
            if len(overlap) == 0:
                selected_flat_indices.append(cluster_to_flat_idx[idx])
                covered_elements.update(cluster_elements)

        selected_rows = np.zeros(self.total_clusters, dtype=int)
        selected_rows[selected_flat_indices] = 1

        selected_clusters = []
        offset = 0
        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]
            for idx in range(Nsub):
                if selected_rows[offset + idx] == 1:
                    selected_clusters.append(S[idx])
            offset += Nsub

        return selected_clusters, selected_rows

    def _rows_to_clusters(self, selected_rows: np.ndarray):
        """OPT: Convert selection array to cluster list"""
        clusters = []
        offset = 0
        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]
            for idx in range(Nsub):
                if selected_rows[offset + idx] == 1:
                    clusters.append(S[idx])
            offset += Nsub
        return clusters

    def _local_search(self, selected_rows: np.ndarray, current_Cm: int,
                      max_iterations: int = 10):
        """OPT: Local search for refinement (reduced from 50 to 10 for speed)"""
        best_rows = selected_rows.copy()
        best_Cm = current_Cm

        for _ in range(max_iterations):
            improved = False
            # OPT: Reduced from 20 to 10 clusters to try for speed
            indices_to_try = np.random.permutation(self.total_clusters)[:min(10, self.total_clusters)]

            for idx in indices_to_try:
                candidate_rows = best_rows.copy()
                candidate_rows[idx] = 1 - candidate_rows[idx]
                clusters = self._rows_to_clusters(candidate_rows)
                if len(clusters) == 0:
                    continue

                result = self.array.evaluate_clustering(clusters)
                candidate_Cm = result["Cm"]

                if candidate_Cm < best_Cm:
                    best_rows = candidate_rows
                    best_Cm = candidate_Cm
                    improved = True
                    break

            if not improved:
                break

        return best_rows, best_Cm

    def run(self, verbose: bool = True) -> Dict:
        """Execute 3-phase optimization"""
        # Apply seed for reproducibility
        if self.sim_config.random_seed is not None:
            np.random.seed(self.sim_config.random_seed)

        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("MONTE CARLO CLUSTERING OPTIMIZATION")
            print(f"  3-Phase approach: Greedy → Random → Adaptive")
            print("=" * 60)
            print(f"Array: {self.array.lattice.Nz}x{self.array.lattice.Ny} = {self.array.Nel} elements")
            print(f"Iterations: {self.sim_config.Niter}")
            print("=" * 60)

        sss = 0
        best_Cm_so_far = float("inf")

        for ij_cont in range(1, self.sim_config.Niter + 1):
            # Progress bar
            if verbose and ij_cont % 10 == 0:
                pct = (ij_cont / self.sim_config.Niter) * 100
                print(f"  [Progress: {ij_cont}/{self.sim_config.Niter} ({pct:.0f}%) | Best Cm: {best_Cm_so_far:.0f}]", end="\r")

            # 3-Phase selection strategy
            if ij_cont <= 10:
                if verbose and ij_cont == 1:
                    print("  >> Phase 1: Greedy initialization (iter 1-10)")
                Cluster, selected_rows = self._greedy_initialization()
            elif ij_cont <= 50:
                if verbose and ij_cont == 11:
                    print("\n  >> Phase 2: Random sampling (iter 11-50)")
                Cluster, selected_rows = self._select_random_clusters()
            else:
                if verbose and ij_cont == 51:
                    print("\n  >> Phase 3: Adaptive sampling (iter 51+)")
                Cluster, selected_rows = self._select_adaptive_clusters()

            if len(Cluster) == 0:
                self.all_Cm.append(float("inf"))
                self.all_Ntrans.append(0)
                self.all_Nel.append(0)
                continue

            result = self.array.evaluate_clustering(Cluster)
            Cm = result["Cm"]
            Ntrans = result["Ntrans"]
            Nel_active = int(np.sum(result["Lsub"]))

            # Local search refinement (every 10 iterations if promising)
            if Cm < best_Cm_so_far * 0.8 and ij_cont % 10 == 0:
                old_Cm = Cm
                selected_rows, Cm = self._local_search(selected_rows, Cm, max_iterations=10)
                if verbose and Cm < old_Cm:
                    print(f"\n  >> Local search: Cm {old_Cm} → {Cm}")
                Cluster = self._rows_to_clusters(selected_rows)
                if len(Cluster) > 0:
                    result = self.array.evaluate_clustering(Cluster)
                    Cm = result["Cm"]
                    Ntrans = result["Ntrans"]
                    Nel_active = int(np.sum(result["Lsub"]))

            # Update adaptive scores
            self._update_adaptive_scores(selected_rows, Cm)

            self.all_Cm.append(Cm)
            self.all_Ntrans.append(Ntrans)
            self.all_Nel.append(Nel_active)

            if Cm < best_Cm_so_far:
                best_Cm_so_far = Cm

            if Cm < self.sim_config.Cost_thr:
                sss += 1
                solution = {
                    "selected_rows": selected_rows.copy(),
                    "Cm": Cm,
                    "Ntrans": Ntrans,
                    "Nel": Nel_active,
                    "sll_in": result["sll_in"],
                    "sll_out": result["sll_out"],
                    "iteration": ij_cont,
                }
                self.simulation.append(solution)

        elapsed_time = time.time() - start_time

        if verbose:
            print("\n")
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
                print(f"  Nel: {best_sol['Nel']}")
                print(f"  SLL out: {best_sol['sll_out']:.2f} dB")
                print(f"  SLL in: {best_sol['sll_in']:.2f} dB")

        return {
            "simulation": self.simulation,
            "all_Cm": self.all_Cm,
            "all_Ntrans": self.all_Ntrans,
            "all_Nel": self.all_Nel,
            "elapsed_time": elapsed_time,
            "n_valid_solutions": len(self.simulation),
        }


def main():
    """Execute Monte Carlo optimization"""
    lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.6, dist_y=0.53, lattice_type=1)
    system = SystemConfig(freq=29.5e9, azi0=0, ele0=0, dele=0.5, dazi=0.5)
    mask = MaskConfig(elem=30, azim=60, SLL_level=20, SLLin=15)
    eef = ElementPatternConfig(P=1, Gel=5, load_file=0)
    cluster_config = ClusterConfig(Cluster_type=[np.array([[0, 0], [0, 1]])], rotation_cluster=0)
    sim_config = SimulationConfig(Niter=1000, Cost_thr=1000)

    print("Initializing antenna array...")
    array = AntennaArray(lattice, system, mask, eef)

    optimizer = IrregularClusteringMonteCarlo(array, cluster_config, sim_config)
    results = optimizer.run(verbose=True)

    return optimizer


if __name__ == "__main__":
    optimizer = main()
