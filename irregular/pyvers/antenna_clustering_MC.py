"""
Ottimizzazione clustering antenna - Monte Carlo
Allineato al notebook clustering_comparison.ipynb
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
    """Parametri simulazione"""
    Niter: int = 1000
    Cost_thr: int = 1000


@dataclass
class ClusterConfig:
    """Configurazione cluster"""
    Cluster_type: List[np.ndarray] = field(default_factory=list)
    rotation_cluster: int = 0

    def __post_init__(self):
        if not self.Cluster_type:
            self.Cluster_type = [np.array([[0, 0], [0, 1]])]


class FullSubarraySetGeneration:
    """Genera il set completo di subarray possibili"""

    def __init__(self, cluster_type: np.ndarray, lattice: LatticeConfig,
                 NN: np.ndarray, MM: np.ndarray, rotation_cluster: int = 0):
        self.cluster_type = np.atleast_2d(cluster_type)
        self.lattice = lattice
        self.NN = NN
        self.MM = MM
        self.rotation_cluster = rotation_cluster
        self.S, self.Nsub = self._generate()

    def _generate(self):
        B = self.cluster_type
        A = np.sum(B, axis=0)
        M = self.MM.flatten()
        N = self.NN.flatten()

        if A[0] == 0:
            step_M = B.shape[0]
            step_N = 1
        elif A[1] == 0:
            step_N = B.shape[0]
            step_M = 1
        else:
            step_M = 1
            step_N = 1

        S = []
        min_M, max_M = int(np.min(M)), int(np.max(M))
        min_N, max_N = int(np.min(N)), int(np.max(N))

        for kk in range(min_M, max_M + 1, step_M):
            for hh in range(min_N, max_N + 1, step_N):
                Bshift = B.copy()
                Bshift[:, 0] = B[:, 0] + hh
                Bshift[:, 1] = B[:, 1] + kk

                check = not np.any(
                    (Bshift[:, 0] > max_N) | (Bshift[:, 0] < min_N) |
                    (Bshift[:, 1] > max_M) | (Bshift[:, 1] < min_M)
                )

                if check:
                    S.append(Bshift)

        return S, len(S)


class IrregularClusteringMonteCarlo:
    """Ottimizzazione clustering con approccio Monte Carlo + ottimizzazioni"""

    def __init__(self, array: AntennaArray, cluster_config: ClusterConfig,
                 sim_config: SimulationConfig):
        self.array = array
        self.cluster_config = cluster_config
        self.sim_config = sim_config

        self.S_all = []
        self.N_all = []
        self.L = []

        for bb, cluster_type in enumerate(cluster_config.Cluster_type):
            gen = FullSubarraySetGeneration(
                cluster_type, array.lattice, array.NN, array.MM,
                cluster_config.rotation_cluster
            )
            self.S_all.append(gen.S)
            self.N_all.append(gen.Nsub)
            self.L.append(cluster_type.shape[0])

        self.simulation = []
        self.all_Cm = []
        self.all_Ntrans = []
        self.all_Nel = []

        # OPT: Adaptive probability tracking
        self.total_clusters = sum(self.N_all)
        self._cluster_scores = np.ones(self.total_clusters)
        self._selection_counts = np.ones(self.total_clusters)

    def _select_random_clusters(self):
        """Selezione random originale (probabilità 50%)"""
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
        """OPT: Selezione con probabilità adattiva"""
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
        """OPT: Aggiorna score adattivi"""
        reward = max(0, self.sim_config.Cost_thr - Cm) / self.sim_config.Cost_thr
        indices = np.where(selected_rows == 1)[0]
        self._cluster_scores[indices] += reward
        self._selection_counts[indices] += 1

    def _greedy_initialization(self, max_clusters: int = None):
        """OPT: Inizializzazione greedy"""
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
        """OPT: Converte array selezione in lista cluster"""
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
        """OPT: Local search per raffinamento (reduced from 50 to 10 for speed)"""
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

    def run(self, verbose: bool = True, use_optimizations: bool = True) -> Dict:
        """Esegue ottimizzazione"""
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("IRREGULAR CLUSTERING - MONTE CARLO OPTIMIZATION")
            if use_optimizations:
                print("  [OPTIMIZED MODE: greedy init + local search + adaptive]")
            else:
                print("  [ORIGINAL MODE: random sampling only]")
            print("=" * 60)
            print(f"Array: {self.array.lattice.Nz}x{self.array.lattice.Ny} = {self.array.Nel} elementi")
            print(f"Iterazioni: {self.sim_config.Niter}")
            print("=" * 60)

        sss = 0
        best_Cm_so_far = float("inf")

        for ij_cont in range(1, self.sim_config.Niter + 1):
            # Progress bar
            if verbose and ij_cont % 10 == 0:
                pct = (ij_cont / self.sim_config.Niter) * 100
                print(f"  [Progresso: {ij_cont}/{self.sim_config.Niter} ({pct:.0f}%) | Best Cm: {best_Cm_so_far:.0f}]", end="\r")

            # Selezione cluster
            if use_optimizations and ij_cont <= 10:
                if verbose and ij_cont == 1:
                    print("  >> Fase 1: Greedy initialization (iter 1-10)")
                Cluster, selected_rows = self._greedy_initialization()
            elif use_optimizations and ij_cont == 11:
                if verbose:
                    print("\n  >> Fase 2: Random sampling (iter 11-50)")
                Cluster, selected_rows = self._select_random_clusters()
            elif use_optimizations and ij_cont == 51:
                if verbose:
                    print("\n  >> Fase 3: Adaptive sampling (iter 51+)")
                Cluster, selected_rows = self._select_adaptive_clusters()
            elif use_optimizations and ij_cont > 50:
                Cluster, selected_rows = self._select_adaptive_clusters()
            else:
                Cluster, selected_rows = self._select_random_clusters()

            if len(Cluster) == 0:
                self.all_Cm.append(float("inf"))
                self.all_Ntrans.append(0)
                self.all_Nel.append(0)
                continue

            result = self.array.evaluate_clustering(Cluster)
            Cm = result["Cm"]
            Ntrans = result["Ntrans"]
            Nel_active = int(np.sum(result["Lsub"]))

            # OPT: Local search: only every 10 iterations AND only if Cm < 80% of best so far
            if use_optimizations and Cm < best_Cm_so_far * 0.8 and ij_cont % 10 == 0:
                old_Cm = Cm
                selected_rows, Cm = self._local_search(selected_rows, Cm, max_iterations=10)
                if verbose and Cm < old_Cm:
                    print(f"\n  >> Local search: Cm {old_Cm} -> {Cm}")
                Cluster = self._rows_to_clusters(selected_rows)
                if len(Cluster) > 0:
                    result = self.array.evaluate_clustering(Cluster)
                    Cm = result["Cm"]
                    Ntrans = result["Ntrans"]
                    Nel_active = int(np.sum(result["Lsub"]))

            if use_optimizations:
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
            print("RISULTATI")
            print("=" * 60)
            print(f"Iterazioni: {self.sim_config.Niter}")
            print(f"Soluzioni valide: {len(self.simulation)}")
            print(f"Tempo: {elapsed_time:.2f} s")

            if self.simulation:
                best_sol = min(self.simulation, key=lambda x: x["Cm"])
                print(f"\nMIGLIORE SOLUZIONE:")
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
    """Esegui ottimizzazione Monte Carlo"""
    lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.6, dist_y=0.53, lattice_type=1)
    system = SystemConfig(freq=29.5e9, azi0=0, ele0=0, dele=0.5, dazi=0.5)
    mask = MaskConfig(elem=30, azim=60, SLL_level=20, SLLin=15)
    eef = ElementPatternConfig(P=1, Gel=5, load_file=0)
    cluster_config = ClusterConfig(Cluster_type=[np.array([[0, 0], [0, 1]])], rotation_cluster=0)
    sim_config = SimulationConfig(Niter=1000, Cost_thr=1000)

    print("Inizializzando array antenna...")
    array = AntennaArray(lattice, system, mask, eef)

    optimizer = IrregularClusteringMonteCarlo(array, cluster_config, sim_config)
    results = optimizer.run(verbose=True)

    return optimizer


if __name__ == "__main__":
    optimizer = main()
