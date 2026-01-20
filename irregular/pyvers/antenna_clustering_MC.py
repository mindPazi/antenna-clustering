"""
Ottimizzazione clustering antenna - Fedele al MATLAB
Tradotto da MATLAB "Generation_code.m"

Implementa l'approccio Monte Carlo del MATLAB originale
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json

from antenna_physics import (
    AntennaArray,
    LatticeConfig,
    SystemConfig,
    MaskConfig,
    ElementPatternConfig,
)


@dataclass
class SimulationConfig:
    """
    Parametri simulazione - come Input_Conf.m

    Niter: Number of iteration (Monte Carlo iterations)
    Cost_thr: Cost function threshold for saving solutions
    """

    Niter: int = 1000  # Number of iteration
    Cost_thr: int = 1000  # Cost function threshold


@dataclass
class ClusterConfig:
    """
    Configurazione cluster - come Input_Conf.m

    Cluster_type: lista di tipi di cluster da usare
    rotation_cluster: flag per rotazione cluster
    """

    Cluster_type: List[np.ndarray] = field(default_factory=list)
    rotation_cluster: int = 0

    def __post_init__(self):
        if not self.Cluster_type:
            # Default: cluster verticale 2x1
            self.Cluster_type = [np.array([[0, 0], [0, 1]])]


class FullSubarraySetGeneration:
    """
    Genera il set completo di subarray possibili
    FEDELE a FullSubarraySet_Generation.m
    """

    def __init__(
        self,
        cluster_type: np.ndarray,
        lattice: LatticeConfig,
        NN: np.ndarray,
        MM: np.ndarray,
        rotation_cluster: int = 0,
    ):
        self.cluster_type = np.atleast_2d(cluster_type)
        self.lattice = lattice
        self.NN = NN
        self.MM = MM
        self.rotation_cluster = rotation_cluster

        self.S, self.Nsub = self._generate()

    def _generate(self) -> Tuple[List[np.ndarray], int]:
        """
        Genera tutte le posizioni possibili per il tipo di cluster
        FEDELE a SubArraySet_Generation.m
        """
        B = self.cluster_type

        A = np.sum(B, axis=0)

        M = self.MM.flatten()
        N = self.NN.flatten()

        if A[0] == 0:  # vertical cluster
            step_M = B.shape[0]
            step_N = 1
        elif A[1] == 0:  # horizontal cluster
            step_N = B.shape[0]
            step_M = 1
        else:  # generic cluster
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

                # Check bounds
                check = not np.any(
                    (Bshift[:, 0] > max_N)
                    | (Bshift[:, 0] < min_N)
                    | (Bshift[:, 1] > max_M)
                    | (Bshift[:, 1] < min_M)
                )

                if check:
                    S.append(Bshift)

        Nsub = len(S)
        return S, Nsub


class IrregularClusteringMonteCarlo:
    """
    Ottimizzazione clustering con approccio Monte Carlo
    FEDELE a Generation_code.m

    Questo è l'approccio ORIGINALE del MATLAB:
    - Loop su Niter iterazioni
    - In ogni iterazione, seleziona random subset di cluster
    - Valuta cost function (punti che eccedono maschera)
    - Salva soluzioni sotto soglia
    """

    def __init__(
        self,
        array: AntennaArray,
        cluster_config: ClusterConfig,
        sim_config: SimulationConfig,
    ):
        self.array = array
        self.cluster_config = cluster_config
        self.sim_config = sim_config

        # Genera tutti i set di subarray possibili per ogni tipo di cluster
        self.S_all = []
        self.N_all = []
        self.L = []

        for bb, cluster_type in enumerate(cluster_config.Cluster_type):
            gen = FullSubarraySetGeneration(
                cluster_type,
                array.lattice,
                array.NN,
                array.MM,
                cluster_config.rotation_cluster,
            )
            self.S_all.append(gen.S)
            self.N_all.append(gen.Nsub)
            self.L.append(cluster_type.shape[0])

        # Risultati
        self.simulation = []  # Lista di soluzioni
        self.all_Cm = []  # Cost function per ogni iterazione
        self.all_Ntrans = []  # Numero di cluster per ogni iterazione
        self.all_Nel = []  # Numero di elementi per ogni iterazione

        # OPT: Adaptive probability array - tracks which clusters appear in good solutions
        self.total_clusters = sum(self.N_all)
        self._cluster_scores = np.ones(self.total_clusters)  # OPT: Initialized uniformly
        self._selection_counts = np.ones(self.total_clusters)  # OPT: Avoid division by zero

    def _select_random_clusters(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Seleziona un sottoinsieme random di cluster
        FEDELE alla logica di Generation_code.m
        """
        selected_clusters = []
        selected_rows = []  # Per tracciare quali righe sono selezionate

        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]

            # Genera selezione binaria random per ogni possibile cluster
            # Ogni cluster ha probabilità 0.5 di essere selezionato
            selection = np.random.randint(0, 2, size=Nsub)
            selected_rows.append(selection)

            # Seleziona i cluster attivi
            for idx in np.where(selection == 1)[0]:
                selected_clusters.append(S[idx])

        # Flatten selected_rows per compatibilità con MATLAB simulation format
        all_selected = np.concatenate(selected_rows)

        return selected_clusters, all_selected

    def _select_adaptive_clusters(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        OPT: Seleziona cluster con probabilità adattiva basata su performance passate.
        Cluster che appaiono in buone soluzioni hanno maggiore probabilità di essere selezionati.
        """
        selected_clusters = []
        selected_rows = []

        # OPT: Compute selection probabilities from historical performance
        probs = self._cluster_scores / self._selection_counts
        probs = np.clip(probs, 0.1, 0.9)  # OPT: Keep probabilities bounded to maintain exploration

        offset = 0
        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]

            # OPT: Use adaptive probabilities instead of uniform 0.5
            cluster_probs = probs[offset : offset + Nsub]
            selection = (np.random.random(Nsub) < cluster_probs).astype(int)
            selected_rows.append(selection)

            for idx in np.where(selection == 1)[0]:
                selected_clusters.append(S[idx])

            offset += Nsub

        all_selected = np.concatenate(selected_rows)
        return selected_clusters, all_selected

    def _update_adaptive_scores(self, selected_rows: np.ndarray, Cm: int):
        """
        OPT: Update adaptive scores based on solution quality.
        Lower cost = better solution = higher future selection probability.
        """
        # OPT: Reward good solutions (low Cm) and penalize bad ones
        reward = max(0, self.sim_config.Cost_thr - Cm) / self.sim_config.Cost_thr
        indices = np.where(selected_rows == 1)[0]

        self._cluster_scores[indices] += reward
        self._selection_counts[indices] += 1

    def _greedy_initialization(self, max_clusters: int = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        OPT: Greedy initialization - builds a solution by iteratively adding
        clusters that provide the best coverage without overlapping.
        This provides a much better starting point than random selection.
        """
        if max_clusters is None:
            # OPT: Target approximately half the maximum possible clusters
            max_clusters = self.total_clusters // 2

        # OPT: Flatten all clusters into a single list with tracking
        all_clusters = []
        cluster_to_flat_idx = []
        offset = 0
        for bb, S in enumerate(self.S_all):
            for idx, cluster in enumerate(S):
                all_clusters.append(cluster)
                cluster_to_flat_idx.append(offset + idx)
            offset += self.N_all[bb]

        # OPT: Track which element positions are already covered
        covered_elements = set()
        selected_flat_indices = []

        # OPT: Greedy selection - prioritize clusters that cover new elements
        available_indices = list(range(len(all_clusters)))
        np.random.shuffle(available_indices)  # OPT: Randomize order for diversity

        for idx in available_indices:
            if len(selected_flat_indices) >= max_clusters:
                break

            cluster = all_clusters[idx]
            # OPT: Convert cluster positions to hashable tuples
            cluster_elements = set(tuple(pos) for pos in cluster)

            # OPT: Check for overlap with already covered elements
            overlap = cluster_elements & covered_elements
            if len(overlap) == 0:  # OPT: No overlap - add this cluster
                selected_flat_indices.append(cluster_to_flat_idx[idx])
                covered_elements.update(cluster_elements)

        # OPT: Build selection array and cluster list
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

    def _local_search(
        self,
        selected_rows: np.ndarray,
        current_Cm: int,
        max_iterations: int = 10,
    ) -> Tuple[np.ndarray, int]:
        """
        OPT: Local search refinement - tries to improve a solution by
        flipping individual cluster selections (add/remove one cluster at a time).
        OPT: Reduced max_iterations from 50 to 10 for speed.
        """
        best_rows = selected_rows.copy()
        best_Cm = current_Cm

        for _ in range(max_iterations):
            improved = False

            # OPT: Try flipping each cluster selection (reduced from 20 to 10 for speed)
            indices_to_try = np.random.permutation(self.total_clusters)[:min(10, self.total_clusters)]

            for idx in indices_to_try:
                # OPT: Create candidate by flipping one bit
                candidate_rows = best_rows.copy()
                candidate_rows[idx] = 1 - candidate_rows[idx]

                # OPT: Reconstruct clusters from selection
                clusters = self._rows_to_clusters(candidate_rows)
                if len(clusters) == 0:
                    continue

                # OPT: Evaluate candidate
                result = self.array.evaluate_clustering(clusters)
                candidate_Cm = result["Cm"]

                # OPT: Accept if better
                if candidate_Cm < best_Cm:
                    best_rows = candidate_rows
                    best_Cm = candidate_Cm
                    improved = True
                    break  # OPT: First improvement strategy for speed

            # OPT: Stop if no improvement found
            if not improved:
                break

        return best_rows, best_Cm

    def _rows_to_clusters(self, selected_rows: np.ndarray) -> List[np.ndarray]:
        """
        OPT: Helper to convert selection array back to cluster list.
        """
        clusters = []
        offset = 0
        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]
            for idx in range(Nsub):
                if selected_rows[offset + idx] == 1:
                    clusters.append(S[idx])
            offset += Nsub
        return clusters

    def run(self, verbose: bool = True, use_optimizations: bool = True) -> Dict:
        """
        Esegue ottimizzazione Monte Carlo
        FEDELE a Generation_code.m loop principale

        OPT: use_optimizations=True enables greedy init, local search, adaptive sampling
        """
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("IRREGULAR CLUSTERING - MONTE CARLO OPTIMIZATION")
            if use_optimizations:
                print("  [OPTIMIZED MODE: greedy init + local search + adaptive]")
            print("=" * 60)
            print(f"Array: {self.array.lattice.Nz}x{self.array.lattice.Ny} = {self.array.Nel} elementi")
            print(f"Frequenza: {self.array.system.freq/1e9:.1f} GHz")
            print(f"Iterazioni: {self.sim_config.Niter}")
            print(f"Cost threshold: {self.sim_config.Cost_thr}")
            print("=" * 60)
            print()

        sss = 0  # Contatore soluzioni valide
        best_Cm_so_far = float("inf")  # OPT: Track best solution for progress reporting

        for ij_cont in range(1, self.sim_config.Niter + 1):
            # OPT: Print progress every 10 iterations for visibility
            if verbose and ij_cont % 10 == 0:
                pct = (ij_cont / self.sim_config.Niter) * 100
                print(f"  [Progresso: {ij_cont}/{self.sim_config.Niter} ({pct:.0f}%)]", end="\r")

            # OPT: Use greedy initialization for first few iterations, then adaptive sampling
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
                # OPT: Switch to adaptive sampling after warmup period
                Cluster, selected_rows = self._select_adaptive_clusters()
            else:
                # Original random selection
                Cluster, selected_rows = self._select_random_clusters()

            if len(Cluster) == 0:
                # Nessun cluster selezionato, skip
                self.all_Cm.append(float("inf"))
                self.all_Ntrans.append(0)
                self.all_Nel.append(0)
                continue

            # Valuta clustering
            result = self.array.evaluate_clustering(Cluster)

            Cm = result["Cm"]
            Ntrans = result["Ntrans"]
            Nel_active = int(np.sum(result["Lsub"]))

            # OPT: Apply local search refinement to promising solutions
            # Only every 10 iterations AND only if Cm is better than 80% of best so far
            if use_optimizations and Cm < best_Cm_so_far * 0.8 and ij_cont % 10 == 0:
                old_Cm = Cm
                selected_rows, Cm = self._local_search(selected_rows, Cm, max_iterations=10)
                if verbose and Cm < old_Cm:
                    print(f"\n  >> Local search migliorato: Cm {old_Cm} -> {Cm}")
                # OPT: Re-evaluate with refined solution
                Cluster = self._rows_to_clusters(selected_rows)
                if len(Cluster) > 0:
                    result = self.array.evaluate_clustering(Cluster)
                    Cm = result["Cm"]
                    Ntrans = result["Ntrans"]
                    Nel_active = int(np.sum(result["Lsub"]))

            # OPT: Update adaptive scores based on solution quality
            if use_optimizations:
                self._update_adaptive_scores(selected_rows, Cm)

            self.all_Cm.append(Cm)
            self.all_Ntrans.append(Ntrans)
            self.all_Nel.append(Nel_active)

            # OPT: Track best solution
            if Cm < best_Cm_so_far:
                best_Cm_so_far = Cm

            # Salva soluzione se sotto soglia
            if Cm < self.sim_config.Cost_thr:
                sss += 1
                # Formato MATLAB: [selected_rows, Cm, Ntrans, Nel]
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

            # Progress report ogni 100 iterazioni
            if verbose and ij_cont % 100 == 0:
                print(f"Iterazione {ij_cont:4d} | "
                      f"Soluzioni valide: {sss:4d} | "
                      f"Ultimo Cm: {Cm:5d} | "
                      f"Best Cm: {best_Cm_so_far:5.0f} | "  # OPT: Show best so far
                      f"Ntrans: {Ntrans:3d}")

        elapsed_time = time.time() - start_time

        if verbose:
            print()
            print("=" * 60)
            print("RISULTATI")
            print("=" * 60)
            print(f"Iterazioni completate: {self.sim_config.Niter}")
            print(f"Soluzioni valide trovate: {len(self.simulation)}")
            print(f"Tempo esecuzione: {elapsed_time:.2f} s")

            if self.simulation:
                # Trova la migliore soluzione
                best_sol = min(self.simulation, key=lambda x: x["Cm"])
                print()
                print("MIGLIORE SOLUZIONE:")
                print(f"  Cm (cost function): {best_sol['Cm']}")
                print(f"  Ntrans (num cluster): {best_sol['Ntrans']}")
                print(f"  Nel (elementi attivi): {best_sol['Nel']}")
                print(f"  SLL out FoV: {best_sol['sll_out']:.2f} dB")
                print(f"  SLL in FoV: {best_sol['sll_in']:.2f} dB")
                print(f"  Iterazione: {best_sol['iteration']}")

            print("=" * 60)

        return {
            "simulation": self.simulation,
            "all_Cm": self.all_Cm,
            "all_Ntrans": self.all_Ntrans,
            "all_Nel": self.all_Nel,
            "elapsed_time": elapsed_time,
            "n_valid_solutions": len(self.simulation),
        }

    def get_best_solution(self) -> Optional[Dict]:
        """Ritorna la migliore soluzione trovata"""
        if not self.simulation:
            return None
        return min(self.simulation, key=lambda x: x["Cm"])

    def reconstruct_clusters(self, solution: Dict) -> List[np.ndarray]:
        """
        Ricostruisce i cluster da una soluzione salvata
        Come in PostProcessing_singlesolution.m
        """
        selected_rows = solution["selected_rows"]

        delta = 0
        clusters = []

        for bb, S in enumerate(self.S_all):
            Nsub = self.N_all[bb]
            vectorrow_bb = selected_rows[delta : delta + Nsub]

            selected_idx = np.where(vectorrow_bb == 1)[0]
            for idx in selected_idx:
                clusters.append(S[idx])

            delta += Nsub

        return clusters

    def save_results(self, filename: str = "mc_results.json"):
        """Salva risultati in JSON"""
        # Converti arrays numpy in liste per JSON
        results = {
            "config": {
                "Nz": self.array.lattice.Nz,
                "Ny": self.array.lattice.Ny,
                "freq_GHz": self.array.system.freq / 1e9,
                "Niter": self.sim_config.Niter,
                "Cost_thr": self.sim_config.Cost_thr,
            },
            "n_valid_solutions": len(self.simulation),
            "solutions": [
                {
                    "selected_rows": sol["selected_rows"].tolist(),
                    "Cm": int(sol["Cm"]),
                    "Ntrans": int(sol["Ntrans"]),
                    "Nel": int(sol["Nel"]),
                    "sll_in": float(sol["sll_in"]),
                    "sll_out": float(sol["sll_out"]),
                    "iteration": int(sol["iteration"]),
                }
                for sol in self.simulation
            ],
            "statistics": {
                "all_Cm": [int(x) if x != float("inf") else -1 for x in self.all_Cm],
                "all_Ntrans": self.all_Ntrans,
                "all_Nel": self.all_Nel,
            },
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Risultati salvati in {filename}")


def main():
    """
    Esegui ottimizzazione Monte Carlo
    FEDELE a Generation_code.m
    """
    # Configurazione array - come Input_Conf.m
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

    # Configurazione cluster - come Input_Conf.m
    # CL.Cluster_type{1} = [0,0;0,1]; % vertical linear cluster
    cluster_config = ClusterConfig(
        Cluster_type=[np.array([[0, 0], [0, 1]])],  # 2x1 verticale
        rotation_cluster=0,
    )

    # Configurazione simulazione
    sim_config = SimulationConfig(
        Niter=1000,
        Cost_thr=1000,
    )

    # Crea array
    print("Inizializzando array antenna...")
    array = AntennaArray(lattice, system, mask, eef)

    # Crea ottimizzatore Monte Carlo
    optimizer = IrregularClusteringMonteCarlo(
        array,
        cluster_config,
        sim_config,
    )

    # Esegui ottimizzazione
    results = optimizer.run(verbose=True)

    # Salva risultati
    optimizer.save_results("mc_results.json")

    # Se ci sono soluzioni valide, valuta la migliore
    best_sol = optimizer.get_best_solution()
    if best_sol:
        print("\nValutazione dettagliata migliore soluzione...")
        clusters = optimizer.reconstruct_clusters(best_sol)
        result = array.evaluate_clustering(clusters)

        print(f"\nDettagli pattern:")
        print(f"  G_boresight: {result['G_boresight']:.2f} dBi")
        print(f"  Max pointing: theta={result['theta_max']:.1f}, phi={result['phi_max']:.1f}")
        print(f"  Scan loss: {result['SL_theta_phi']:.2f} dB")

    return optimizer


if __name__ == "__main__":
    optimizer = main()
