"""
Genetic Algorithm per ottimizzazione clustering antenna
Usa la stessa fisica reale di antenna_physics.py (come il Monte Carlo)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
import time

# Import fisica antenna reale (stessi import del MC)
from antenna_physics import (
    AntennaArray,
    LatticeConfig,
    SystemConfig,
    MaskConfig,
    ElementPatternConfig,
)


@dataclass
class ClusterConfig:
    """Configurazione cluster - come nel MC"""
    Cluster_type: List[np.ndarray] = field(default_factory=list)
    rotation_cluster: int = 0

    def __post_init__(self):
        if not self.Cluster_type:
            self.Cluster_type = [np.array([[0, 0], [0, 1]])]


@dataclass
class GAParams:
    """Parametri Genetic Algorithm"""
    population_size: int = 50
    max_generations: int = 25
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_size: int = 5


class FullSubarraySetGeneration:
    """
    Genera il set completo di subarray possibili
    IDENTICO a quello nel MC (antenna_clustering_MC.py)
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
        """Genera tutte le posizioni possibili per il tipo di cluster"""
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


class GeneticAlgorithm:
    """
    Algoritmo Genetico per ottimizzazione clustering.
    Usa la stessa fisica reale del Monte Carlo (evaluate_clustering).
    """

    def __init__(
        self,
        array: AntennaArray,
        cluster_config: ClusterConfig,
        ga_params: GAParams,
    ):
        self.array = array
        self.cluster_config = cluster_config
        self.params = ga_params

        # Genera subarrays (IDENTICO al MC)
        self.S_all = []
        self.N_all = []

        for cluster_type in cluster_config.Cluster_type:
            gen = FullSubarraySetGeneration(
                cluster_type,
                array.lattice,
                array.NN,
                array.MM,
                cluster_config.rotation_cluster,
            )
            self.S_all.append(gen.S)
            self.N_all.append(gen.Nsub)

        # Flatten per accesso diretto
        self.all_subarrays = []
        for S_type in self.S_all:
            self.all_subarrays.extend(S_type)

        self.total_clusters = len(self.all_subarrays)
        self.population = []
        self.best_individual = None
        self.elapsed_time = 0

        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_Cm": [],
            "best_sll_out": [],
            "best_sll_in": [],
            "best_n_clusters": [],
            "diversity": [],
        }

        print(f"GA inizializzato: {self.total_clusters} subarrays disponibili")

    def _create_random_genes(self) -> np.ndarray:
        """Crea un cromosoma random (array binario)"""
        return np.random.randint(0, 2, size=self.total_clusters)

    def _genes_to_clusters(self, genes: np.ndarray) -> List[np.ndarray]:
        """Converte genes (array binario) in lista di cluster"""
        return [self.all_subarrays[i] for i in range(len(genes)) if genes[i] == 1]

    def _evaluate_genes(self, genes: np.ndarray) -> Dict:
        """
        Valuta un cromosoma usando fisica reale.
        USA evaluate_clustering() come il MC!
        """
        Cluster = self._genes_to_clusters(genes)

        if len(Cluster) == 0:
            return {
                "valid": False,
                "fitness": -10000,
                "Cm": 99999,
                "sll_out": 0,
                "sll_in": 0,
                "n_clusters": 0,
            }

        # Usa fisica reale (IDENTICO al MC)
        result = self.array.evaluate_clustering(Cluster)

        Cm = result["Cm"]
        n_clusters = result["Ntrans"]
        
        # Fitness: minimizza Cm + penalita hardware
        hardware_penalty = (n_clusters / self.array.Nel) * 50
        fitness = -Cm - hardware_penalty  # Negativo perche GA massimizza

        return {
            "valid": True,
            "fitness": fitness,
            "Cm": Cm,
            "sll_out": result["sll_out"],
            "sll_in": result["sll_in"],
            "n_clusters": n_clusters,
        }

    def initialize_population(self):
        """Crea popolazione iniziale"""
        self.population = []
        for _ in range(self.params.population_size):
            genes = self._create_random_genes()
            result = self._evaluate_genes(genes)
            self.population.append({
                "genes": genes,
                "fitness": result["fitness"],
                "Cm": result["Cm"],
                "sll_out": result["sll_out"],
                "sll_in": result["sll_in"],
                "n_clusters": result["n_clusters"],
            })

    def _tournament_selection(self, tournament_size: int = 3) -> Dict:
        """Selezione per torneo"""
        contestants = np.random.choice(len(self.population), tournament_size, replace=False)
        winner_idx = max(contestants, key=lambda i: self.population[i]["fitness"])
        return self.population[winner_idx].copy()

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover uniforme"""
        mask = np.random.randint(0, 2, size=self.total_clusters)
        child1_genes = np.where(mask, parent1["genes"], parent2["genes"])
        child2_genes = np.where(mask, parent2["genes"], parent1["genes"])
        return child1_genes, child2_genes

    def _mutate(self, genes: np.ndarray) -> np.ndarray:
        """Mutazione bit flip"""
        genes = genes.copy()
        mask = np.random.random(self.total_clusters) < self.params.mutation_rate
        genes[mask] = 1 - genes[mask]
        return genes

    def _calculate_diversity(self) -> float:
        """Calcola diversita genetica"""
        genes_matrix = np.array([ind["genes"] for ind in self.population])
        return np.mean(np.std(genes_matrix, axis=0))

    def run(self, verbose: bool = True) -> Dict:
        """Esegui GA"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"GENETIC ALGORITHM - ANTENNA CLUSTERING")
            print(f"{'='*60}")
            print(f"Array: {self.array.lattice.Nz}x{self.array.lattice.Ny} elementi")
            print(f"Subarrays disponibili: {self.total_clusters}")
            print(f"Popolazione: {self.params.population_size}")
            print(f"Generazioni: {self.params.max_generations}")
            print(f"{'='*60}\n")

        start_time = time.time()

        # Inizializzazione
        if verbose:
            print("Inizializzazione popolazione...")
        self.initialize_population()
        if verbose:
            print(f"   Completata in {time.time() - start_time:.1f}s\n")

        # Evoluzione
        for generation in range(self.params.max_generations):
            gen_start = time.time()

            # Ordina per fitness
            self.population.sort(key=lambda x: x["fitness"], reverse=True)
            best = self.population[0]
            avg_fitness = np.mean([ind["fitness"] for ind in self.population])
            diversity = self._calculate_diversity()

            # Salva storia
            self.history["best_fitness"].append(best["fitness"])
            self.history["avg_fitness"].append(avg_fitness)
            self.history["best_Cm"].append(best["Cm"])
            self.history["best_sll_out"].append(best["sll_out"])
            self.history["best_sll_in"].append(best["sll_in"])
            self.history["best_n_clusters"].append(best["n_clusters"])
            self.history["diversity"].append(diversity)

            gen_time = time.time() - gen_start

            if verbose:
                print(
                    f"Gen {generation+1:3d}/{self.params.max_generations} | "
                    f"Cm: {best['Cm']:4d} | "
                    f"SLL_out: {best['sll_out']:6.2f} dB | "
                    f"Clusters: {best['n_clusters']:3d} | "
                    f"Time: {gen_time:.1f}s"
                )

            # Early stopping
            if generation > 5:
                recent = self.history["best_Cm"][-5:]
                if max(recent) - min(recent) < 5:
                    if verbose:
                        print(f"\nConvergenza alla generazione {generation+1}")
                    break

            # Nuova generazione
            elite = self.population[:self.params.elite_size]
            new_population = [e.copy() for e in elite]

            while len(new_population) < self.params.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                if np.random.random() < self.params.crossover_rate:
                    child1_genes, child2_genes = self._crossover(parent1, parent2)
                else:
                    child1_genes = parent1["genes"].copy()
                    child2_genes = parent2["genes"].copy()

                child1_genes = self._mutate(child1_genes)
                child2_genes = self._mutate(child2_genes)

                for genes in [child1_genes, child2_genes]:
                    if len(new_population) < self.params.population_size:
                        result = self._evaluate_genes(genes)
                        new_population.append({
                            "genes": genes,
                            "fitness": result["fitness"],
                            "Cm": result["Cm"],
                            "sll_out": result["sll_out"],
                            "sll_in": result["sll_in"],
                            "n_clusters": result["n_clusters"],
                        })

            self.population = new_population

        # Risultato finale
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        self.best_individual = self.population[0]
        self.elapsed_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"RISULTATO FINALE GA")
            print(f"{'='*60}")
            print(f"Cost Function (Cm): {self.best_individual['Cm']}")
            print(f"SLL fuori FoV: {self.best_individual['sll_out']:.2f} dB")
            print(f"SLL dentro FoV: {self.best_individual['sll_in']:.2f} dB")
            print(f"Numero cluster: {self.best_individual['n_clusters']}")
            print(f"Tempo totale: {self.elapsed_time:.1f}s")
            print(f"{'='*60}\n")

        return {
            "best": self.best_individual,
            "history": self.history,
            "elapsed_time": self.elapsed_time,
        }

    def save_results(self, filename: str = "ga_results.json"):
        """Salva risultati in JSON"""
        results = {
            "config": {
                "Nz": self.array.lattice.Nz,
                "Ny": self.array.lattice.Ny,
                "freq_GHz": self.array.system.freq / 1e9,
            },
            "ga_params": {
                "population_size": self.params.population_size,
                "max_generations": self.params.max_generations,
                "mutation_rate": self.params.mutation_rate,
            },
            "best_solution": {
                "Cm": int(self.best_individual["Cm"]),
                "sll_out": float(self.best_individual["sll_out"]),
                "sll_in": float(self.best_individual["sll_in"]),
                "n_clusters": int(self.best_individual["n_clusters"]),
                "fitness": float(self.best_individual["fitness"]),
            },
            "history": {
                k: [float(v) for v in vals] for k, vals in self.history.items()
            },
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Risultati salvati in {filename}")


def main():
    """Esegui ottimizzazione GA"""
    # Configurazione (IDENTICA al MC)
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
        Cluster_type=[np.array([[0, 0], [0, 1]])],
        rotation_cluster=0,
    )

    ga_params = GAParams(
        population_size=20,
        max_generations=15,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=3,
    )

    # Crea array
    print("Inizializzando array antenna...")
    array = AntennaArray(lattice, system, mask, eef)

    # Crea e esegui GA
    ga = GeneticAlgorithm(array, cluster_config, ga_params)
    results = ga.run(verbose=True)

    # Salva risultati
    ga.save_results("ga_results.json")

    return ga


if __name__ == "__main__":
    ga = main()
