"""
Genetic Algorithm for antenna clustering optimization
Aligned with clustering_comparison.ipynb notebook
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from antenna_physics import (
    AntennaArray,
    LatticeConfig,
    SystemConfig,
    MaskConfig,
    ElementPatternConfig,
)

from antenna_clustering_MC import (
    ClusterConfig,
    FullSubarraySetGeneration,
)


@dataclass
class GAParams:
    """Genetic Algorithm parameters"""
    population_size: int = 50   # FIX: increased from 20 for better convergence
    max_generations: int = 100  # FIX: increased from 15 for better convergence
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_size: int = 5         # FIX: increased from 3
    random_seed: int = None     # FIX: for reproducibility


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for clustering optimization - INDEPENDENT from MC.
    Generates its own subarrays and uses real physics.
    """

    def __init__(self, array: AntennaArray, cluster_config: ClusterConfig,
                 ga_params: GAParams):
        self.array = array
        self.cluster_config = cluster_config
        self.params = ga_params

        # Genera subarrays (come fa il MC)
        self.S_all = []  # Lista di liste di subarrays
        self.N_all = []  # Numero subarrays per tipo

        for cluster_type in cluster_config.Cluster_type:
            gen = FullSubarraySetGeneration(
                cluster_type, array.lattice, array.NN, array.MM,
                cluster_config.rotation_cluster
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

        print(f"GA initialized: {self.total_clusters} subarrays available")

    def _create_random_genes(self) -> np.ndarray:
        """Create a random chromosome"""
        return np.random.randint(0, 2, size=self.total_clusters)

    def _evaluate_genes(self, genes: np.ndarray) -> Dict:
        """Evaluate a chromosome using real physics"""
        # Select active clusters
        Cluster = [self.all_subarrays[i] for i in range(len(genes)) if genes[i] == 1]

        if len(Cluster) == 0:
            return {
                "valid": False, "fitness": -10000,
                "Cm": 99999, "sll_out": 0, "sll_in": 0, "n_clusters": 0
            }

        # Use real physics
        result = self.array.evaluate_clustering(Cluster)

        # Fitness: minimize Cm (cost function) + hardware penalty
        Cm = result["Cm"]
        n_clusters = result["Ntrans"]
        hardware_penalty = (n_clusters / self.array.Nel) * 50

        # Negative fitness because GA maximizes
        fitness = -Cm - hardware_penalty

        return {
            "valid": True,
            "fitness": fitness,
            "Cm": Cm,
            "sll_out": result["sll_out"],
            "sll_in": result["sll_in"],
            "n_clusters": n_clusters,
        }

    def initialize_population(self):
        """Create initial population"""
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
        """Tournament selection"""
        contestants = np.random.choice(len(self.population), tournament_size, replace=False)
        winner_idx = max(contestants, key=lambda i: self.population[i]["fitness"])
        return self.population[winner_idx].copy()

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover"""
        mask = np.random.randint(0, 2, size=self.total_clusters)
        child1_genes = np.where(mask, parent1["genes"], parent2["genes"])
        child2_genes = np.where(mask, parent2["genes"], parent1["genes"])
        return child1_genes, child2_genes

    def _mutate(self, genes: np.ndarray) -> np.ndarray:
        """Bit flip mutation.

        FIX: Copy the array before modifying to avoid side effects.
        """
        genes = genes.copy()  # FIX: avoid in-place modification
        mask = np.random.random(self.total_clusters) < self.params.mutation_rate
        genes[mask] = 1 - genes[mask]
        return genes

    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity"""
        genes_matrix = np.array([ind["genes"] for ind in self.population])
        return np.mean(np.std(genes_matrix, axis=0))

    def run(self, verbose=True):
        """Execute GA"""
        # FIX: Apply seed for reproducibility
        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed)

        if verbose:
            print(f"\n{'='*60}")
            print(f"GENETIC ALGORITHM - ANTENNA CLUSTERING")
            print(f"{'='*60}")
            print(f"Array: {self.array.lattice.Nz}x{self.array.lattice.Ny} elements")
            print(f"Available subarrays: {self.total_clusters}")
            print(f"Population: {self.params.population_size}")
            print(f"Generations: {self.params.max_generations}")
            print(f"{'='*60}\n")

        start_time = time.time()

        # Initialization
        if verbose:
            print("Initializing population...")
        self.initialize_population()
        if verbose:
            print(f"   Completed in {time.time() - start_time:.1f}s\n")

        # Evolution
        for generation in range(self.params.max_generations):
            gen_start = time.time()

            # Sort by fitness
            self.population.sort(key=lambda x: x["fitness"], reverse=True)
            best = self.population[0]
            avg_fitness = np.mean([ind["fitness"] for ind in self.population])
            diversity = self._calculate_diversity()

            # Save history
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
                        print(f"\n[OK] Convergence at generation {generation+1}")
                    break

            # New generation
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

        # Final result
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        self.best_individual = self.population[0]
        self.elapsed_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"FINAL GA RESULT")
            print(f"{'='*60}")
            print(f"Cost Function (Cm): {self.best_individual['Cm']}")
            print(f"SLL outside FoV: {self.best_individual['sll_out']:.2f} dB")
            print(f"SLL inside FoV: {self.best_individual['sll_in']:.2f} dB")
            print(f"Number of clusters: {self.best_individual['n_clusters']}")
            print(f"Total time: {self.elapsed_time:.1f}s")
            print(f"{'='*60}\n")

        return self.best_individual


def main():
    """Execute Genetic Algorithm optimization"""
    lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.6, dist_y=0.53, lattice_type=1)
    system = SystemConfig(freq=29.5e9, azi0=0, ele0=0, dele=0.5, dazi=0.5)
    mask = MaskConfig(elem=30, azim=60, SLL_level=20, SLLin=15)
    eef = ElementPatternConfig(P=1, Gel=5, load_file=0)
    cluster_config = ClusterConfig(Cluster_type=[np.array([[0, 0], [0, 1]])], rotation_cluster=0)

    ga_params = GAParams(
        population_size=15,
        max_generations=10,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=2,
    )

    print("Initializing antenna array...")
    array = AntennaArray(lattice, system, mask, eef)

    ga_optimizer = GeneticAlgorithmOptimizer(array, cluster_config, ga_params)
    best_ga = ga_optimizer.run(verbose=True)

    print(f"\nGA completed in {ga_optimizer.elapsed_time:.1f}s")

    return ga_optimizer


if __name__ == "__main__":
    ga_optimizer = main()
