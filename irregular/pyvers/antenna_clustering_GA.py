import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class AntennaConfig:
    """Configurazione array 16x16"""

    Nz: int = 16
    Ny: int = 16
    freq: float = 29.5e9
    dist_z: float = 0.6
    dist_y: float = 0.53


@dataclass
class GAParams:
    """Parametri Genetic Algorithm"""

    population_size: int = 50
    max_generations: int = 25
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_size: int = 5


class ClusterChromosome:
    """Un individuo = una configurazione di clustering"""

    def __init__(self, Nz: int, Ny: int, cluster_type: str = "2x1"):
        self.Nz = Nz
        self.Ny = Ny
        self.cluster_type = cluster_type
        self.genes = self._initialize_random()
        self.fitness = None
        self.sll_out = None
        self.sll_in = None
        self.n_clusters = None

    def _initialize_random(self) -> np.ndarray:
        """Genera configurazione cluster random"""
        if self.cluster_type == "2x1":

            n_possible = (self.Nz // 2) * self.Ny
            genes = np.random.randint(0, 2, size=n_possible)
        else:

            n_possible = self.Nz * self.Ny
            genes = np.random.randint(0, 2, size=n_possible)
        return genes

    def calculate_fitness(self, config: AntennaConfig) -> float:
        """
        Calcola fitness = -SLL (più alto è meglio)
        In realtà qui semplifico, tu poi inserisci il calcolo vero
        """

        self.n_clusters = int(np.sum(self.genes))

        if self.n_clusters == 0:
            self.fitness = -1000
            self.sll_out = 0
            self.sll_in = 0
            return self.fitness

        optimal_clusters = 128
        cluster_penalty = abs(self.n_clusters - optimal_clusters) / optimal_clusters

        base_sll = -18.0
        self.sll_out = base_sll - cluster_penalty * 5 + np.random.randn() * 0.5
        self.sll_in = self.sll_out + np.random.uniform(3, 5)

        hardware_penalty = (self.n_clusters / 256) * 2
        self.fitness = -abs(self.sll_out) - hardware_penalty

        return self.fitness

    def mutate(self, mutation_rate: float):
        """Mutazione: flippa bit random"""
        mask = np.random.random(len(self.genes)) < mutation_rate
        self.genes[mask] = 1 - self.genes[mask]
        self.fitness = None

    def crossover(
        self, other: "ClusterChromosome"
    ) -> Tuple["ClusterChromosome", "ClusterChromosome"]:
        """Crossover: crea 2 figli da 2 genitori"""
        crossover_point = np.random.randint(1, len(self.genes))

        child1 = ClusterChromosome(self.Nz, self.Ny, self.cluster_type)
        child2 = ClusterChromosome(self.Nz, self.Ny, self.cluster_type)

        child1.genes = np.concatenate(
            [self.genes[:crossover_point], other.genes[crossover_point:]]
        )
        child2.genes = np.concatenate(
            [other.genes[:crossover_point], self.genes[crossover_point:]]
        )

        return child1, child2


class GeneticAlgorithm:
    """Algoritmo Genetico per ottimizzazione clustering"""

    def __init__(self, antenna_config: AntennaConfig, ga_params: GAParams):
        self.config = antenna_config
        self.params = ga_params
        self.population = []
        self.best_individual = None
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_sll_out": [],
            "best_sll_in": [],
            "best_n_clusters": [],
            "diversity": [],
        }

    def initialize_population(self):
        """Crea popolazione iniziale random"""
        print("🧬 Inizializzazione popolazione...")
        self.population = [
            ClusterChromosome(self.config.Nz, self.config.Ny)
            for _ in range(self.params.population_size)
        ]

    def evaluate_population(self):
        """Calcola fitness di tutta la popolazione"""
        for individual in self.population:
            if individual.fitness is None:
                individual.calculate_fitness(self.config)

    def selection(self) -> List[ClusterChromosome]:
        """Selezione: Tournament selection"""
        selected = []
        tournament_size = 5

        for _ in range(self.params.population_size - self.params.elite_size):
            tournament = np.random.choice(
                self.population, tournament_size, replace=False
            )
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        return selected

    def calculate_diversity(self) -> float:
        """Calcola diversità genetica (quanti geni diversi)"""
        if len(self.population) < 2:
            return 0.0
        genes_matrix = np.array([ind.genes for ind in self.population])
        diversity = np.mean(np.std(genes_matrix, axis=0))
        return diversity

    def evolve(self):
        """Un ciclo di evoluzione"""

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elite = self.population[: self.params.elite_size]

        selected = self.selection()

        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if np.random.random() < self.params.crossover_rate:
                child1, child2 = selected[i].crossover(selected[i + 1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i], selected[i + 1]])

        for individual in offspring:
            individual.mutate(self.params.mutation_rate)

        self.population = (
            elite + offspring[: self.params.population_size - self.params.elite_size]
        )

    def run(self) -> ClusterChromosome:
        """Esegui algoritmo genetico completo"""
        print(f"\n{'='*60}")
        print(f"🚀 GENETIC ALGORITHM - ANTENNA CLUSTERING OPTIMIZATION")
        print(f"{'='*60}")
        print(
            f"Array: {self.config.Nz}x{self.config.Ny} = {self.config.Nz*self.config.Ny} elementi"
        )
        print(f"Frequenza: {self.config.freq/1e9:.1f} GHz")
        print(f"Popolazione: {self.params.population_size}")
        print(f"Generazioni: {self.params.max_generations}")
        print(f"{'='*60}\n")

        self.initialize_population()
        self.evaluate_population()

        for generation in range(self.params.max_generations):

            self.evaluate_population()

            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            diversity = self.calculate_diversity()

            self.history["best_fitness"].append(best.fitness)
            self.history["avg_fitness"].append(avg_fitness)
            self.history["best_sll_out"].append(best.sll_out)
            self.history["best_sll_in"].append(best.sll_in)
            self.history["best_n_clusters"].append(best.n_clusters)
            self.history["diversity"].append(diversity)

            print(
                f"Gen {generation+1:3d} | "
                f"Best SLL_out: {best.sll_out:6.2f} dB | "
                f"SLL_in: {best.sll_in:6.2f} dB | "
                f"Clusters: {best.n_clusters:3d} | "
                f"Fitness: {best.fitness:7.2f} | "
                f"Diversity: {diversity:.3f}"
            )

            if generation > 10:
                recent_improvement = abs(
                    self.history["best_fitness"][-1] - self.history["best_fitness"][-5]
                )
                if recent_improvement < 0.1:
                    print(f"\n✅ Convergenza raggiunta alla generazione {generation+1}")
                    break

            self.evolve()

        self.best_individual = self.population[0]

        print(f"\n{'='*60}")
        print(f"🏆 RISULTATO FINALE")
        print(f"{'='*60}")
        print(f"SLL fuori FoV: {self.best_individual.sll_out:.2f} dB")
        print(f"SLL dentro FoV: {self.best_individual.sll_in:.2f} dB")
        print(f"Numero cluster: {self.best_individual.n_clusters}")
        print(
            f"Riduzione hardware: {(1 - self.best_individual.n_clusters/256)*100:.1f}%"
        )
        print(f"{'='*60}\n")

        return self.best_individual

    def save_results(self, filename: str = "ga_results.json"):
        """Salva risultati in JSON"""
        results = {
            "config": {
                "Nz": self.config.Nz,
                "Ny": self.config.Ny,
                "freq_GHz": self.config.freq / 1e9,
            },
            "ga_params": {
                "population_size": self.params.population_size,
                "max_generations": self.params.max_generations,
                "mutation_rate": self.params.mutation_rate,
            },
            "best_solution": {
                "sll_out": float(self.best_individual.sll_out),
                "sll_in": float(self.best_individual.sll_in),
                "n_clusters": int(self.best_individual.n_clusters),
                "fitness": float(self.best_individual.fitness),
            },
            "history": {
                k: [float(v) for v in vals] for k, vals in self.history.items()
            },
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Risultati salvati in {filename}")


def main():
    """Esegui ottimizzazione"""

    antenna_config = AntennaConfig(Nz=16, Ny=16, freq=29.5e9, dist_z=0.6, dist_y=0.53)

    ga_params = GAParams(
        population_size=50,
        max_generations=25,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=5,
    )

    ga = GeneticAlgorithm(antenna_config, ga_params)
    best_solution = ga.run()

    ga.save_results("ga_results.json")

    from plot_results import plot_all_results

    plot_all_results(ga.history, best_solution)

    return ga


if __name__ == "__main__":
    ga = main()
