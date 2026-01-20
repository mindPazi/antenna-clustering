import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.gridspec import GridSpec

def plot_all_results(history: dict, best_solution):
    """Genera tutti i grafici dei risultati GA"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    generations = range(1, len(history['best_fitness']) + 1)
    
    # 1. Convergenza Fitness
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(generations, history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(generations, history['avg_fitness'], 'r--', linewidth=1.5, label='Avg Fitness')
    ax1.fill_between(generations, history['best_fitness'], history['avg_fitness'], alpha=0.3)
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('GA Convergence - Fitness Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. SLL Out of FoV
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(generations, history['best_sll_out'], 'g-', linewidth=2.5, marker='o', markersize=4)
    ax2.axhline(y=-20, color='r', linestyle='--', label='Target -20dB')
    ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax2.set_ylabel('SLL [dB]', fontsize=11, fontweight='bold')
    ax2.set_title('SLL Outside FoV', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SLL Inside FoV
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(generations, history['best_sll_in'], 'm-', linewidth=2.5, marker='s', markersize=4)
    ax3.axhline(y=-15, color='r', linestyle='--', label='Target -15dB')
    ax3.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax3.set_ylabel('SLL [dB]', fontsize=11, fontweight='bold')
    ax3.set_title('SLL Inside FoV', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Numero Cluster
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(generations, history['best_n_clusters'], 'c-', linewidth=2.5, marker='^', markersize=4)
    ax4.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax4.set_ylabel('N° Clusters', fontsize=11, fontweight='bold')
    ax4.set_title('Hardware Complexity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Diversità Popolazione
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(generations, history['diversity'], 'orange', linewidth=2.5)
    ax5.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Diversity', fontsize=11, fontweight='bold')
    ax5.set_title('Population Diversity', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Box
    ax6 = fig.add_subplot(gs[0, 2])
    ax6.axis('off')
    summary_text = f"""
    FINAL RESULTS
    ═══════════════════
    
    SLL out FoV: {best_solution.sll_out:.2f} dB
    SLL in FoV:  {best_solution.sll_in:.2f} dB
    
    N° Clusters: {best_solution.n_clusters}
    N° Elements: 256
    
    HW Reduction: {(1-best_solution.n_clusters/256)*100:.1f}%
    
    Convergence: Gen {len(generations)}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Trade-off Plot
    ax7 = fig.add_subplot(gs[2, 1:])
    sc = ax7.scatter(history['best_n_clusters'], history['best_sll_out'], 
                     c=generations, cmap='viridis', s=100, alpha=0.6)
    ax7.set_xlabel('N° Clusters (Hardware Cost)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('SLL out FoV [dB]', fontsize=11, fontweight='bold')
    ax7.set_title('Performance vs Complexity Trade-off', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(sc, ax=ax7)
    cbar.set_label('Generation', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Genetic Algorithm Optimization - 16x16 Antenna Array Clustering', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('GA_optimization_results.png', dpi=300, bbox_inches='tight')
    print("📊 Grafici salvati in 'GA_optimization_results.png'")
    plt.show()


def plot_convergence_simple(filename: str = 'ga_results.json'):
    """Plot semplice della convergenza"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    generations = range(1, len(history['best_fitness']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, history['best_sll_out'], 'b-', linewidth=3, marker='o')
    plt.axhline(y=-20, color='r', linestyle='--', linewidth=2, label='Target SLL = -20dB')
    plt.xlabel('Generation', fontsize=14, fontweight='bold')
    plt.ylabel('Best SLL out FoV [dB]', fontsize=14, fontweight='bold')
    plt.title('GA Convergence - Side Lobe Level Optimization', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('GA_convergence_simple.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Plot da file salvato
    plot_convergence_simple()