"""
Plotting functions for Genetic Algorithm results
Aligned with clustering_comparison.ipynb notebook

Note: The notebook does not define a separate plot_ga_results function.
GA plots are done inline in the notebook cells.
This file re-exports functions from plot_results_mc for compatibility.
"""

# Re-export all plotting functions from plot_results_mc
from plot_results_mc import (
    extract_lobe_metrics,
    plot_lobe_analysis,
    rows_to_clusters,
    genes_to_clusters,
    get_ff_from_mc,
    get_ff_from_ga,
    plot_cluster_layout,
)
