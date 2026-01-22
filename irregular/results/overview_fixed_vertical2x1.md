# Clustering Algorithms for Antenna Arrays

## Overview

This document describes the clustering optimization system for antenna arrays.

---

## INPUT

### Array Configuration (`LatticeConfig`)

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| `Nz` | Number of rows (Z axis, vertical) | 16 |
| `Ny` | Number of columns (Y axis, horizontal) | 16 |
| `dist_z` | Vertical distance between elements [in λ] | 0.7 |
| `dist_y` | Horizontal distance between elements [in λ] | 0.5 |
| `lattice_type` | Lattice type (1=Rectangular) | 1 |

### System Configuration (`SystemConfig`)

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| `freq` | Operating frequency [Hz] | 29.5e9 |
| `azi0` | Azimuth pointing direction [°] | 0 |
| `ele0` | Elevation pointing direction [°] | 0 |
| `dele` | Elevation resolution [°] | 0.5 |
| `dazi` | Azimuth resolution [°] | 0.5 |
| `lambda_` | Wavelength [m] (calculated) | 0.01017 |
| `beta` | Wave number [rad/m] (calculated) | 617.8 |

### Cluster Configuration (`ClusterConfig`)

| Parameter          | Description                                 | Example                      |
| ------------------ | ------------------------------------------- | ---------------------------- |
| `Cluster_type`     | List of subarray shapes [coordinates (n,m)] | `[np.array([[0,0], [0,1]])]` |
| `rotation_cluster` | Enable cluster rotation                     | 0                            |

**Note on Cluster_type format:**

- `[[0,0], [0,1]]` = Vertical 2×1 (same n, consecutive m)
- `[[0,0], [1,0]]` = Horizontal 1×2 (same m, consecutive n)
- Coordinates: `[n, m]` where n=column (Y), m=row (Z)

### SLL Mask Configuration (`MaskConfig`)

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| `elem` | Elevation pattern extent [±°] | 30 |
| `azim` | Azimuth pattern extent [±°] | 60 |
| `SLL_level` | SLL threshold outside FoV [dB] | 20 |
| `SLLin` | SLL threshold inside FoV [dB] | 15 |

### Element Pattern Configuration (`ElementPatternConfig`)

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| `P` | Pattern type (1=cosine, other=isotropic) | 1 |
| `Gel` | Element gain [dBi] | 5 |
| `load_file` | Load from file (0=no) | 0 |

### Simulation Parameters (`SimulationConfig`) - MC Only

| Parameter  | Description                      | Example |
| ---------- | -------------------------------- | ------- |
| `Niter`    | Number of Monte Carlo iterations | 200     |
| `Cost_thr` | Cost function threshold          | 1000    |

### Genetic Algorithm Parameters (`GAParams`) - GA Only

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| `population_size` | Population size | 15 |
| `max_generations` | Number of generations | 10 |
| `mutation_rate` | Mutation rate | 0.15 |
| `crossover_rate` | Crossover rate | 0.8 |
| `elite_size` | Preserved elite individuals | 2 |

---

## OBJECTIVE

**Minimize the Cost Function `Cm`** which penalizes:

1. **SLL violations outside the Field of View (FoV)**

   ```text
   Cm_out = Σ max(0, FF_I_dB[out] - (-SLL_level))
   ```

2. **SLL violations inside the FoV** (excluding main lobe)

   ```text
   Cm_in = Σ max(0, FF_I_dB[in] - (-SLLin))
   ```

3. **Boresight penalty** (if gain < -0.5 dB)

   ```text
   Cm_boresight = |boresight_val| × 10  (if < -0.5 dB)
   ```

**Total formula:**

```text
Cm = Cm_out + Cm_in + Cm_boresight
```

**FoV (Field of View):** region ±8° around the pointing direction (ele0, azi0).

---

## EVALUATION METHODS

### 1. Fixed Configuration (test_vertical_2x1.ipynb)

- **Method**: Direct evaluation of a predefined configuration
- **Use**: Verify theoretical results, baseline testing
- **Input**: Manually selected cluster list
- **Output**: Single configuration metrics

### 2. Original Monte Carlo

- **Method**: Random cluster selection (50% probability for each subarray)
- **Exploration**: Purely random
- **Pros**: Simple, reference baseline
- **Cons**: Inefficient, doesn't leverage good solutions

### 3. Optimized Monte Carlo

- **Method**: Adaptive selection based on cluster scores
- **Exploration**: Probability proportional to historical quality
- **Pros**: Converges faster to good solutions
- **Cons**: May get stuck in local minima

### 4. Genetic Algorithm (GA)

- **Method**: Evolution of a solution population
- **Operators**: Selection, Crossover, Mutation, Elitism
- **Pros**: Global exploration, maintains diversity
- **Cons**: More parameters to configure

---

## OUTPUT

### Output of `evaluate_clustering()`

| Metric | Dict Key | Description | Unit |
| ------- | ----------- | ----------- | ----- |
| Cost Function | `Cm` | Cost function (lower = better) | int |
| N. Clusters | `Ntrans` | Number of selected clusters | int |
| Elements/cluster | `Lsub` | Array with elements per cluster | array |
| SLL out FoV | `sll_out` | Max side lobe outside FoV | dB |
| SLL in FoV | `sll_in` | Max side lobe inside FoV (excluding main) | dB |
| Gain | `G_boresight` | Boresight gain | dBi |
| Pattern | `FF_I_dB` | Normalized 2D far-field | dB |
| Max θ | `theta_max` | Elevation of maximum | ° |
| Max φ | `phi_max` | Azimuth of maximum | ° |
| Scan Loss | `SL_maxpointing` | Loss vs maximum | dB |
| Scan Loss (0,0) | `SL_theta_phi` | Loss vs nominal boresight | dB |

**`G_boresight` calculation:**

```text
G_boresight = Gel + 10*log10(Σ Lsub)
            = Element_gain + 10*log10(N_total_elements)
```

### Lobe analysis output (`extract_lobe_metrics`)

| Metric | Key | Description | Unit |
| ------- | ------ | ----------- | ----- |
| Main Lobe | `main_lobe_gain` | Main lobe gain | dBi |
| HPBW Ele | `hpbw_ele` | Beam width at -3dB (elevation) | ° |
| HPBW Azi | `hpbw_azi` | Beam width at -3dB (azimuth) | ° |
| SLL Ele | `sll_ele_relative` | Relative SLL elevation cut | dB |
| SLL Azi | `sll_azi_relative` | Relative SLL azimuth cut | dB |
| N. Lobes Ele | `n_lobes_ele` | Number of lobes in elevation | int |
| N. Lobes Azi | `n_lobes_azi` | Number of lobes in azimuth | int |

---

## SUMMARY DIAGRAM

```text
┌─────────────────────────────────────────────────────────────────┐
│                           INPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  • LatticeConfig: Nz×Ny elements, dist_z, dist_y [λ]            │
│  • SystemConfig: freq, azi0, ele0, dele, dazi                   │
│  • MaskConfig: elem, azim, SLL_level, SLLin                     │
│  • ClusterConfig: Cluster_type (e.g. [[0,0],[0,1]] = vert 2×1)  │
│  • ElementPatternConfig: P, Gel, load_file                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SUBARRAY GENERATION                           │
├─────────────────────────────────────────────────────────────────┤
│  FullSubarraySetGeneration generates all possible               │
│  placements of the cluster_type on the array                    │
│  E.g.: 16×16 array with 2×1 cluster → 128 possible subarrays    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CLUSTER SELECTION                                │
├─────────────────────────────────────────────────────────────────┤
│  • Fixed: all clusters (full tiling)                            │
│  • MC: random selection 50%                                     │
│  • MC Opt: adaptive selection                                   │
│  • GA: population evolution                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EVALUATION (evaluate_clustering)                │
├─────────────────────────────────────────────────────────────────┤
│  1. Calculate physical positions (Yc, Zc) of clusters           │
│  2. Calculate phase coefficients (c0)                           │
│  3. Calculate far-field pattern (FF_I_dB)                       │
│  4. Calculate cost function (Cm)                                │
│  5. Calculate SLL in/out FoV                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  • Cm, Ntrans, Lsub, sll_in, sll_out, G_boresight               │
│  • FF_I_dB (2D pattern)                                         │
│  • Lobe metrics: HPBW, relative SLL, n_lobes                    │
│  • Plots: lobe analysis, 2D pattern, cluster layout             │
└─────────────────────────────────────────────────────────────────┘
```

---

## PROJECT FILES

| File | Description |
| ---- | ----------- |
| `clustering_comparison.ipynb` | Notebook with MC, MC Opt, GA |
| `test_vertical_2x1.ipynb` | Fixed vertical 2×1 configuration test |
| `antenna_physics.py` | AntennaArray class |
| `plot_results_mc.py` | Plotting functions |

---

## EXAMPLE: test_vertical_2x1.ipynb

**Configuration:**

```python
lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.7, dist_y=0.5)
cluster_type = np.array([[0, 0], [0, 1]])  # Vertical 2×1
```

**Typical output (full tiling 128 clusters):**

```text
Ntrans:      128 clusters
Nel:         256 elements (128 × 2)
Cm:          7150
G_boresight: 29.08 dBi
SLL out:     -13.32 dB
SLL in:      -3.08 dB
HPBW ele:    5.0°
HPBW azi:    7.0°
```
