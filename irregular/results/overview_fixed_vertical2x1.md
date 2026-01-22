# Algoritmi di Clustering per Array di Antenne

## Panoramica

Questo documento descrive il sistema di ottimizzazione del clustering per array di antenne.

---

## INPUT

### Configurazione Array (`LatticeConfig`)

| Parametro | Descrizione | Esempio |
| --------- | ----------- | ------- |
| `Nz` | Numero di righe (asse Z, verticale) | 16 |
| `Ny` | Numero di colonne (asse Y, orizzontale) | 16 |
| `dist_z` | Distanza verticale tra elementi [in λ] | 0.7 |
| `dist_y` | Distanza orizzontale tra elementi [in λ] | 0.5 |
| `lattice_type` | Tipo di reticolo (1=Rettangolare) | 1 |

### Configurazione Sistema (`SystemConfig`)

| Parametro | Descrizione | Esempio |
| --------- | ----------- | ------- |
| `freq` | Frequenza operativa [Hz] | 29.5e9 |
| `azi0` | Azimuth direzione puntamento [°] | 0 |
| `ele0` | Elevation direzione puntamento [°] | 0 |
| `dele` | Risoluzione elevation [°] | 0.5 |
| `dazi` | Risoluzione azimuth [°] | 0.5 |
| `lambda_` | Lunghezza d'onda [m] (calcolato) | 0.01017 |
| `beta` | Numero d'onda [rad/m] (calcolato) | 617.8 |

### Configurazione Cluster (`ClusterConfig`)

| Parametro            | Descrizione                                | Esempio                      |
| -------------------- | ------------------------------------------ | ---------------------------- |
| `Cluster_type`       | Lista di forme subarray [coordinate (n,m)] | `[np.array([[0,0], [0,1]])]` |
| `rotation_cluster`   | Abilitazione rotazione cluster             | 0                            |

**Nota sul formato Cluster_type:**

- `[[0,0], [0,1]]` = Vertical 2×1 (stesso n, m consecutivi)
- `[[0,0], [1,0]]` = Horizontal 1×2 (stesso m, n consecutivi)
- Coordinate: `[n, m]` dove n=colonna (Y), m=riga (Z)

### Configurazione Maschera SLL (`MaskConfig`)

| Parametro | Descrizione | Esempio |
| --------- | ----------- | ------- |
| `elem` | Estensione elevation pattern [±°] | 30 |
| `azim` | Estensione azimuth pattern [±°] | 60 |
| `SLL_level` | Soglia SLL fuori dal FoV [dB] | 20 |
| `SLLin` | Soglia SLL dentro il FoV [dB] | 15 |

### Configurazione Pattern Elemento (`ElementPatternConfig`)

| Parametro | Descrizione | Esempio |
| --------- | ----------- | ------- |
| `P` | Tipo pattern (1=coseno, altro=isotropico) | 1 |
| `Gel` | Guadagno elemento [dBi] | 5 |
| `load_file` | Carica da file (0=no) | 0 |

### Parametri Simulazione (`SimulationConfig`) - Solo per MC

| Parametro  | Descrizione                    | Esempio |
|------------|--------------------------------|---------|
| `Niter`    | Numero iterazioni Monte Carlo  | 200     |
| `Cost_thr` | Soglia cost function           | 1000    |

### Parametri Genetic Algorithm (`GAParams`) - Solo per GA

| Parametro | Descrizione | Esempio |
| --------- | ----------- | ------- |
| `population_size` | Dimensione popolazione | 15 |
| `max_generations` | Numero generazioni | 10 |
| `mutation_rate` | Tasso mutazione | 0.15 |
| `crossover_rate` | Tasso crossover | 0.8 |
| `elite_size` | Individui elite preservati | 2 |

---

## OBIETTIVO

**Minimizzare la Cost Function `Cm`** che penalizza:

1. **Violazioni SLL fuori dal Field of View (FoV)**

   ```text
   Cm_out = Σ max(0, FF_I_dB[out] - (-SLL_level))
   ```

2. **Violazioni SLL dentro il FoV** (escluso lobo principale)

   ```text
   Cm_in = Σ max(0, FF_I_dB[in] - (-SLLin))
   ```

3. **Penalità boresight** (se guadagno < -0.5 dB)

   ```text
   Cm_boresight = |boresight_val| × 10  (se < -0.5 dB)
   ```

**Formula totale:**

```text
Cm = Cm_out + Cm_in + Cm_boresight
```

**FoV (Field of View):** regione ±8° attorno alla direzione di puntamento (ele0, azi0).

---

## METODI DI VALUTAZIONE

### 1. Configurazione Fissa (test_vertical_2x1.ipynb)

- **Metodo**: Valutazione diretta di una configurazione predefinita
- **Uso**: Verifica risultati teorici, test baseline
- **Input**: Lista di cluster selezionati manualmente
- **Output**: Metriche della singola configurazione

### 2. Monte Carlo Originale

- **Metodo**: Selezione casuale dei cluster (probabilità 50% per ogni subarray)
- **Esplorazione**: Puramente random
- **Pro**: Semplice, baseline di riferimento
- **Contro**: Inefficiente, non sfrutta soluzioni buone

### 3. Monte Carlo Ottimizzato

- **Metodo**: Selezione adattiva basata su score dei cluster
- **Esplorazione**: Probabilità proporzionale alla qualità storica
- **Pro**: Converge più velocemente a soluzioni buone
- **Contro**: Può rimanere in minimi locali

### 4. Genetic Algorithm (GA)

- **Metodo**: Evoluzione di una popolazione di soluzioni
- **Operatori**: Selezione, Crossover, Mutazione, Elitismo
- **Pro**: Esplorazione globale, mantiene diversità
- **Contro**: Più parametri da configurare

---

## OUTPUT

### Output di `evaluate_clustering()`

| Metrica | Chiave Dict | Descrizione | Unità |
| ------- | ----------- | ----------- | ----- |
| Cost Function | `Cm` | Funzione costo (più basso = migliore) | int |
| N. Cluster | `Ntrans` | Numero di cluster selezionati | int |
| Elementi/cluster | `Lsub` | Array con elementi per ogni cluster | array |
| SLL out FoV | `sll_out` | Max side lobe fuori dal FoV | dB |
| SLL in FoV | `sll_in` | Max side lobe dentro il FoV (escluso main) | dB |
| Guadagno | `G_boresight` | Guadagno al boresight | dBi |
| Pattern | `FF_I_dB` | Far-field normalizzato 2D | dB |
| Max θ | `theta_max` | Elevation del massimo | ° |
| Max φ | `phi_max` | Azimuth del massimo | ° |
| Scan Loss | `SL_maxpointing` | Perdita vs massimo | dB |
| Scan Loss (0,0) | `SL_theta_phi` | Perdita vs boresight nominale | dB |

**Calcolo `G_boresight`:**

```text
G_boresight = Gel + 10*log10(Σ Lsub)
            = Guadagno_elemento + 10*log10(N_elementi_totali)
```

### Output analisi lobi (`extract_lobe_metrics`)

| Metrica | Chiave | Descrizione | Unità |
| ------- | ------ | ----------- | ----- |
| Main Lobe | `main_lobe_gain` | Guadagno lobo principale | dBi |
| HPBW Ele | `hpbw_ele` | Larghezza fascio a -3dB (elevation) | ° |
| HPBW Azi | `hpbw_azi` | Larghezza fascio a -3dB (azimuth) | ° |
| SLL Ele | `sll_ele_relative` | SLL relativo taglio elevation | dB |
| SLL Azi | `sll_azi_relative` | SLL relativo taglio azimuth | dB |
| N. Lobi Ele | `n_lobes_ele` | Numero lobi in elevation | int |
| N. Lobi Azi | `n_lobes_azi` | Numero lobi in azimuth | int |

---

## SCHEMA RIASSUNTIVO

```text
┌─────────────────────────────────────────────────────────────────┐
│                           INPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  • LatticeConfig: Nz×Ny elementi, dist_z, dist_y [λ]            │
│  • SystemConfig: freq, azi0, ele0, dele, dazi                   │
│  • MaskConfig: elem, azim, SLL_level, SLLin                     │
│  • ClusterConfig: Cluster_type (es. [[0,0],[0,1]] = vert 2×1)   │
│  • ElementPatternConfig: P, Gel, load_file                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERAZIONE SUBARRAYS                         │
├─────────────────────────────────────────────────────────────────┤
│  FullSubarraySetGeneration genera tutti i possibili             │
│  posizionamenti del cluster_type sull'array                     │
│  Es: array 16×16 con cluster 2×1 → 128 subarrays possibili      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SELEZIONE CLUSTER                                │
├─────────────────────────────────────────────────────────────────┤
│  • Fissa: tutti i cluster (full tiling)                         │
│  • MC: selezione random 50%                                     │
│  • MC Opt: selezione adattiva                                   │
│  • GA: evoluzione popolazione                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VALUTAZIONE (evaluate_clustering)               │
├─────────────────────────────────────────────────────────────────┤
│  1. Calcola posizioni fisiche (Yc, Zc) dei cluster              │
│  2. Calcola coefficienti di fase (c0)                           │
│  3. Calcola far-field pattern (FF_I_dB)                         │
│  4. Calcola cost function (Cm)                                  │
│  5. Calcola SLL in/out FoV                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  • Cm, Ntrans, Lsub, sll_in, sll_out, G_boresight               │
│  • FF_I_dB (pattern 2D)                                         │
│  • Metriche lobi: HPBW, SLL relativi, n_lobes                   │
│  • Plot: lobe analysis, pattern 2D, cluster layout              │
└─────────────────────────────────────────────────────────────────┘
```

---

## FILE DEL PROGETTO

| File | Descrizione |
| ---- | ----------- |
| `clustering_comparison.ipynb` | Notebook con MC, MC Opt, GA |
| `test_vertical_2x1.ipynb` | Test configurazione fissa vertical 2×1 |
| `antenna_physics.py` | Classe AntennaArray |
| `plot_results_mc.py` | Funzioni di plotting |

---

## ESEMPIO: test_vertical_2x1.ipynb

**Configurazione:**

```python
lattice = LatticeConfig(Nz=16, Ny=16, dist_z=0.7, dist_y=0.5)
cluster_type = np.array([[0, 0], [0, 1]])  # Vertical 2×1
```

**Output tipico (full tiling 128 cluster):**

```text
Ntrans:      128 cluster
Nel:         256 elementi (128 × 2)
Cm:          7150
G_boresight: 29.08 dBi
SLL out:     -13.32 dB
SLL in:      -3.08 dB
HPBW ele:    5.0°
HPBW azi:    7.0°
```
