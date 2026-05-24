# Results Directory

Every subdirectory corresponds to one experiment.  The table below lists each
output artifact, the script that produces it, the recommended command, the
approximate wall-clock runtime on a modern laptop, and the expected output files.

## Paper 1

### `exp1/` — Exp 1: Certificate geometry across matrix families

| Artifact | Description |
|----------|-------------|
| `margins_*.png` | Determinant vs. resolvent margin profiles for 4 families |
| `summary.csv` | Per-family cert%, false-negative rates, mean margins |

**Command:**
```bash
python experiments/synthetic_certification.py --n 8 --num-matrices 200 --seed 0
```
**Smoke:** `python experiments/synthetic_certification.py --quick`  
**Runtime:** ~2–3 min  
**Expected files:** `margins_normal_stable.png`, `margins_normal_unstable.png`,
`margins_nonnormal_stable.png`, `margins_nonnormal_unstable.png`, `summary.csv`

---

### `exp2/` — Exp 2: Discretisation correctness

| Artifact | Description |
|----------|-------------|
| `disc_*.png` | Sampled vs. rigorous cert convergence per test case |
| `summary.csv` | Per-(case,N) min margin, L estimate, sampled/rigorous cert |

**Command:**
```bash
python experiments/discretisation_correctness.py --seed 42
```
**Smoke:** `python experiments/discretisation_correctness.py --quick`  
**Runtime:** ~10 sec  
**Expected files:** `disc_diag_r080_same_phase.png` … (5 figures), `summary.csv`

---

### `exp3/` — Exp 3: Trained SSMs + reference ablation (Exp 5 in paper)

| Artifact | Description |
|----------|-------------|
| `margins_*.png` | Margin profiles for 4 AR tasks |
| `summary.csv` | Per-task loss, spectral radius, margin per reference method |

**Command:**
```bash
python experiments/trained_ssm.py --n-state 8 --seed 7
```
**Smoke:** `python experiments/trained_ssm.py --quick`  
**Runtime:** ~5 min  
**Expected files:** `margins_long-memory_AR(4).png` … (4 figures), `summary.csv`

---

### `exp4/` — Exp 4: Runtime scalability

| Artifact | Description |
|----------|-------------|
| `runtime_scaling.png` | Wall-time vs. n and vs. K |
| `summary.csv` | Per-(type,n,K) timing in ms |

**Command:**
```bash
python experiments/runtime_scalability.py
```
**Smoke:** `python experiments/runtime_scalability.py --quick`  
**Runtime:** ~30 min  
**Expected files:** `runtime_scaling.png`, `summary.csv`

---

## Paper 2

### `p2exp1/` — Exp 1: Reference quality comparison

| Artifact | Description |
|----------|-------------|
| `reference_quality_margins.png` | Mean min-margin by method and family |
| `reference_opt_convergence.png` | Gradient-ascent convergence curves |
| `summary.csv` | Per-(family,method) cert%, mean margin, mean time |

**Command:**
```bash
python experiments/reference_quality.py --m 50
```
**Smoke:** `python experiments/reference_quality.py --quick`  
**Runtime:** ~5 min  
**Expected files:** `reference_quality_margins.png`, `reference_opt_convergence.png`, `summary.csv`

---

### `p2exp2/` — Exp 2: Regulariser comparison on AR tasks

| Artifact | Description |
|----------|-------------|
| `training_curves.png` | Task + reg loss curves on AR(4) hard |
| `certification_rates.png` | Post-hoc cert% by regulariser and task |
| `summary_with_std.csv` | Per-(task,reg) mean±std loss, rho, margin, cert% |

**Command:**
```bash
python experiments/regulariser_comparison.py --seeds 5 --n-epochs 500
```
**Smoke:** `python experiments/regulariser_comparison.py --quick`  
**Runtime:** ~10 min  
**Expected files:** `training_curves.png`, `certification_rates.png`, `summary_with_std.csv`

---

### `p2exp3/` — Exp 3: Adversarial unconstrained regime (Table P2-3)

| Artifact | Description |
|----------|-------------|
| `rho_distribution.pdf` | Boxplot of final spectral radius per method |
| `stability_and_cert.pdf` | Stability rate and cert% bar charts |
| `rho_trajectory.pdf` | Per-epoch rho trajectory (seed 0) for 3 methods |
| `summary.csv` | Per-method stable%, mean rho, n_diverged, cert%_stable |

**Command:**
```bash
python experiments/unstable_regime.py
```
**Smoke:** `python experiments/unstable_regime.py --quick`  
**Runtime:** ~10 min  
**Expected files:** `rho_distribution.pdf`, `stability_and_cert.pdf`, `rho_trajectory.pdf`, `summary.csv`

---

## Paper 3

### `p3exp1/` — Exp 1: Synthetic polynomial classifiers (Table P3-1)

| Artifact | Description |
|----------|-------------|
| `cert_rate_heatmap.pdf` | Cert-rate heatmap: degree × min_margin × method |
| `margin_distribution.pdf` | Margin histogram for degree=4, min_margin=0.2 |
| `summary.csv` | Per-(degree,min_margin) cert% for dense/real_poly/rouche |

**Command:**
```bash
python experiments/path_cert_synthetic.py
```
**Smoke:** `python experiments/path_cert_synthetic.py --quick`  
**Runtime:** ~5 min  
**Expected files:** `cert_rate_heatmap.pdf`, `margin_distribution.pdf`, `summary.csv`

---

### `p3exp2/` — Exp 2+3: Two-moons MLP + scaling limits (Tables P3-2, P3-3)

| Artifact | Description |
|----------|-------------|
| `two_moons_certs.pdf` | Method comparison + scaling limits figure |
| `decision_boundary.pdf` | Trained MLP decision boundary |
| `summary_phase2.csv` | Per-degree cert% for dense/real_poly/rouche |
| `summary_phase3.csv` | Per-scale cert% and approx error |

**Command:**
```bash
python experiments/path_cert_networks.py
```
**Smoke:** `python experiments/path_cert_networks.py --quick`  
**Runtime:** ~30 min  
**Expected files:** `two_moons_certs.pdf`, `decision_boundary.pdf`,
`summary_phase2.csv`, `summary_phase3.csv`

---

### `p3exp4/` — Exp 4: Verified vs. grid error bounds (Table P3-4)

| Artifact | Description |
|----------|-------------|
| `verified_vs_grid.pdf` | Scatter: eps_rig vs eps_grid, false positives highlighted |
| `summary.csv` | Per-min_margin eps_grid, eps_rig, ratio, cert%, false-pos% |

**Command:**
```bash
python experiments/path_cert_verified.py
```
**Smoke:** `python experiments/path_cert_verified.py --quick`  
**Runtime:** ~2 min  
**Expected files:** `verified_vs_grid.pdf`, `summary.csv`
