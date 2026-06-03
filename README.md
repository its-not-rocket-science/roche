# Roche

**Reference-optimised contour certificates for stable learned state-space models.**

Roche studies what Rouché-type complex-analytic perturbation tools can and cannot offer modern machine learning. The project starts with Schur-stability certificates for learned state-space models (SSMs), turns the certificate geometry into reference optimisation and training-time regularisation experiments, and then tests the same zero-counting idea on classifier-path robustness where it largely fails.

The central message is deliberately modest:

- contour certificates are **sufficient**, not necessary;
- they are not replacements for eigenvalue, Lyapunov, Schur-Cohn, stability-radius, or pseudospectral methods;
- their value is in **post-hoc perturbation margins**, **reference-dependent diagnostics**, and **failure-mode analysis**;
- reference quality is the dominant practical bottleneck;
- for classifier path robustness, real-polynomial positivity dominates the Rouché-style complex certificate in the tested settings.

## Project arc

| Paper | Focus | Main result | Current status |
|---|---|---|---|
| Paper 1 | Contour certificates for Schur-stable learned dynamics | Determinant and resolvent certificates are mathematically sound but conservative; reference selection determines practical usefulness. | Complete draft |
| Paper 2 | Reference optimisation and stability regularisation for diagonal SSMs | Gradient-optimised diagonal references close the post-hoc certification gap on diagonal/trained SSMs; soft penalties do not replace hard stability constraints. | Complete draft |
| Paper 3 | Rouché-type path certificates for neural robustness | A sound complex-analytic path certificate exists under verified surrogate-error bounds, but it is dominated by real-polynomial positivity and is fragile under grid-estimated error. | Complete draft |

The intended publication strategy is likely **one combined SSM paper** from the strongest parts of Papers 1 and 2, plus **Paper 3 as a separate negative-result paper**.

## Core certificate

For a learned transition matrix `A` and a known-stable reference `A0`, the resolvent certificate checks

```math
\sup_{|z|=1} \left\|(zI-A_0)^{-1}(A_0-A)\right\|_2 < 1.
```

If this holds and `A0` is Schur stable, then `A` is Schur stable. This is a Rouché/small-gain-style sufficient condition. A positive margin provides a perturbation buffer around the reference dynamics; a negative margin on a truly stable matrix is a false negative and usually indicates poor reference geometry or strong non-normality.

For diagonal `A` and diagonal `A0`, the certificate reduces to a per-mode geometric condition and can be evaluated cheaply. For dense non-normal matrices, scalar or diagonal references are often too weak.

## What this repository is not claiming

This repository does **not** claim that contour certificates are a better way to test stability than computing eigenvalues for a fixed dense matrix. It also does **not** claim that soft regularisation provides hard stability guarantees, or that Rouché zero-counting is a useful general-purpose replacement for neural-network verification methods.

The project is instead about understanding when these analytic certificates are informative, when they are vacuous, and how reference selection changes the answer.

## Repository layout

```text
src/roche/
  certificates.py      # determinant and resolvent contour certificates
  matrices.py          # random matrix generators and stability utilities
  reference.py         # scalar, eig-shrunk, random-search reference construction
  plotting.py          # margin and eigenvalue plotting helpers
  ssm.py               # diagonal SSMs and training utilities
  reference_opt.py     # gradient-based diagonal and exploratory DLR reference optimisation
  regularisers.py      # spectral, Lyapunov, and contour-barrier penalties
  surrogate.py         # Chebyshev surrogate fitting and approximation-error utilities
  path_cert.py         # real-polynomial and Rouché path certificates

experiments/
  synthetic_certification.py     # Paper 1 Exp 1: synthetic matrix families
  discretisation_correctness.py  # Paper 1 Exp 2: finite-grid theorem checks
  trained_ssm.py                 # Paper 1 Exp 3+5: trained SSMs + reference ablation
  runtime_scalability.py         # Paper 1 Exp 4: runtime scaling
  reference_quality.py           # Paper 2 Exp 1: reference quality comparison
  regulariser_comparison.py      # Paper 2 Exp 2: regulariser comparison on AR tasks
  unstable_regime.py             # Paper 2 Exp 3+3b: adversarial unconstrained training
  path_cert_synthetic.py         # Paper 3 Exp 1: synthetic polynomial classifiers
  path_cert_networks.py          # Paper 3 Exp 2+3: two-moons MLP + scaling limits
  path_cert_verified.py          # Paper 3 Exp 4: verified-vs-grid error bounds

papers/
  paper1/main.tex
  paper2/main.tex
  paper3/main.tex

results/
  exp{1-4}/                      # Paper 1 outputs
  p2exp{1-3}/                    # Paper 2 outputs
  p3exp{1-2,4}/                  # Paper 3 outputs

tests/
  test_certificates.py
  test_matrices.py
  test_new_code.py
  test_reference_opt.py
  test_regularisers.py
```

## Installation

Roche requires Python 3.10 or later.

```bash
git clone https://github.com/its-not-rocket-science/roche.git
cd roche
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

The package dependencies are NumPy, SciPy, Matplotlib, PyTorch, and scikit-learn. Development dependencies currently include pytest and ruff.

## Run tests

```bash
pytest
```

## Reproduce experiments

Run all commands from the repository root.

### Paper 1: contour certificates for learned dynamics

```bash
# Exp 1: certificate geometry across matrix families
python experiments/synthetic_certification.py --n 8 --num-matrices 200 --seed 0

# Exp 2: finite-grid/discretisation correctness checks
python experiments/discretisation_correctness.py --seed 42

# Exp 3 + Exp 5: trained diagonal SSMs and reference ablation
python experiments/trained_ssm.py --n-state 8 --seed 7

# Exp 4: runtime scalability
python experiments/runtime_scalability.py
```

### Paper 2: reference optimisation and stability regularisation

```bash
# Exp 1: scalar, eig-shrunk, random-search, and gradient-optimised references
python experiments/reference_quality.py

# Exp 2: regulariser comparison on controlled AR tasks
python experiments/regulariser_comparison.py

# Exp 3 + 3b: adversarial unconstrained regime, softplus, projection, and constrained baseline
python experiments/unstable_regime.py
```

### Paper 3: analytic surrogates for path robustness

```bash
# Exp 1: synthetic polynomial classifiers
python experiments/path_cert_synthetic.py

# Exp 2 + 3: two-moons MLP and path-scaling limits
python experiments/path_cert_networks.py

# Exp 4: verified-vs-grid approximation-error bounds
python experiments/path_cert_verified.py
```

Figures and result artefacts are written under `results/`.

## Build the papers

Each paper is a standalone LaTeX document.

```bash
cd papers/paper1 && pdflatex main.tex
cd ../paper2 && pdflatex main.tex
cd ../paper3 && pdflatex main.tex
```

Depending on your LaTeX installation, you may need to run `pdflatex` more than once to resolve references.

## Certificate terminology

The papers distinguish three levels of certificate strength.

| Term | Meaning | Interpretation |
|---|---|---|
| `sampled_only` | The minimum margin is positive on a finite evaluation grid. | Useful diagnostic, but not a continuum guarantee. |
| `grid_verified_numeric_L` | The finite-grid theorem is applied with a numerically estimated Lipschitz constant. | Sound conditional on the numerical estimate, but not a fully formal certificate. |
| `fully_validated` | The Lipschitz/error bound is analytically or interval-validated. | Formal certificate under the stated assumptions. |

In Paper 1, the deterministic finite-grid theorem requires a validated Lipschitz bound. When the Lipschitz constant is estimated from the same grid, the result is reported as numerical grid verification rather than a fully rigorous proof. One tractable fully validated case is diagonal `A0`, where the Neumann-series bound gives

```math
\kappa = \sup_{|z|=1}\|(zI-A_0)^{-1}\| \leq \frac{1}{1-\rho(A_0)}.
```

In Paper 3, surrogate-error estimates based on finite grids are empirical. The `verified_error_bound` function in `src/roche/surrogate.py` uses interval subdivision with standard IEEE 754 floating-point arithmetic and no directed rounding. It is more conservative than a 500-point grid estimate, but it should not be described as a formal interval-arithmetic proof.

## Main empirical findings

### Paper 1

- Scalar and random-diagonal references certify very few stable instances across the tested matrix families.
- The bottleneck is reference quality rather than the determinant/resolvent distinction.
- Eig-shrunk diagonal references certify diagonal and trained diagonal SSMs much more reliably.
- The resolvent certificate is numerically better conditioned than the determinant at larger dimensions, but it is not uniformly less conservative.
- Non-normal and Jordan-like matrices expose the limitations of scalar/diagonal references.

### Paper 2

- Gradient-optimised diagonal references achieve strong margins on diagonal and trained-SSM families.
- Diagonal references remain insufficient for dense non-normal and Jordan-near-boundary examples.
- In benign diagonal AR training, spectral/Lyapunov penalties often do little because the model is already comfortably stable.
- The contour barrier gives a modest geometry-aware spectral-radius reduction, but not a hard guarantee.
- Under adversarial high-learning-rate pressure, soft penalties can be overwhelmed; projection or constrained parameterisation is the reliable solution.

### Paper 3

- Real-polynomial positivity matches or exceeds the Rouché path certificate in every tested setting.
- Rouché path certificates fail through complex-domain excursion even when the real path is safely positive.
- Higher-degree polynomial surrogates amplify this failure mode.
- Grid-estimated surrogate errors can create false certificates at tight margins.
- The promising setting for Rouché is not arbitrary neural path verification, but intrinsically analytic problems where the complex contour is natural.

## Known limitations

- The certificates are sufficient only and can have high false-negative rates.
- Dense resolvent checks scale as `O(K n^3)` for `K` contour points and matrix dimension `n`.
- The current DLR reference optimiser is exploratory and not yet systematically evaluated on hard non-normal/Jordan families.
- The contour barrier is a soft training penalty and should not be marketed as a hard stability mechanism.
- The Paper 3 surrogate-error pipeline is not fully formal without directed-rounding interval arithmetic or another validated error-bound method.
- The experiments are controlled and diagnostic; they are not broad benchmark claims.

## Development priorities

The most important next steps are:

1. merge the strongest Paper 1 and Paper 2 material into one clearer SSM paper;
2. add one-command scripts that regenerate every table and figure from raw CSV/JSON outputs;
3. add a CI workflow for tests and linting;
4. add a lockfile or pinned environment file for exact reproducibility;
5. expand tests that distinguish sampled, numerical-grid-verified, and formally validated certificates;
6. either fully evaluate DLR/non-diagonal references or keep them clearly marked as future work;
7. strengthen Paper 3 with a sharper formal statement of the complex-excursion obstruction.

## Citation

No formal citation entry is available yet. If you use this repository, cite the repository URL and the relevant draft paper.

```bibtex
@misc{roche-project,
  title  = {Roche: Reference-Optimised Contour Certificates for Stable Learned State-Space Models},
  author = {Roche Project},
  year   = {2026},
  url    = {https://github.com/its-not-rocket-science/roche}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
