# Roche

This repository studies what Rouché-type complex-analytic perturbation tools can and cannot offer modern
machine learning. Paper 1 develops contour certificates for stability of learned state-space dynamics.
Paper 2 turns the certificate geometry into reference optimisation and training-time stability
regularisation, showing that soft penalties cannot replace hard stability enforcement. Paper 3 tests
whether the same zero-counting idea transfers to classifier-path robustness and finds a constructive
negative result: real-polynomial positivity dominates Rouché on surrogate paths. Together, the papers
trace a theory-to-practice-to-limits arc.

## Papers

| # | Title | Status |
|---|-------|--------|
| 1 | Contour Certificates for Stable Learned Dynamics | Complete |
| 2 | Stability Regularisation in Sequence Models | Complete |
| 3 | Analytic Surrogates for Pathwise Robustness | Complete |

## Repository layout

```text
src/roche/
  certificates.py      # determinant and resolvent contour certificates (Paper 1)
  matrices.py          # random matrix generators and stability utilities (Paper 1)
  reference.py         # scalar, eig-shrunk, random-search reference construction (Paper 1)
  plotting.py          # margin and eigenvalue plotting helpers (Paper 1)
  ssm.py               # diagonal SSM, L-BFGS-B trainer (Paper 1), Adam trainer (Paper 2)
  reference_opt.py     # gradient-based diagonal reference optimisation via PyTorch (Paper 2)
  regularisers.py      # spectral, Lyapunov, contour-barrier training penalties (Paper 2)
  surrogate.py         # Chebyshev surrogate fitting and approximation error (Paper 3)
  path_cert.py         # real-polynomial and Rouche path certificates (Paper 3)

experiments/
  synthetic_certification.py     # Paper 1, Exp 1: synthetic matrix families
  discretisation_correctness.py  # Paper 1, Exp 2: finite-grid theorem verification
  trained_ssm.py                 # Paper 1, Exp 3+5: trained SSMs + reference ablation
  runtime_scalability.py         # Paper 1, Exp 4: wall-time vs matrix size
  reference_quality.py           # Paper 2, Exp 1: reference method comparison
  regulariser_comparison.py      # Paper 2, Exp 2: regulariser comparison on AR tasks
  unstable_regime.py             # Paper 2, Exp 3+3b: adversarial unconstrained SSM + softplus/projection ablation
  path_cert_synthetic.py         # Paper 3, Exp 1: synthetic polynomial classifiers
  path_cert_networks.py          # Paper 3, Exp 2+3: two-moons MLP + scaling limits
  path_cert_verified.py          # Paper 3, Exp 4: verified vs grid error bounds

papers/
  paper1/main.tex                # all tables and figures filled
  paper2/main.tex                # all tables and figures filled
  paper3/main.tex                # all tables and figures filled

results/
  exp{1-4}/                      # Paper 1 figures
  p2exp{1-3}/                    # Paper 2 figures (p2exp3 = adversarial regime)
  p3exp{1-2,4}/                  # Paper 3 figures (p3exp4 = verified error bounds)

tests/
  test_certificates.py
  test_matrices.py
  test_new_code.py
  test_reference_opt.py    # Paper 2: diagonal and DLR reference optimisation
  test_regularisers.py     # Paper 2: spectral/Lyapunov/contour-barrier penalties
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```

Reproduce all experiments (run from repo root):

```bash
# Paper 1
# Exp 1: certificate geometry, n=8, 200 matrices, K=512 (~2-3 min)
python experiments/synthetic_certification.py --n 8 --num-matrices 200 --seed 0

# Exp 2: discretisation correctness, ~10 sec
python experiments/discretisation_correctness.py --seed 42

# Exp 3 + reference ablation (Exp 5 in paper): trained SSMs, ~5 min
python experiments/trained_ssm.py --n-state 8 --seed 7

# Exp 4: runtime scalability (~30 min)
python experiments/runtime_scalability.py

# Paper 2
# Exp 1: reference quality comparison (~5 min)
python experiments/reference_quality.py

# Exp 2: regulariser comparison on AR tasks (~10 min)
python experiments/regulariser_comparison.py

# Exp 3: adversarial unconstrained regime (~10 min)
python experiments/unstable_regime.py

# Paper 3
# Exp 1: synthetic polynomial classifiers (~5 min)
python experiments/path_cert_synthetic.py

# Exp 2+3: two-moons MLP + scaling limits (~30 min)
python experiments/path_cert_networks.py
```

Figures are written to `results/`. LaTeX PDFs compile with:

```bash
cd papers/paper1 && pdflatex main.tex
cd papers/paper2 && pdflatex main.tex
cd papers/paper3 && pdflatex main.tex
```

## Core idea

For a learned SSM transition matrix $A$, the resolvent certificate uses a known-stable diagonal reference $A_0$ to certify Schur stability via

$$\sup_{|z|=1} \|(zI-A_0)^{-1}(A_0-A)\| < 1.$$

Paper 1 establishes the theory and shows the bottleneck is reference quality.  
Paper 2 optimises the reference by gradient ascent and embeds it in training-time regularisers.  
Paper 3 transfers the zero-counting idea to classifier path robustness and documents why it is dominated by a simpler real-polynomial check.
