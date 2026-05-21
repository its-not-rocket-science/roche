# Roche

Research code and paper scaffolding for exploring Rouché-type perturbation certificates in machine learning, especially stability certification for learned state-space models.

## Contents

```text
roche/
  PROJECT_DESCRIPTION.md
  README.md
  pyproject.toml
  src/roche/
    certificates.py        # determinant and resolvent contour certificates
    matrices.py            # random matrix generators and stability utilities
    reference.py           # stable reference construction/search
    plotting.py            # simple plotting helpers
  experiments/
    synthetic_certification.py
  papers/
    paper1/
      outline.md
      main.tex
    paper2/
      outline.md
    paper3/
      outline.md
  tests/
    test_certificates.py
    test_matrices.py
```

## Quick start

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```

Run the starter synthetic experiment:

```bash
python experiments/synthetic_certification.py --n 8 --num-matrices 100 --num-contour 256
```

## Core idea for Paper 1

For a learned transition matrix \(A\), define

\[
p_A(z)=\det(zI-A).
\]

Given a known-stable reference matrix \(A_0\), Rouché's theorem gives the sufficient stability certificate

\[
|p_{A_0}(z)| > |p_A(z)-p_{A_0}(z)|,\qquad |z|=1.
\]

The repository also implements a matrix-valued resolvent variant based on

\[
\sup_{|z|=1}\|(zI-A_0)^{-1}(A_0-A)\| < 1.
\]

The determinant certificate is theorem-faithful but can be ill-conditioned. The resolvent certificate is often numerically preferable and connects the project to robust stability, pseudospectra, and small-gain arguments.

## Development priorities

1. Strengthen the numerical implementation of contour margins.
2. Add deterministic Lipschitz bounds rather than relying only on dense sampling.
3. Expand reference-selection methods.
4. Add structured SSM matrices and trained transition matrices.
5. Build Paper 2 regularisers on top of the Paper 1 certificate machinery.
