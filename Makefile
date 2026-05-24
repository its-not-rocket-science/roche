.PHONY: test lint smoke paper1 paper2 paper3 pdfs all

PYTHON ?= python

# ── Quality ────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

lint:
	ruff check src/ experiments/ tests/ || true
	mypy src/roche/ --ignore-missing-imports || true

# ── Smoke (fast sanity-check, <5 min total) ────────────────────────────────

smoke:
	$(PYTHON) experiments/synthetic_certification.py --quick
	$(PYTHON) experiments/discretisation_correctness.py --quick
	$(PYTHON) experiments/trained_ssm.py --quick
	$(PYTHON) experiments/runtime_scalability.py --quick
	$(PYTHON) experiments/reference_quality.py --quick
	$(PYTHON) experiments/regulariser_comparison.py --quick
	$(PYTHON) experiments/unstable_regime.py --quick
	$(PYTHON) experiments/path_cert_synthetic.py --quick
	$(PYTHON) experiments/path_cert_networks.py --quick
	$(PYTHON) experiments/path_cert_verified.py --quick

# ── Paper 1 experiments (~45 min) ─────────────────────────────────────────

paper1:
	$(PYTHON) experiments/synthetic_certification.py --n 8 --num-matrices 200 --seed 0
	$(PYTHON) experiments/discretisation_correctness.py --seed 42
	$(PYTHON) experiments/trained_ssm.py --n-state 8 --seed 7
	$(PYTHON) experiments/runtime_scalability.py

# ── Paper 2 experiments (~25 min) ─────────────────────────────────────────

paper2:
	$(PYTHON) experiments/reference_quality.py --m 50
	$(PYTHON) experiments/regulariser_comparison.py --seeds 5 --n-epochs 500
	$(PYTHON) experiments/unstable_regime.py

# ── Paper 3 experiments (~40 min) ─────────────────────────────────────────

paper3:
	$(PYTHON) experiments/path_cert_synthetic.py
	$(PYTHON) experiments/path_cert_networks.py
	$(PYTHON) experiments/path_cert_verified.py

# ── Compile PDFs ──────────────────────────────────────────────────────────

pdfs:
	cd papers/paper1 && pdflatex main.tex && pdflatex main.tex
	cd papers/paper2 && pdflatex main.tex && pdflatex main.tex
	cd papers/paper3 && pdflatex main.tex && pdflatex main.tex

# ── Full reproduction ──────────────────────────────────────────────────────

all: test paper1 paper2 paper3 pdfs
