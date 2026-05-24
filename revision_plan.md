# Roche Project — Publication Revision Plan

*Prepared against the roche_claude_final_hardening_prompt.txt specification.*

---

## Executive Summary

Three papers study what Rouché-type complex-analytic perturbation tools can and cannot offer modern machine learning. The honest assessment: **the theorems are not novel**, but the translation into learned SSM post-hoc diagnostics, the reference-selection geometry, and the empirical characterisation of conservatism are genuinely useful and publishable contributions. Paper 3 is the strongest individual paper because it is a clean negative result with a sound theorem and clear geometric explanation. Papers 1 and 2 are best merged: together they tell a complete story that neither can tell alone.

**Recommended action: merge Papers 1+2 into one submission; keep Paper 3 separate.**

---

## Task 1 — Publication Strategy

### Recommended structure: Option B

**Paper A (merged 1+2):** *Interpretable Stability Margins for Learned State-Space Dynamics: A Rouché-Perturbation Framework with Reference Optimisation*

**Paper B (Paper 3 standalone):** *Real-Polynomial Positivity Dominates Rouché on Classifier Path Surrogates: A Negative Result*

### Justification

| Criterion | Three separate | Merged 1+2 + Paper 3 | One unified |
|-----------|---------------|----------------------|-------------|
| Venue fit | Weak — P1 and P2 are each too thin | Strong — each paper fills a natural slot | Too long for conference |
| Novelty per paper | P2 alone is very thin | Merged A covers theory+method+training | Would dilute Paper 3's negative-result identity |
| Reviewer risk | P2 rejected as incremental | Merged A defensible as complete system | Reviewers would demand splitting |
| Story coherence | Fragmented | Complete arc: bottleneck → fix → limits | Same |

Paper 2 alone is indefensible as an independent submission: it adds gradient reference optimisation and three regularisers, but neither contribution is large enough to carry a paper. Merged with Paper 1, the combined paper tells a complete story — theory, bottleneck, algorithmic fix, training-time embedding, limits under adversarial pressure.

**Target venues:**
- Paper A: ICLR workshop (ML for dynamical systems), L4DC, CDC/ACC with ML track, or TMLR (no page limit, suits this style of thorough empirical+theory)
- Paper B: NeurIPS workshop (verification/safety), UAI, or TMLR

---

## Task 2 — Paper Positioning

### Paper A (merged 1+2)

**Revised title:** Interpretable Post-Hoc Stability Margins for Learned SSMs via Rouché Perturbation

**Revised abstract (LaTeX-ready):**
```latex
Learned state-space models require Schur-stable transition dynamics for reliable
long-horizon prediction.  Direct spectral-radius tests verify stability but offer
limited interpretability: they reveal whether a system is stable, not how robustly,
nor which components are closest to instability.  We develop \emph{contour margin
certificates} based on Rouch\'e's theorem applied to characteristic polynomials
(determinant certificate) and matrix-valued resolvents (resolvent certificate)
around a known-stable reference $A_0$.  The resolvent certificate is equivalent to
a discrete-time small-gain condition and yields an interpretable per-mode margin for
diagonal SSMs.  A deterministic finite-grid theorem converts sampled margin checks
into continuum guarantees under a Lipschitz condition; for diagonal $A_0$ the
Lipschitz constant is analytically bounded as $\kappa \leq 1/(1-\varrho(A_0))$,
giving a fully validated certificate without interval arithmetic.

The central finding is that \emph{reference selection} is the dominant lever:
scalar or random-diagonal references certify fewer than $3\%$ of stable instances
across all matrix families.  We develop a gradient-based diagonal reference
optimisation algorithm (maximising a softmin surrogate of the resolvent margin via
Adam) that achieves $100\%$ certification on diagonal and trained-SSM families.
We embed the optimised reference into three training-time stability
regularisers---spectral penalty, Lyapunov per-mode penalty, and a differentiable
contour barrier---and show that all three achieve $100\%$ post-hoc certification in
the standard constrained regime, but none prevents unit-disk escape under
adversarial gradient pressure ($\mathrm{lr}=0.1$, near-boundary initialisation).
A post-step radius projection achieves $100\%$ stability by enforcing $\varrho
\leq 0.98$ every step; a sigmoid-constrained architecture achieves $100\%$
stability by construction.  These results establish a precise separation: soft
gradient penalties are useful diagnostics and mild regularisers, but they do not
provide hard stability guarantees.
```

**Contribution statement:**
We contribute: (1) a complete implementation and empirical characterisation of Rouché-type stability certificates for learned SSMs, including the first systematic study of reference-selection impact; (2) a gradient-based diagonal reference optimisation algorithm that resolves the main post-hoc certification bottleneck for diagonal SSMs; (3) an analytic Lipschitz bound for diagonal references that yields a fully validated finite-grid certificate without interval arithmetic; (4) a training-time contour barrier that remains active below the spectral-radius threshold; (5) an empirical separation between soft gradient penalties and hard stability enforcement.

**What this paper does not claim:**
We do not claim the resolvent certificate is a new stability condition — it is equivalent to the small-gain condition applied to the specific loop $(zI-A_0)^{-1}(A_0-A)$ on the unit circle. We do not claim the certificates compete with eigendecomposition for simple stability checking — they are post-hoc diagnostic tools. We do not claim the contour barrier improves task loss or resolvent margin — it preserves spectral radii on long-memory tasks and no more. We do not certify non-normal or Jordan matrices with diagonal references and do not claim to.

**Reviewer objections and fixes:**

| Objection | Fix |
|-----------|-----|
| "The resolvent condition is just the small-gain theorem" | Agree explicitly in §1; frame contribution as translation and characterisation |
| "Why not just check eigenvalues?" | Add Table: eigendecomp gives stability, not margin, not reference geometry, not training signal |
| "3% certification rate makes this useless" | Lead with gradient-optimised reference results; the 3% is the baseline bottleneck, not the ceiling |
| "DLR extension is underdeveloped" | Remove from main text; move to appendix; say explicitly "future work" |
| "What does the contour barrier add over spectral penalty?" | Keep Table P2-2; emphasise it fires below the spectral-radius threshold — a different mechanism |

---

### Paper B (Paper 3)

**Revised title:** Real-Polynomial Positivity Dominates Rouché Zero-Counting for Classifier Path Certification: A Negative Result

**Revised abstract:** (current abstract is already good; add one sentence)
```latex
[Current abstract unchanged through "...with $40\%$ false-positive rate at tight margins."]
We identify the \emph{complex-excursion obstruction} as the structural reason:
real-positive polynomials generically take large complex values off the real axis,
making the constant-reference Rouch\'e condition structurally impossible even when
the real certificate trivially passes.  This closes a tempting route for neural
path verification and identifies where complex-analytic methods remain useful:
settings where the contour is intrinsic to the model, not imposed through a surrogate.
```

**Contribution statement:**
We contribute: (1) a sound Rouché-type path certificate (Theorem 1) connecting complex zero-counting to pathwise margin positivity; (2) a counterexample showing naive complexification destroys analyticity (Proposition 1); (3) a quantitative characterisation of the complex-excursion obstruction explaining why Rouché must be strictly worse than the real-polynomial check; (4) interval-subdivision experiments (modulo IEEE rounding) confirming 40% false-positive rate from grid-estimated error bounds; (5) a clear delineation of where complex-analytic certificates remain useful.

**What this paper does not claim:**
We do not claim real-polynomial positivity solves neural path verification — it also becomes vacuous as path length grows. We do not compare against neural verification baselines (Reluplex, $\alpha$-$\beta$-CROWN) at an implementation level — our comparison is conceptual: surrogate-based approaches share a common vacuousness failure mode that is orthogonal to the choice of complex vs. real polynomial check.

---

## Task 3 — Strengthen Paper 1 (in merged Paper A)

### Novelty assessment — be specific

| Theorem | Mathematical content | Novelty |
|---------|---------------------|---------|
| Thm 1 (det cert) | Classical Rouché applied to char poly | **Not new.** Known in robust control (Bhattacharyya 1995). Novelty: SSM framing, numerical characterisation |
| Thm 2 (res cert) | Small-gain condition on $(zI-A_0)^{-1}(A_0-A)$ | **Not new.** IS the discrete-time small-gain theorem (Desoer & Vidyasagar). Novelty: per-mode diagonal interpretation |
| Thm 3 (grid cert) | Lipschitz-based finite-grid verification | **Not new in form.** Standard verified numerics. **Novel:** analytic $\kappa \leq 1/(1-\varrho(A_0))$ for diagonal $A_0$ gives fully validated certificate |
| Gradient ref opt | Adam on softmin surrogate of resolvent margin | **Novel empirical contribution.** No prior work on learned SSM post-hoc reference optimisation |

### Required intro revisions

**Current framing (wrong):** "when do Rouché arguments yield useful sufficient certificates?"
**Correct framing:** "Given a learned SSM transition matrix, what interpretable perturbation margin does it have relative to known-stable dynamics, and how do we find the reference that maximises this margin?"

The first paragraph of §1 must acknowledge upfront that the resolvent certificate is a small-gain condition in new notation. A skeptical reviewer will catch this on page 2; better to say it on page 1 and frame the contribution correctly.

### Specific additions needed

**Add to §Background:** A 2×5 comparison table:

| Method | Certifies stability | Margin | Per-mode interpretation | Training signal | Cost |
|--------|--------------------|---------|-----------------------|-----------------|------|
| Eigendecomp | Yes (exact) | No | No | No | O(n³) |
| Lyapunov | Yes (sufficient) | Conservative | No | Yes | O(n³) SDP |
| Stability radius | Yes (sufficient) | Tight | No | No | O(n³) SVD |
| Pseudospectrum | No | Visual | Partial | No | O(n³K) |
| **Contour cert (ours)** | Yes (sufficient) | Interpretable | Yes (diag) | Via barrier | O(Kn³) |

**Rewrite §Limitations:**
- Be explicit: Thm 2 is the small-gain condition. Citation: Desoer & Vidyasagar (1975), Trefethen & Embree (2005) §52.
- Lipschitz bound is numerical unless diagonal: say "for non-diagonal A0, a fully validated certificate requires interval arithmetic or pseudospectral bounds; for diagonal A0, κ ≤ 1/(1-ρ(A0)) gives a tight analytic bound."
- "post-hoc only" is a limitation — flag it, but explain why the contour barrier (Paper 2) addresses it.

**Add Experiment 1b:** Compare resolvent margin to stability radius $r_\mathbb{C}(A) = \min_{|z|=1} \sigma_{\min}(zI-A)$ on the diagonal SSM family. This answers the reviewer who asks "how conservative are you?" The resolvent margin with optimised reference should be within a factor of 2 of the stability radius for diagonal families — quantify this.

### Theorem upgrades

**Theorem 3 (grid cert) — stronger statement:**
For diagonal $A_0$ with $\varrho(A_0) < 1$, set $\kappa = 1/(1-\varrho(A_0))$. Then $L \leq \kappa^2 \|A_0 - A\|$ is analytically bounded without numerical estimation, and Theorem 3 with this $L$ is a **fully validated** certificate. State this as a Corollary to Theorem 3.

---

## Task 4 — Strengthen Paper 2 (in merged Paper A)

### Merge decision: yes, merge Papers 1+2

Section mapping for merged paper:

```
§1  Introduction (combined)
§2  Background: Schur stability, Rouché, small-gain, pseudospectra, SSMs
§3  Contour certificates (current Paper 1 §3-5, condensed)
§4  Finite-grid verification (current Paper 1 §6)
§5  Reference optimisation (current Paper 2 §3)
§6  Training-time stability regularisers (current Paper 2 §4-5)
§7  Adversarial regime and ablation (current Paper 2 §6-7)
§8  Experiments (combined, 5 experiments)
§9  Limitations
§10 Conclusion
Appendix A: DLR extension (future work, clearly labelled)
Appendix B: Pseudocode for reference optimisation
```

### Reference optimisation — formal statement needed

Currently the optimisation problem is stated informally. Add:

```
Problem (Diagonal Reference Optimisation)
Given A ∈ Cⁿˣⁿ, solve
    max_{d ∈ Cⁿ, |dᵢ| < 1}  min_{k=1..K} (1 - ‖diag(zₖ - d)⁻¹(A₀ - A)‖₂)
    s.t.  A₀ = diag(d)
```

Note: this is a max-min problem, non-convex in d. The softmin relaxation makes it differentiable; local optima exist.

### Pseudocode (add as Algorithm 1)

```
Algorithm 1: Gradient Diagonal Reference Optimisation
Input: A ∈ Cⁿˣⁿ, K (grid size), n_iter, lr, τ (softmin temperature)
Output: A₀ = diag(d)
1: Initialise dᵢ = σ(ℓᵢ)·eⁱᵠᵢ with ℓᵢ, φᵢ ← eigenvalues of A (shrunk)
2: For t = 1..n_iter:
3:   Compute zₖ = eⁱ²ᵖᵏ/ᴷ for k=1..K
4:   If A diagonal: margin_k = 1 - max_i |dᵢ - aᵢ|/|zₖ - dᵢ|  (O(Kn))
   Else: margin_k = 1 - σ_max(diag(zₖ-d)⁻¹(A₀-A))  (O(Kn²))
5:   loss = -softmin_τ({margin_k}ₖ)
6:   d ← Adam step on ∂loss/∂d
7:   Project: |dᵢ| ← min(|dᵢ|, 1-ε)
8: Return diag(d)
```

### Contour barrier — clarify contribution

The contour barrier's contribution is subtle and currently undersold:
- Spectral penalty fires when ρ > τ = 0.9
- Contour barrier fires when the **resolvent margin** is tight, which can happen even when ρ < 0.9
- On long-memory AR(4) tasks, ρ ≈ 0.74-0.82, so spectral/Lyapunov are inactive; contour barrier still fires
- Net effect: Δρ ≈ 0.03 reduction, modest but the only soft penalty that distinguishes itself

The current paper understates this. Add a sentence in the results: "The contour barrier is the only regulariser that remains sensitive to resolvent geometry below the spectral-radius threshold, which is why it differentiates from the baseline on long-memory tasks."

### DLR extension

**Decision: demote to Appendix, clearly future work.**

Current status: U, V initialised with Gaussian noise (scale 1e-2); stability enforced via softplus penalty; tested on n=6 rank-1 cases only. This is not enough for a main-text claim. Move to Appendix A with header: "Appendix A: Diagonal-Plus-Low-Rank Extension (Exploratory, Future Work)." Remove from abstract. Keep the implementation.

### Adversarial regime — don't make it sound straw-manned

The adversarial regime (lr=0.1, ρ_init=0.96) sounds contrived. Strengthen the motivation:

Add a paragraph in §7 intro: "High learning rates in combination with near-boundary initialisation are not hypothetical. Mamba uses unstructured A with diagonal-only stability constraints that can be violated during training, and several papers report instability under large lr in S4-style models [cite]. We deliberately construct an adversarial regime to characterise the failure mode cleanly."

---

## Task 5 — Strengthen Paper 3 (standalone)

### Formalise the complex-excursion obstruction

Currently described qualitatively in §3 and via one numerical example. It needs a Proposition.

**Proposed Proposition (Complex-Excursion Obstruction):**

> **Proposition (Complex-Excursion Lower Bound).** Let $Q:[0,1]\to\mathbb{R}$ be a degree-$d$ polynomial with $\min_{t\in[0,1]} Q(t) = \mu > 0$. Then for any constant reference $R > 0$, the Rouché condition $|Q(z) - R| < R$ on the contour $\partial D$ of a rectangular domain $D \supset [0,1]$ requires $\max_{z \in \partial D} |Q(z)| < 2R$. For a generic degree-$d$ polynomial, $\max_{z \in \partial D} |Q(z)| = \Omega(w^d \mu)$ where $w$ is the contour half-width. Thus for $d \geq 2$ and any fixed $w > 0$, the optimal reference satisfies $R = \Omega(w^d \mu)$, and the Rouché margin $\hat{m}_{\rm Rouché} = 1 - \max |Q(z)-R|/R$ degrades as degree increases even when the real margin $\mu$ is fixed.

This makes the negative result structural, not just empirical. The key insight: the real-polynomial certificate requires only $\min Q > \epsilon$; the Rouché certificate requires controlling $\max_{\partial D} Q$, which grows polynomially in degree regardless of real positivity.

**Formalise as Theorem, not Proposition, if possible.** For the specific case of Chebyshev polynomials on $[-1,1]$ extended to a Bernstein ellipse, the growth rate is known exactly — use this.

### Verified error bounds — move to main text

Currently in §6 (late) and Appendix. This is the most important finding of the paper for practical use. Restructure:

```
§4 Approximation error and the certificate gap (moved earlier)
  4.1 Grid estimate vs. interval-subdivision bound
  4.2 Exp P3-4: 40% false-positive rate at tight margins
  4.3 Implication: grid-estimated certificates are not sound
```

This makes the paper's strongest practical finding prominent.

### Comparison to neural verification baselines (conceptual)

The paper does not need to implement Reluplex or α-β-CROWN. But it must position itself:

Add a paragraph in §1: "Mainstream neural verification methods (Reluplex~\cite{katz2017reluplex}, CROWN~\cite{zhang2018efficient}, α-β-CROWN~\cite{wang2021complete}) operate on the network directly without a surrogate. Our approach is complementary: it asks whether a smooth surrogate abstraction can reduce the problem to classical complex analysis. The answer is no for general deep classifiers, but the question clarifies when surrogate-based methods can work — specifically, when the function is intrinsically analytic and the surrogate error can be verified."

### Rewrite conclusion emphasis

Current conclusion: balanced, accurate, but not punchy enough as a negative result. The key sentence should be: "Complex-analytic zero-counting is not a useful building block for classifier path certification because the complex-excursion obstruction makes the Rouché condition structurally harder to satisfy than real-polynomial positivity, at every degree tested, and the surrogate error cannot be verified cheaply enough to matter."

---

## Task 6 — Repo / Reproducibility Upgrade

### What was already implemented (done)

- `--quick` flags on all 10 experiment scripts
- CSV output on all experiments
- `results/README.md` with full experiment documentation
- `Makefile` with `smoke`, `paper1`, `paper2`, `paper3`, `pdfs` targets
- 65 unit tests passing

### Changes implemented in this session

**`.github/workflows/ci.yml` — upgraded** to include all 10 smoke tests after unit tests. Every push now runs `--quick` for each experiment, catching import errors, API breaks, and basic numerical sanity.

**`environment.yml` — created** with pinned versions (torch 2.6.0, numpy 1.26.4, scipy 1.13.1, matplotlib 3.10.0, scikit-learn 1.6.1). Install with:
```bash
conda env create -f environment.yml
conda activate roche
```

### Remaining concrete issues (GitHub issue list)

**Issue #1: Add mypy type checking to CI**
```yaml
- name: Type check
  run: pip install mypy && mypy src/roche/ --ignore-missing-imports
```
Currently no type annotations in src/roche/. Add gradually.

**Issue #2: Test the grid certificate false-positive guarantee**
Current tests check that the certificate returns the correct value; no test verifies the core theorem guarantee (Theorem 3) — that grid-verified results have no false positives. Add:
```python
def test_grid_verified_has_no_false_positives():
    # Generate 100 random diagonal stable matrices
    # For each, run grid-verified certificate
    # If certified, verify with dense (N=4096) ground truth
    # Assert: cert_grid implies cert_dense (no false positives allowed)
```

**Issue #3: Distinguish certificate levels in test assertions**
Current tests use `certified=True/False`. Tests should also check `validation_level` for all code paths — currently the `validation_level` field exists in `GridCertificateResult` but is not asserted in any test beyond `test_certificate_validation_level.py`.

**Issue #4: Deterministic seeds for all experiments**
Most experiments use `RNG_SEED = 0` or similar. Verify that all random operations (torch, numpy, sklearn) are seeded before any random call. Add `torch.use_deterministic_algorithms(True)` where feasible.

**Issue #5: results/ gitignore**
`results/` is gitignored. `results/README.md` is force-added. Raw CSV outputs should also be committed as reference baselines so reviewers can reproduce and diff. Add to `.gitignore` exceptions:
```
!results/README.md
!results/**/*.csv
```

**Issue #6: Test coverage for path_cert.py exact minimum**
`test_path_cert_exact.py` exists and passes. Add one fuzz test: for 50 random degree-8 polynomials, verify that `exact_polynomial_minimum` result ≤ grid minimum on 10000 points (the exact method should be ≤ grid).

**Issue #7: DLR Woodbury exploit**
Currently DLR resolvent uses `torch.linalg.solve` without exploiting rank-1 Woodbury. For n=256, rank=1, Woodbury reduces O(n³) → O(n²). Implement as `_woodbury_resolvent()` in `reference_opt.py`. This is a performance issue, not a correctness issue.

**Issue #8: Paper PDFs in CI**
Add optional PDF build step (requires LaTeX):
```yaml
- name: Build PDFs (optional)
  if: runner.os == 'Linux'
  run: |
    sudo apt-get install -y texlive-latex-extra texlive-science
    cd papers/paper1 && pdflatex -interaction=nonstopmode main.tex
    cd ../../papers/paper2 && pdflatex -interaction=nonstopmode main.tex
    cd ../../papers/paper3 && pdflatex -interaction=nonstopmode main.tex
```

---

## Task 7 — Revised Combined Paper A Outline (Papers 1+2)

```
Title: Interpretable Post-Hoc Stability Margins for Learned SSMs
       via Rouché Perturbation and Reference Optimisation

§1  Introduction
    - Motivation: stability checking ≠ stability understanding
    - What contour margins add over eigenvalues (Table: method comparison)
    - Paper positioning: translation + characterisation + algorithmic fix
    - Claim explicitly: resolvent cert = small-gain; novelty is reference geometry + optimisation

§2  Background
    - Schur stability, spectral radius
    - Rouché's theorem (classical)
    - Connection: stability radius, small-gain, pseudospectra
    - SSM architectures (S4, LRU, Mamba)

§3  Contour Certificates
    §3.1 Determinant certificate (Theorem 1) — note: direct Rouché application
    §3.2 Resolvent certificate (Theorem 2) — note: small-gain condition
    §3.3 Diagonal case: per-mode geometric interpretation (Corollary 1)
    §3.4 Connection to stability radius (Remark)

§4  Finite-Grid Verification (Theorem 3)
    §4.1 Lipschitz bound (numerical)
    §4.2 Analytic bound for diagonal A₀: κ ≤ 1/(1-ρ(A₀)) (Corollary 2 — FULLY VALIDATED)
    §4.3 Certificate levels: sampled / grid-verified numerical-L / fully validated

§5  Reference Optimisation
    §5.1 The reference selection problem (formal definition)
    §5.2 Gradient diagonal optimisation (Algorithm 1 — pseudocode)
    §5.3 Reference families: scalar, eig-shrunk, random, gradient
    §5.4 Complexity: O(Kn) diagonal fast path, O(Kn²) general

§6  Training-Time Stability Regularisers
    §6.1 Spectral penalty, Lyapunov penalty, contour barrier (definitions)
    §6.2 When the contour barrier is distinctive: resolvent geometry below spectral-radius threshold
    §6.3 Hard constraints: sigmoid parameterisation, radius projection

§7  Experiments
    Exp 1: Certificate geometry across matrix families (Table 1)
           — add Exp 1b: resolvent margin vs. stability radius comparison
    Exp 2: Discretisation correctness, no false positives (Table 2)
           — grid-verified with analytic L for diagonal cases
    Exp 3: Trained SSM post-hoc certification (Table 3)
    Exp 4: Runtime scaling (Table 4)
    Exp 5: Reference optimisation ablation — scalar/eig-shrunk/random/gradient (Table 5)
    Exp 6: Regulariser comparison — benign regime (Table 6)
    Exp 7: Adversarial regime + softplus/projection ablation (Table 7)

§8  Limitations
    - Sufficient not necessary; no false-positive guarantee for sampled-only
    - Diagonal references fail for non-normal/Jordan (quantified)
    - Numerical Lipschitz bound for non-diagonal A₀ (analytic bound available for diagonal)
    - O(Kn³) cost (timing table; not a bottleneck for n ≤ 64)
    - DLR extension: future work (moved to Appendix A)

§9  Conclusion
    - Separation: soft penalties useful, not sufficient for hard stability
    - Reference optimisation resolves the main post-hoc bottleneck for diagonal SSMs
    - Open: non-normal references, contour barriers at scale, formal validation

Appendix A: DLR Extension (Exploratory)
Appendix B: Pseudocode and implementation details
Appendix C: Proof details for Theorem 3
```

**Figure list:** (1) Contour margin profiles for diagonal vs. Jordan; (2) Discretisation convergence; (3) Reference comparison scatter; (4) Runtime scaling; (5) Regulariser ρ trajectories; (6) Adversarial regime stability rates bar chart

**Claims to avoid:** "certifies more instances than eigendecomposition," "novel theorem" (without qualification), "competitive with Lyapunov SDP" (different tradeoffs), "resolves instability for non-normal matrices"

---

## Task 8 — Revised Paper B Outline (Paper 3)

```
Title: Real-Polynomial Positivity Dominates Rouché Zero-Counting
       for Classifier Path Certification: A Negative Result

§1  Introduction
    - Setting: classifier path certification, why surrogates, why complex analysis tempting
    - Paper positioning: we close this route; identify geometric obstruction; delineate where Rouché remains useful
    - Conceptual comparison to neural verification baselines (Reluplex, CROWN)

§2  Background
    §2.1 Classifier path margin
    §2.2 Rouché's theorem
    §2.3 Real-polynomial positivity check (tighter, cheaper — establish upfront)

§3  Analytic Surrogate Construction
    §3.1 Chebyshev fitting
    §3.2 Approximation error — grid estimate vs. verified bound (MOVE HERE from §6)
    §3.3 Two certificate types: real-polynomial, Rouché

§4  The Complex-Excursion Obstruction (KEY SECTION)
    §4.1 Quantitative example: degree-8 polynomial, real-positive, Rouché fails
    §4.2 Proposition: Rouché condition requires controlling max|Q(z)| on ∂D
    §4.3 For Chebyshev polynomials on Bernstein ellipse: max grows as ρ_ellipse^d
    §4.4 Consequence: Rouché margin degrades as degree increases; real-polynomial does not

§5  Why Naive Complexification Fails (Proposition 1)
    §5.1 ReLU/tanh poles inside the domain
    §5.2 Counterexample: tanh network, poles at Im = π/4

§6  Experiments
    Exp 1: Synthetic polynomial classifiers — real-poly 100%, Rouché 0-46% (Table 1)
    Exp 2: Two-moons MLP — degree × method comparison (Table 2)
    Exp 3: Scaling limits — both methods collapse as path length grows (Table 3)
    Exp 4: Error bound gap — 40% false-positive rate at tight margins (Table 4, PROMINENT)

§7  Discussion and Practical Guidelines
    §7.1 When Rouché path certs are not worth attempting (checklist)
    §7.2 Where Rouché remains useful: intrinsically analytic models (unit circle, SSM)
    §7.3 What would make surrogate-based certs sound: Bernstein ellipse bound OR interval arithmetic

§8  Conclusion
    - Complex-excursion obstruction closes the Rouché route for general classifier paths
    - Real-polynomial positivity is tighter and cheaper with no geometric downside
    - Grid-estimated error bounds are unsound at tight margins; verified bounds required
    - Niche: models where the complex domain is intrinsic

Appendix A: Towards Rigorous Error Bounds (current Appendix, unchanged)
```

**Claims to avoid:** "Rouché certificates never useful," "real-polynomial check is a formal certificate" (it is not, without verified ε), "interval arithmetic is cheap"

---

## Task 9 — Publication-Standard LaTeX Text

### Combined Paper A — Abstract

```latex
\begin{abstract}
Learned state-space models require Schur-stable transition dynamics for reliable
long-horizon prediction and gradient-stable training.  Spectral-radius tests
verify stability exactly but provide no margin information: they reveal
\emph{whether} a system is stable, not \emph{how robustly}, nor which components
are closest to instability.  We develop post-hoc \emph{contour margin certificates}
based on Rouch\'e's theorem applied to characteristic polynomials (determinant
certificate) and to matrix-valued resolvents (resolvent certificate).  The resolvent
certificate is equivalent to a discrete-time small-gain condition and yields a
clean per-mode geometric interpretation for diagonal transition matrices.  A
deterministic finite-grid verification theorem converts sampled margin checks into
continuum guarantees under a Lipschitz bound; for diagonal references the bound is
analytically computable as $\kappa \leq 1/(1-\varrho(A_0))$, giving a fully
validated certificate without interval arithmetic.

We show that reference selection is the dominant practical lever: scalar or
random-diagonal references certify fewer than $3\%$ of stable instances across all
matrix families tested, while a gradient-based diagonal reference optimisation
algorithm (maximising a softmin surrogate via Adam) achieves $100\%$ certification
on diagonal and trained-SSM families.  We embed the optimised reference into three
training-time stability regularisers---spectral penalty, Lyapunov per-mode penalty,
and a differentiable contour barrier---and establish a precise separation: in the
standard constrained regime all three achieve $100\%$ post-hoc certification, but
under adversarial gradient pressure (lr~$=0.1$, near-boundary initialisation
$\varrho_\mathrm{init}=0.96$) hinge regularisers are overwhelmed by the task
gradient, while a post-step radius projection and a sigmoid-constrained architecture
achieve $100\%$ stability by enforcing the constraint every step or by construction.
These results show that soft gradient penalties are useful diagnostics and mild
regularisers, but they cannot substitute for hard stability enforcement.
\end{abstract}
```

### Combined Paper A — Introduction (key paragraphs)

```latex
\paragraph{What this paper does and does not claim.}
The resolvent certificate (Theorem~2) is not a new stability condition.  It is the
discrete-time small-gain theorem~\cite{desoer1975feedback} applied to the specific
loop $(zI-A_0)^{-1}(A_0-A)$ on the unit circle; the connection to stability radius
and pseudospectra is explicit in Remark~1.  The determinant certificate
(Theorem~1) is a direct application of classical Rouch\'e to characteristic
polynomials, known in robust control~\cite{bhattacharyya1995robust}.  Our
contribution is not these conditions themselves but: (i) their translation into an
interpretable post-hoc diagnostic for learned SSMs; (ii) the empirical
characterisation of conservatism across diagonal, non-normal, Jordan, and
trained-SSM matrix families; (iii) a gradient-based reference optimisation
algorithm that resolves the main practical bottleneck; and (iv) the training-time
contour barrier, which remains sensitive to resolvent geometry below the
spectral-radius threshold where spectral and Lyapunov penalties are inactive.

\paragraph{On non-normal matrices.}
Diagonal references cannot certify highly non-normal or Jordan matrices regardless
of optimisation quality---this is not a failure of the algorithm but a structural
limitation of the diagonal reference class.  For non-normal $A$, a positive
resolvent margin requires $\norm{(zI-A_0)^{-1}}\cdot\norm{A_0-A} < 1$; the
resolvent norm of a diagonal $A_0$ with $\varrho(A_0)=r$ is $(1-r)^{-1}$, while
the resolvent of a non-normal $A$ can be $O(n)$ times larger than its eigenvalue
gap suggests.  A diagonal reference cannot compensate for this.
```

### Combined Paper A — Limitations (replacement)

```latex
\section{Limitations}
\label{sec:limitations}

\paragraph{Sufficient conditions only.}
Both certificates are sufficient but not necessary for Schur stability.  A matrix
can be Schur stable and fail both certificates if no reference in the chosen family
yields a positive margin.  The false-negative rates reported in Experiment~1 are
the price of this sufficiency.

\paragraph{Reference class restriction.}
Diagonal references cannot certify non-normal or Jordan matrices---this is
structural.  The resolvent norm of a diagonal $A_0$ cannot compensate for the
pseudospectral sensitivity of non-normal $A$.  Extending to non-diagonal references
requires solving a harder optimisation problem; the DLR extension in
Appendix~A is exploratory.

\paragraph{Numerical vs.\ validated Lipschitz bound.}
For non-diagonal $A_0$, the Lipschitz constant $L = \kappa^2\norm{A_0-A}$ requires
numerically estimating $\kappa = \sup_{|z|=1}\norm{(zI-A_0)^{-1}}$; the
grid-verified certificate with this estimate is sound only given the numerical
estimate.  For diagonal $A_0$ with $\varrho(A_0)<1$, the Neumann series gives
$\kappa \leq 1/(1-\varrho(A_0))$ analytically, yielding a fully validated
certificate (Corollary~2).

\paragraph{Soft vs.\ hard stability.}
Gradient-based penalties cannot provide hard stability guarantees under adversarial
optimisation: any penalty whose gradient can be dominated by the task gradient at
the learning rate in use provides no stability assurance.  The experimental
separation between soft penalties (unreliable under lr~$=0.1$) and hard constraints
(100\% stable by construction) is a general finding, not specific to our
regulariser choices.

\paragraph{Cost.}
Certificate evaluation costs $O(Kn^3)$ for dense $A_0$ (or $O(Kn)$ for diagonal
$A_0$).  For $n \leq 64$ and $K \leq 512$ this is under 400\,ms per evaluation
(Experiment~4).  Reference optimisation adds roughly $10\times$ this cost over
$n_\mathrm{iter}$ iterations.  At $n=128$ or beyond, wall-clock cost becomes
significant.
```

### Paper B (Paper 3) — Abstract

```latex
\begin{abstract}
We ask whether complex-analytic zero-counting (Rouch\'e's theorem) can soundly
certify the absence of classifier boundary crossings along continuous input paths.
Given a path $\gamma:[0,1]\to\mathbb{R}^d$ between two same-class inputs, a
Chebyshev polynomial surrogate $Q_j$ is fitted to the per-class margin and the
approximation error $\epsilon_j$ is estimated on a finite grid.  We establish a
sound certificate (Theorem~1), but find that it is dominated in every tested
setting by a direct real-polynomial positivity check: the Rouch\'e condition
requires controlling $\max_{\partial D}|Q_j(z)|$ on a complex domain containing
$[0,1]$, which grows polynomially in degree for any fixed contour width, while the
real check requires only $\min_{[0,1]} Q_j > \epsilon_j$.  We call this the
\emph{complex-excursion obstruction} and characterise it for Chebyshev surrogates.

On synthetic polynomial classifiers with exact surrogates, real-polynomial
certification achieves $100\%$ across all settings while Rouch\'e achieves only
$0$--$46\%$; the failure is geometric, not error-driven.  On a two-moons MLP,
Rouch\'e lags $12$--$44$ percentage points behind real-polynomial across all
degrees and path scales.  Beyond short interpolation scales, both certificates
become vacuous as surrogate error exceeds the available margin.  Interval-subdivision
experiments (modulo IEEE floating-point rounding) reveal that grid-estimated
$\epsilon_j$ understates the true error bound by $1.44\times$ on average, with
$40\%$ false-positive rate at tight margins, making grid-based certificates unsound.
We identify where complex-analytic path certificates are and are not worth
attempting: they are structurally useful when the complex domain is intrinsic to the
model (as in SSM stability), not when imposed via a surrogate over an arbitrary deep
network path.
\end{abstract}
```

### Paper B — Introduction (closing paragraph)

```latex
\paragraph{Why this negative result is useful.}
The route we close is tempting precisely because Rouch\'e is elegant: a one-line
complex-analytic theorem that converts zero-counting into a simple domination check.
The obstruction is not theoretical but geometric: real-positive polynomials
generically have large complex values off the real axis, defeating any
constant-reference Rouch\'e condition.  Knowing this saves effort.  It also
identifies where complex-analytic certificates remain genuinely useful: settings
where the relevant contour is already part of the model (the unit circle for
Schur stability of learned dynamics), not imposed after the fact through a surrogate
fitted to an arbitrary network path.  The companion paper exploits exactly this
structure.
```

### Paper B — Conclusion (replacement)

```latex
\section{Conclusion}
\label{sec:conclusion}

We have shown that Rouch\'e-type path certificates are dominated by direct
real-polynomial positivity in every setting we tested.  The cause is structural:
the complex-excursion obstruction makes the Rouch\'e condition harder to satisfy
as polynomial degree grows, even when the real margin $\mu = \min_{[0,1]} Q_j$ is
fixed and positive.  For Chebyshev surrogates the complex values on the contour
boundary grow as the $d$-th power of the ellipse radius, while real-polynomial
certification requires only that the minimum exceeds the approximation error.
No change of reference or contour shape resolves this without defeating the
purpose of the certificate.

The sound theorem (Theorem~\ref{thm:main}) has the correct structure, but
\emph{experimental certificates are not formal}: grid-estimated $\epsilon_j$ is
not a verified upper bound, and interval-subdivision experiments confirm a
$40\%$ false-positive rate at tight margins.  Making the certificate sound in
practice requires a Bernstein ellipse bound or full interval arithmetic over
the network forward pass---both are expensive and the resulting certificate would
still be dominated by the real-polynomial check on real-valued paths.

The niche where Rouch\'e certificates remain useful is narrow but real: low-dimensional,
smooth, intrinsically analytic models where the complex domain is part of the
model specification rather than imposed post-hoc.  SSM stability on the unit
circle (the companion paper) is the canonical case.  Arbitrary deep network paths
are not.

\paragraph{What to do instead.}
For classifier path certification: use real-polynomial positivity with a verified
error bound.  For the verified bound: implement the Bernstein ellipse approach
(cheap for smooth functions, requires bounding the nearest singularity) or
interval arithmetic (network-architecture agnostic, $O(K)$ overhead).  Neither
approach requires complex analysis.
```

---

## Task 10 — Final Recommended Action Plan

### Immediate (before any submission)

1. **Fix remaining stale text in Paper 2 Key Findings (Exp 3b)** — *done this session*
2. **Rebuild PDFs** — *done this session*
3. **Commit all paper fixes and repo changes** — *pending*

### Short term (1–2 weeks)

4. **Draft merged Paper A** using the outline in Task 7. The main writing work is:
   - Unified introduction (∼1 page) that explicitly positions the resolvent cert as small-gain
   - Comparison table against eigendecomp/Lyapunov/stability-radius
   - Unified limitations section (text in Task 9 is ready)
   - Merge experiment sections (most text already written)

5. **Add Exp 1b to Paper A**: resolvent margin vs. stability radius comparison on diagonal SSMs. This directly answers "how conservative are you?" Expected result: optimised reference gives margin within 2× of stability radius for diagonal families; add a scatter plot.

6. **Add Corollary 2 (analytic Lipschitz, fully validated)** to Theorem 3 in Paper A. Text for the corollary: "For diagonal $A_0$ with $\varrho(A_0)<1$, setting $L = \kappa^2\|A_0-A\|$ with $\kappa = 1/(1-\varrho(A_0))$ yields a fully validated continuum certificate: no numerical $\kappa$ estimation is required."

7. **Formalise complex-excursion obstruction in Paper B** as a Proposition with proof. Use Chebyshev polynomial growth on Bernstein ellipse as the quantitative vehicle.

8. **Move verified error bounds to §3 of Paper B** (out of §6/Appendix).

### Medium term (2–4 weeks)

9. **Add mypy and the grid-cert false-positive test** to CI.
10. **Commit raw CSV results** to repo (add `!results/**/*.csv` to `.gitignore`).
11. **Implement Woodbury resolvent for DLR** (performance only, correctness unchanged).
12. **Write `CONTRIBUTING.md`** with seed policy, CSV result policy, and experiment naming conventions.

### Submission checklist

- [ ] No TBD markers remain in any .tex file
- [ ] All theorems have correct assumption lists
- [ ] Novelty claims qualified: det cert = classical Rouché, res cert = small-gain condition
- [ ] Limitations section is candid and publication-grade
- [ ] DLR is demoted to appendix in Paper A
- [ ] Paper B verified bounds are in main text §3-4, not appendix
- [ ] All experiments have `--quick` smoke flags and pass CI
- [ ] environment.yml committed
- [ ] PDFs build clean from `make pdfs`
