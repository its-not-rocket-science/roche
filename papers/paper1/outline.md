# Paper 1 Outline

## Working title

**Contour Certificates for Stable Learned Dynamics: A Rouché--Perturbation Framework for State-Space Models**

## Target venues

Primary: ICLR / NeurIPS theory track / TMLR.  
Alternative: IEEE CDC, L4DC, NeurIPS workshop on dynamical systems or ML theory.

## One-sentence contribution

We develop scalar and matrix-valued Rouché-type contour certificates for Schur stability of learned state-space transition matrices, quantify their numerical conservatism, and introduce reference-selection algorithms that make the certificates practically informative for structured learned dynamics.

## Central claim

Rouché's theorem should not be presented as a replacement for the exact spectral-radius stability condition. Instead, it gives a family of **sufficient perturbation certificates** around known-stable reference dynamics. These certificates are useful when they provide interpretable stability regions, post-hoc verification, perturbation robustness, or structure-exploiting checks that complement direct eigenvalue tests and naive norm penalties.

## Mathematical setup

Let \(A\in\mathbb C^{n\times n}\) be a learned discrete-time transition matrix. Schur stability means

\[
\rho(A)<1.
\]

For a known-stable reference \(A_0\), define

\[
p_A(z)=\det(zI-A),\qquad p_0(z)=\det(zI-A_0).
\]

The scalar Rouché margin is

\[
m_{\det}(\theta)=|p_0(e^{i\theta})|-|p_A(e^{i\theta})-p_0(e^{i\theta})|.
\]

The matrix-valued resolvent margin is

\[
m_{\mathrm{res}}(\theta)
=1-\left\|(e^{i\theta}I-A_0)^{-1}(A_0-A)\right\|.
\]

If either margin is positive for all \(\theta\in[0,2\pi]\), the learned transition is certified stable under the corresponding theorem.

## Main theorems

### Theorem 1: scalar determinant certificate

If \(A_0\) is Schur stable and

\[
|\det(zI-A_0)|>|\det(zI-A)-\det(zI-A_0)|
\]

for all \(|z|=1\), then \(A\) is Schur stable.

### Theorem 2: matrix-valued resolvent certificate

Let \(F_0(z)=zI-A_0\) and \(F(z)=zI-A\). If \(A_0\) is Schur stable and

\[
\sup_{|z|=1}\left\|F_0(z)^{-1}(F(z)-F_0(z))\right\|<1,
\]

then \(A\) is Schur stable. Since \(F(z)-F_0(z)=A_0-A\), the condition becomes

\[
\sup_{|z|=1}\left\|(zI-A_0)^{-1}(A_0-A)\right\|<1.
\]

This is the preferred numerical certificate and should be connected to small-gain reasoning, pseudospectral stability radius, and robust control.

### Theorem 3: deterministic discretisation theorem

Let \(m(\theta)\) denote either the determinant or resolvent margin. Suppose \(m\) is \(L\)-Lipschitz. For an equispaced grid \(\theta_k=2\pi k/N\), if

\[
\min_k m(\theta_k)>\frac{\pi L}{N},
\]

then \(m(\theta)>0\) for all \(\theta\), and the corresponding continuum certificate holds.

The paper should distinguish between:

- a rigorous certificate using a valid upper bound \(L\);
- a dense sampled diagnostic using an estimated \(L\).

### Theorem 4: reference optimisation

Given \(A\), choose a stable reference \(A_0\) by solving a max-min problem:

\[
\max_{A_0\in\mathcal S}\ \min_{\theta\in[0,2\pi]} m(A,A_0,\theta),
\]

where \(\mathcal S\) is a stable reference family such as scalar, diagonal, normal, block-diagonal, or structured SSM references.

Initial practical algorithms:

1. scalar search \(A_0=rI\);
2. diagonal reference \(A_0=\operatorname{diag}(r_i e^{i\phi_i})\);
3. shrunken-spectrum reference;
4. random search / coordinate search over diagonal references;
5. future gradient-based optimisation over differentiable margins.

## Experiments

### Experiment 1: certificate geometry

Generate matrices from controlled families:

- normal stable;
- normal unstable;
- non-normal stable;
- nearly defective / Jordan-like;
- block-diagonal;
- diagonal-plus-low-rank SSM-like.

Report:

- true spectral radius;
- determinant certificate success;
- resolvent certificate success;
- sampled minimum margin;
- false negatives among stable matrices;
- numerical failures;
- relationship between non-normality and certificate failure.

### Experiment 2: discretisation correctness

Use known margins and oversampled baselines to test finite-grid behaviour. The theorem predicts no false positives when the Lipschitz condition is met. Any false positive indicates numerical error or an invalid Lipschitz bound.

### Experiment 3: trained SSM case study

Train small diagonal and diagonal-plus-low-rank SSMs on synthetic long-memory tasks. Evaluate certificates post-training and compare with spectral-radius checks and norm-based sufficient conditions.

### Experiment 4: runtime and scalability

Measure wall-clock time for dense matrices up to \(n=256\). Discuss structure-exploiting opportunities for diagonal, low-rank, and block matrices.

## Expected limitations

- The certificates are sufficient, not necessary.
- Determinant margins become ill-conditioned for large \(n\).
- Resolvent margins can reject stable but highly non-normal matrices.
- Dense resolvent checks are \(O(Kn^3)\), though structure can reduce this.
- The method is initially post-hoc; training-time losses belong to Paper 2.

## Success criteria

A good Paper 1 does not need to beat spectral-radius tests. It should establish:

1. clean theorems;
2. reliable implementation;
3. meaningful numerical regimes;
4. an honest account of conservatism;
5. a reference-selection method that improves certification rates.
