# Paper 2 Outline

## Working title

**Stability Regularisation in Sequence Models: A Principled Comparison of Spectral, Lyapunov, and Rouché-Inspired Penalties**

## Target venues

Primary: TMLR, ICML, NeurIPS.  
Alternative: L4DC, ICLR workshop, NeurIPS workshop on sequence models or dynamical systems.

## One-sentence contribution

We provide a controlled empirical comparison of stability regularisers for recurrent and state-space sequence models, separating theorem-faithful Rouché contour barriers from spectral, Lyapunov, and parameterisation-based stability mechanisms.

## Corrected mathematical framing

For finite-dimensional discrete-time linear dynamics, the exact stability condition is

\[
\rho(A)<1.
\]

This condition is necessary and sufficient for asymptotic stability. Therefore, a spectral-radius penalty is not merely a proxy for Rouché's theorem. It directly targets the exact Schur condition, though practical estimates of \(\rho(A)\) may be inaccurate or unstable during training.

Rouché-faithful losses are different: they penalise violations of sufficient contour inequalities derived from Paper 1. They are likely conservative but provide a direct bridge to certified post-hoc stability.

## Taxonomy of regularisers

### Class A: Rouché-faithful contour losses

#### Determinant margin loss

\[
L_{\det}=\frac{1}{K}\sum_{k=1}^K \operatorname{softplus}\left(|p_A(z_k)-p_0(z_k)|-|p_0(z_k)|+\tau\right).
\]

Pros: theorem-faithful.  
Cons: expensive and ill-conditioned.

#### Resolvent margin loss

\[
L_{\mathrm{res}}=\frac{1}{K}\sum_{k=1}^K \operatorname{softplus}\left(\left\|(z_kI-A_0)^{-1}(A_0-A)\right\|-1+\tau\right).
\]

Pros: closer to matrix-valued certificate.  
Cons: requires solves on contour points.

### Class B: spectral penalties

#### Spectral-radius barrier

\[
L_\rho=\operatorname{softplus}(\hat\rho(A)-1+\epsilon).
\]

Here \(\hat\rho\) may be estimated using power iteration, Arnoldi iteration, or direct eigendecomposition for small matrices.

#### Spectral-norm barrier

\[
L_\sigma=\operatorname{softplus}(\|A\|_2-1+\epsilon).
\]

This guarantees stability if successful but is often too restrictive because \(\|A\|_2<1\) is sufficient, not necessary.

### Class C: Lyapunov penalties

Seek \(P\succ0\) such that

\[
A^*PA-P\prec0.
\]

Penalty:

\[
L_{\mathrm{Lyap}} = \operatorname{softplus}\left(\lambda_{\max}(A^*PA-P+\eta I)\right).
\]

For structured SSMs, use diagonal or low-rank parameterisations of \(P\).

### Class D: stable parameterisations

Parameterise dynamics to be stable by construction, for example

\[
\lambda_i=(1-\epsilon)\sigma(s_i)e^{i\phi_i}
\]

for diagonal complex SSMs.

## Hypotheses

1. **H1:** Rouché-faithful losses avoid instability but are conservative and can reduce expressivity.
2. **H2:** Spectral-radius penalties work well for normal or diagonal matrices but may underrepresent transient growth in non-normal systems.
3. **H3:** Spectral-norm barriers are stable but overly restrictive.
4. **H4:** Lyapunov penalties better capture non-normal transient amplification.
5. **H5:** No single regulariser dominates; choice depends on matrix geometry and task requirements.

## Models

Core models:

1. linear recurrent model;
2. vanilla RNN with learned transition;
3. diagonal complex SSM;
4. diagonal-plus-low-rank SSM.

Optional exploratory model:

5. small selective SSM / Mamba-mini, only if the core results are clean.

## Tasks

Core tasks:

1. synthetic impulse-response prediction;
2. delayed copy;
3. adding problem;
4. sequential MNIST.

Optional larger task:

5. one Long Range Arena task, only after the smaller tasks validate the method.

## Metrics

- task loss / accuracy;
- spectral radius of learned transition;
- spectral norm;
- pseudospectral radius or resolvent norm proxy;
- gradient norm spikes;
- hidden-state norm growth;
- Paper 1 certification rate;
- runtime and memory overhead.

## Practitioner guide

The final paper should include a decision tree such as:

- diagonal/normal transition: use spectral-radius or stable parameterisation;
- highly non-normal transition: consider Lyapunov penalty;
- post-hoc certification needed: use Paper 1 resolvent certificate;
- strict stability with cheap implementation: use stable parameterisation;
- expressive learned dynamics: avoid overly strong spectral-norm constraints.

## Expected contribution

This paper is not a state-of-the-art benchmark chase. It should be a careful map of which stability mechanisms help in which regimes, with code and diagnostics reusable by other researchers.

## Main risks

- Too many models and tasks can make the paper diffuse.
- Rouché-faithful losses may be too expensive for training.
- Lyapunov penalties may be difficult to optimise.
- Stability can improve trainability without improving final task performance.

## Success criteria

The paper succeeds if it produces a clear, empirically grounded taxonomy of stability regularisation methods, even if no method dominates.
