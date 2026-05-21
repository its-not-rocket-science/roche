# Paper 3 Outline

## Working title

**Analytic Surrogates for Pathwise Robustness: On the Promise and Limits of Rouché-Type Zero-Counting in Neural Network Verification**

## Target venues

Primary: verification / robustness workshops, TMLR, JAIR special issue, JMLR if the theory and negative results are strong.  
Alternative: NeurIPS/ICML workshop on robustness, reliable ML, or negative results.

## One-sentence contribution

We investigate whether complex-analytic zero-counting can soundly certify the absence of classifier boundary crossings along input paths, and show under explicit analytic-surrogate assumptions where the method works, fails, and becomes impractically conservative.

## Key correction from the original idea

A naive Cauchy or complex extension of a real path-margin function does not automatically preserve the sign-change or adversarial-crossing structure of the original real path. The paper must therefore build an analytic surrogate with an explicit approximation-error bound.

## Setup

Let

\[
f:\mathbb R^d\to\mathbb R^K
\]

be a classifier. Let \(x_a,x_b\) be two inputs with the same predicted class \(c\), and let

\[
\gamma:[0,1]\to\mathbb R^d
\]

be a continuous path between them.

For each rival class \(j\ne c\), define the margin

\[
g_j(t)=f_c(\gamma(t))-f_j(\gamma(t)).
\]

The path is certified class-preserving if

\[
g_j(t)>0
\]

for all \(t\in[0,1]\) and all \(j\ne c\).

## Analytic surrogate construction

Fit a polynomial or Chebyshev surrogate \(q_j(t)\) to \(g_j(t)\) on \([0,1]\). Extend it analytically to a complex polynomial \(Q_j(z)\) using the same coefficients.

Compute or bound the approximation error

\[
\epsilon_j\ge \max_{t\in[0,1]} |g_j(t)-Q_j(t)|.
\]

A direct real-path certificate is

\[
\min_{t\in[0,1]}Q_j(t)>\epsilon_j.
\]

## Rouché zero-free certificate

Let \(D\subset\mathbb C\) be a domain containing \([0,1]\). Let \(R_j(z)\) be a zero-free analytic reference on \(D\). If

\[
|R_j(z)|>|Q_j(z)-R_j(z)|
\]

for every \(z\in\partial D\), then \(Q_j\) has no zeros in \(D\). If additionally \(Q_j(0)>\epsilon_j\) and the approximation bound holds, then the original margin \(g_j\) remains positive on the path.

## Main theorem

**Theorem.** Suppose \(Q_j\) is analytic on a domain \(D\supset[0,1]\), \(R_j\) is analytic and zero-free on \(D\), and

\[
|R_j(z)|>|Q_j(z)-R_j(z)|
\]

on \(\partial D\). Then \(Q_j\) is zero-free on \(D\). If furthermore

\[
\max_{t\in[0,1]}|g_j(t)-Q_j(t)|\le\epsilon_j
\]

and \(Q_j(t)>\epsilon_j\) for at least one connected sign-fixed verification condition on \([0,1]\), then \(g_j(t)>0\) on the path.

The final paper should state the last positivity step carefully. Zero-freeness alone proves that \(Q_j\) does not cross zero; the sign is fixed by evaluating \(Q_j\) at one real point.

## Didactic counterexample

Include a counterexample showing that a naive analytic extension can introduce zeros that do not correspond to real path crossings, or hide real-path behaviour. This motivates the surrogate-plus-error framework.

## Empirical phases

### Phase 1: synthetic polynomial classifiers

Use 2D polynomial decision functions with known boundaries. Verify that Rouché certificates work for low-degree, high-margin paths. Increase degree and reduce margins to observe breakdown.

### Phase 2: small polynomial networks

Train quadratic/cubic activation networks on two-moons and concentric-circles datasets. Fit Chebyshev surrogates to path margins and evaluate:

- dense-sampling ground truth;
- real polynomial positivity certificate;
- Rouché zero-free certificate;
- approximation error;
- certification rate;
- false certification rate.

### Phase 3: small MNIST MLP

Use a one-hidden-layer MLP. Choose short same-class paths, such as pixel interpolation or latent-space interpolation. Attempt certification and document why it succeeds or fails.

### Phase 4: negative-result analysis

Quantify:

- polynomial degree required for accurate surrogates;
- how approximation error scales with path length;
- how margins shrink with input dimension;
- how contour size affects Rouché margins;
- whether interval/Bernstein methods outperform Rouché certificates.

## Expected outcome

The likely outcome is that Rouché certificates work on controlled analytic toy examples but become extremely conservative for realistic neural paths because approximation error consumes the available margin.

This is still a valuable negative result because it clarifies the boundary between elegant complex analysis and practical neural network verification.

## Success criteria

The paper succeeds if it provides:

1. a sound theorem;
2. a clear counterexample to naive complexification;
3. controlled positive examples;
4. convincing negative evidence on neural paths;
5. practical guidelines for when complex-analytic certification is not worth attempting.
