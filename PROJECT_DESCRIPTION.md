# Roche Project Description

This repository supports a research programme on using Rouché-type complex-analytic perturbation tools in machine learning, with a primary focus on stability certification for learned state-space models (SSMs).

The project is organised around three paper proposals:

1. **Paper 1: Contour Certificates for Stable Learned Dynamics**  
   A theory-and-numerics paper developing scalar determinant and matrix-valued resolvent certificates for Schur stability of learned transition matrices.

2. **Paper 2: Stability Regularisation in Sequence Models**  
   A controlled empirical study comparing Rouché-faithful contour barriers, spectral penalties, Lyapunov penalties, and stable parameterisations for recurrent and SSM layers.

3. **Paper 3: Analytic Surrogates for Pathwise Robustness**  
   A high-risk, negative-results-oriented investigation into whether Rouché-type zero-counting can soundly certify absence of classifier boundary crossings along continuous input paths.

The immediate implementation priority is **Paper 1**. The codebase includes a starter Python package for computing contour margins, deterministic finite-grid certificates, random matrix testbeds, and synthetic certification experiments.

## Research philosophy

The project should avoid overclaiming. Rouché's theorem does not replace the spectral-radius condition for finite-dimensional Schur stability. Instead, the aim is to identify when Rouché-type perturbation and contour arguments provide useful, computable, and interpretable sufficient certificates, especially for structured learned dynamics and post-hoc verification.

## Initial milestones

- Implement scalar determinant and resolvent contour certificates.
- Validate deterministic grid certification with Lipschitz margins.
- Compare certificates on normal, non-normal, unstable, and near-defective matrices.
- Develop reference-selection algorithms for stable comparison matrices.
- Use Paper 1's certificate code as infrastructure for Paper 2.
