Varying-beta GHP BO (separate from the constant-beta experiment)
=============================================================
Run from the repository root in the existing Python environment:
  python -B bo_scripts/run_beta_gp.py --seed 0 --steps 5
Resume to ten TOTAL BO steps (not ten more):
  python -B bo_scripts/run_beta_gp.py --seed 0 --steps 10
New initial design:
  python -B bo_scripts/run_beta_gp.py --seed 1 --steps 5
Optional selection: --methods map hmc slice (default: all three).
Tests: python -B bo_scripts/test_beta_gp_bo.py

Outputs: bo_figure/gh_beta_gp_pilot/seed_000/{map,hmc,slice}/
Root experiment.json freezes settings, scaling, priors and source hashes.
Root summary.csv and comparison.png compare regret and timing.
Per-method checkpoint.json and observations.csv retain continuation state.
step_NNN contains full effective-coordinate chains, HMC diagnostic arrays,
acquisition ensemble/grid/plot, diagnostics.json and selection.json.
Interrupted steps are replayed from deterministic seeds. Completed steps are
skipped. Resume preserves chain endpoints and adapts anew on updated data.
Changing inference settings or source requires a new --output directory.

The model matches GH_proc_joint_hmc.py's beta_gp branch:
  beta ~ GP(0,Kbeta), g ~ GP(0,Kf), f=W beta+sqrt(W)g.
  lambda=omega=1, mu=0; covariance after integrating beta is W Kf+W^2 Kbeta.
Both kernels are SE. The observation covariance also has W*k*I.
There is no simulated noise, but k is inferred as in the regression reference.
Log-prior medians on [v0,beta_v0,k,w,beta_w] are [.5,.5,1e-4,25,6.25],
with SD 2 on logs. w is inverse squared lengthscale. These priors remain
fixed across BO steps. Five initial observations determine fixed centering
and scaling; mu=0 refers to standardized outputs, not the raw objective.

MAP: multistart log(theta),log(W) joint mode with Gaussian processes integrated;
estimate beta's conditional GP mean, freeze that function and the kernels,
recompute and integrate the GIG posterior of W for latent-f EI. This is the
plug-in approximation in map_fixed_function_30_experiment.py, NOT full Bayes.
HMC: reuse the existing adaptive HMC, NOT NUTS; 12-30 random leapfrog steps,
diagonal metric adaptation, posterior-corrected kernel-component exchanges.
Slice: sample the same joint density in effective log coordinates, using
coordinate-wise stepping out/shrinkage and the same component exchanges.
Both MCMC methods retain beta-function uncertainty analytically in each
conditional Gaussian; they do not sample beta values at every observation.

Defaults: 2 chains, 256 warmup + 512 draws/chain, 64 draws for EI; 8 MAP starts;
HMC target acceptance .95. Pilot settings, not guaranteed converged inference.
Inspect rank-normalized Rhat, bulk/tail ESS and divergences; diagnostics_ok
requires Rhat<1.01, ESS>=400 and zero retained HMC divergences. Scalar W
quadrature for plug-in EI uses 96 log-space nodes; EI has zero margin.
Use --warmup, --samples, --chains, --map-starts, --target-accept and
--acquisition-draws to configure NEW experiments. The numerical comparisons
test EI against adaptive integration and predictions against the existing
regression implementations. Resume is tested for all three methods.

Validation-only outputs are kept separately in gh_beta_gp_validation_smoke
and gh_beta_gp_validation_resume. Their very short chains deliberately test
execution and reproducibility, not optimization quality or convergence.
