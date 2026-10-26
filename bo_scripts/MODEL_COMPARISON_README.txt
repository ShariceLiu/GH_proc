Eight-model Bayesian optimization comparison
==========================================
Run from repository root in the existing Python environment:
  python -B bo_scripts/run_model_comparison.py --seed 0 --steps 5
All eight models and MAP/HMC/slice run by default (24 trajectories).
Five shared initial points, then five BO evaluations per trajectory.
Continue ALL completed trajectories to ten total BO steps:
  python -B bo_scripts/run_model_comparison.py --seed 0 --steps 10
New seed:
  python -B bo_scripts/run_model_comparison.py --seed 1 --steps 5
Subset, or append previously omitted methods/models with the same settings:
  python -B bo_scripts/run_model_comparison.py --seed 0 --steps 5 --methods map
  python -B bo_scripts/run_model_comparison.py --seed 0 --steps 5 --models ghp_beta_gp ghp_mu_gp

Models (zero statements refer to standardized outputs):
  ghp_const_beta: scalar inferred mu and beta; f=mu+W beta+sqrt(W)g.
  ghp_beta_gp: beta~GP(0,Kbeta), mu=0; f=W beta(x)+sqrt(W)g(x).
  ghp_mu_gp: mu~GP(0,Kmu), beta=0; f=mu(x)+sqrt(W)g(x), W~GIG(1,1,1).
  gp_mu_gp: independent GP mu and residual g; f=mu(x)+g(x), W=1.
  stp_mu_gp: independent GP mu plus Student-t residual; W~IG(5/2,3/2).
All kernels are squared exponential; GHP lambda=omega=1 and STP nu=5 fixed.
STP mu is an independent GP mean plus a t residual, not a plain t process
over the sum of both kernels. Latent GPs are integrated analytically.

MAP: same fixed-function plug-in approach as map_fixed_function_30_experiment:
joint log(theta),log(W) mode, conditional GP mean for beta/mu, freeze the
function and recompute/integrate W. GP has no W. Constant beta/mu are fixed
at their joint posterior mode; W is then integrated conditional on them.
HMC: existing adaptive HMC (NOT NUTS), joint theta,W, retaining latent GP
uncertainty in Gaussian conditional predictions. Slice: same posterior using
coordinate-wise stepping-out/shrinkage. Both use posterior-corrected kernel
exchanges where there are two components. For constant beta, EI integrates W
conditional on each theta draw to reduce Monte Carlo error.

This constant-beta arm is a new, harmonized experiment: SE instead of Matern,
inferred noise instead of a fixed nugget, joint-W MAP instead of marginal-W
MAP. Do not treat its results as continuation of gh_fixed_mixing_pilot.
The MAP vs MCMC comparison also changes whether GP-function uncertainty is
retained; it does not isolate only hyperparameter uncertainty.

Common setup: original objective on [5,10], internal inputs [0,1], initial
five-point Latin hypercube, output centering/scaling frozen at initialization.
No observation noise generated. Infer k, conditionally scaled by W except GP.
Two-kernel prior medians [v0,latent_v0,k,w,latent_w]=[.5,.5,1e-4,25,6.25];
independent log Normal SD 2. Constant arm uses the same residual/noise/weight
priors, scalar mu~N(0,4), beta~N(0,1). w is inverse squared lengthscale.
Priors are fixed across updates. No held-out objective values inform fitting.

Defaults: 2 chains, 256 warmup, 512 retained draws per chain, 64 EI draws,
8 MAP starts, target acceptance .95. HMC uses randomized 12-30 leapfrog steps.
These are pilot settings; inspect diagnostic flags before scientific use.
Options: --warmup --samples --chains --map-starts --target-accept
--acquisition-draws. Changes require a new --output directory.

Output: bo_figure/eight_model_comparison/seed_000/
  experiment.json: model definitions, priors, scaling, settings, source hashes
  summary.csv: all completed model/method outcomes and timing
  comparison.png: regret by model, three panels for inference methods
  MODEL/METHOD/checkpoint.json: atomic continuation state
  MODEL/METHOD/observations.csv: initial points and completed queries
  MODEL/METHOD/step_NNN/: posterior chains, EI grid/plot, diagnostics, timing
--steps is TOTAL BO evaluations after initialization, not additional steps.
Per-model/method/update seeds and chain endpoints ensure reproducible resume.
Existing completed steps are skipped; interrupted uncommitted steps replay.

Tests: python -B bo_scripts/test_model_comparison.py
Checks gradients for all eight models, constant-model normalized joint density,
Gaussian predictions against regression code, MAP predictive moments against
fixed-function regression, Student-t EI against its analytic expression,
and exact split/uninterrupted trajectories for all 24 combinations.
Validation smoke/resume folders contain short test runs, not research results.

Zero-mean additions:
  gp_zero: f=g, W=1; mu=beta=0.
  stp_zero: f=sqrt(W)g, W~IG(5/2,3/2); mu=beta=0.
  ghp_zero: f=sqrt(W)g, W~GIG(1,1,1); mu=beta=0.
Each uses ONE residual SE kernel, with no latent mean or skew GP. Priors
on log(v0,k,w) match the existing residual priors. STP/GHP sample or optimize
log W jointly with kernel/noise parameters; prediction integrates W again
conditional on each parameter setting for all three inference methods.
GP samples/optimizes only the three kernel/noise parameters.
Zero mean refers to the standardized objective (initial-data centering is
still applied identically to all eight models).

Run only the additions:
  python -B bo_scripts/run_model_comparison.py --seed 0 --steps 5 --models gp_zero stp_zero ghp_zero
The new default output is eight_model_comparison. Existing five-model
results are left intact; their old source hashes will not match this version.
Use the new output folder for the expanded comparison.
