Resumable GHP Bayesian optimization pilot
========================================
Run from the repository root with the venv Python interpreter:
  python -B bo_scripts/test_bo.py
  python -B bo_scripts/run_pilot.py --seed 0 --steps 5
Continue the SAME experiments to 10 total BO steps:
  python -B bo_scripts/run_pilot.py --seed 0 --steps 10

--steps means TOTAL BO steps, excluding the five initial observations.
--methods map nuts slice selects methods (all three are the default).
Completed steps are skipped. Checkpoint writes are atomic; interrupted,
uncommitted steps are replayed with their deterministic per-step seed.
Resume preserves initial scaling, prior, observations, and chain endpoints.
NUTS adapts anew at each update, starting from previous chain endpoints.
Source hashes and sampler settings are checked on resume to prevent mixing
different implementations. To change a model/setting, use --output NEW_FOLDER.

Dependencies: numpy scipy torch matplotlib pyro-ppl pyro-api opt_einsum tqdm.
Pyro 1.9.2 and pyro-api 0.1.2 are installed locally in ignored _vendor/.
Install if needed: python -m pip install --target bo_scripts/_vendor pyro-ppl pyro-api --no-deps

Model: f = mu + beta W + sqrt(W) g, g ~ GP(0, v0 Matern52(lengthscale)).
GIG W has lambda=omega=1. Constant mu and beta are inferred, alongside log v0
and log lengthscale. Nugget=1e-6 is inside W-scaled covariance; no simulated
observation noise. Priors on [mu,beta,log v0,log ell] are independent normals
with means [0,0,0,log(.2)] and SDs [2,1,1.5,1]. MAP is in these coordinates.
Original x in [5,10] maps to u in [0,1]. Output scaling is fixed from initial
five observations. Reference optimum is calculated independently for scoring.

Pilot settings: 2 chains, 128 warmup + 256 retained draws per chain per update;
32 draws for acquisition integration, 96-node log-W quadrature, 1001 grid
points plus local acquisition refinement; EI margin zero. No duplicate queries.
These SHORT PILOT CHAINS are not intended as evidence of posterior convergence.
Report saved split R-hat, ESS and NUTS divergences before scientific comparison.
MAP uses six optimization starts. Slice sampling uses stepping-out/shrinkage
on the marginal hyperparameter posterior, without auxiliary latent functions.

Outputs: bo_figure/gh_fixed_mixing_pilot/seed_000/
  experiment.json             frozen configuration, scaling, source hashes
  requested_budget.json       last requested total BO budget
  summary.csv, comparison.png aggregate objective/regret/timing results
  {map,nuts,slice}/
    checkpoint.json           authoritative continuation state
    observations.csv          initial points and all completed BO queries
    step_NNN/inference.npz     full chains and draws used for EI
    step_NNN/diagnostics.json  sampler or MAP diagnostics
    step_NNN/acquisition.npz   evaluated grid and EI values
    step_NNN/selection.json    chosen point, objective and timing

Tests independently check the GH density by numerical mixing integration,
gradients by finite differences, conditional GIG moments by Bessel ratios,
EI against adaptive integration, and resumed vs uninterrupted MAP trajectories.
