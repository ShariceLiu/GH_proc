"""Inspect Tecator joint MAP starts and compare spectral/dense gradients."""
import argparse
import json
import numpy as np
from scipy import optimize
from threadpoolctl import threadpool_limits
from GH_multiple_hmc import make_target, coordinates, MODELS
from hmc_tecator_experiment import load_data, DEFAULT_OUTPUT, PRIOR_CENTER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='tp_mu_gp')
    parser.add_argument('--from-gp', action='store_true')
    args = parser.parse_args()
    data = load_data(DEFAULT_OUTPUT)
    groups = [(data['x'], data['residuals'][data['train_rows']])]
    n = len(data['train_rows'])
    rng = np.random.default_rng(25026+MODELS.index(args.model))
    target = make_target(groups, model=args.model, prior_center=PRIOR_CENTER, backend='shared_grid')
    dense = make_target(groups, model=args.model, prior_center=PRIOR_CENTER)
    t = np.log(3/5) if args.model == 'tp_mu_gp' else np.arcsinh(1.)
    modes = []
    records = []
    for i in range(4):
        q = PRIOR_CENTER if args.model == 'gp' else np.r_[PRIOR_CENTER, np.full(n, t)]
        if args.from_gp:
            report = json.loads((DEFAULT_OUTPUT/'gp'/'fit.json').read_text())
            with np.load(DEFAULT_OUTPUT/'gp'/report['selected_samples']) as saved:
                gp_theta = saved['theta_log_samples'][i, 0]
            q = gp_theta if args.model == 'gp' else np.r_[gp_theta, np.zeros(n)]
        z = coordinates(q, args.model)+rng.normal(0., .5, len(q))
        if i % 2:
            z[[0, 1, 3, 4]] = z[[1, 0, 4, 3]]
        def objective(z):
            value, grad = target(z)
            return -value, -grad
        opt = optimize.minimize(objective, z, jac=True, method='L-BFGS-B',
                                options=dict(maxiter=600, ftol=1e-10, maxls=50))
        if args.from_gp:
            for _ in range(24):
                value, gradient = target(opt.x)
                if np.max(abs(gradient)) < 1.:
                    break
                local = optimize.minimize(objective, opt.x, jac=True, method='L-BFGS-B',
                                           bounds=list(zip(opt.x-1., opt.x+1.)),
                                           options=dict(maxiter=400, ftol=1e-11, maxls=50))
                if not np.isfinite(target(local.x)[0]) or target(local.x)[0] <= value:
                    break
                opt = local
        modes.append(opt.x)
        physical = coordinates(opt.x, args.model, inverse=True)
        exact, grad = dense(opt.x)
        value, spectral_grad = target(opt.x)
        print(i, opt.success, opt.message, 'density',value, 'theta',physical[:5],
              'W range', np.exp(physical[5:]).min() if len(physical)>5 else None,
              np.exp(physical[5:]).max() if len(physical)>5 else None,
              'max gradient',np.max(abs(spectral_grad)), 'dense delta',exact-value,
              'gradient delta',np.max(abs(spectral_grad-grad)), flush=True)
        records.append(dict(log_density=float(value), max_abs_gradient=float(np.max(abs(spectral_grad))),
                            physical_log_theta=physical[:5].tolist(),
                            dense_log_density_difference=float(exact-value)))
    suffix = '_from_gp' if args.from_gp else ''
    np.savez_compressed(DEFAULT_OUTPUT/f'initialization_diagnostic_{args.model}{suffix}.npz', modes=modes)
    (DEFAULT_OUTPUT/f'initialization_diagnostic_{args.model}{suffix}.json').write_text(json.dumps(records,indent=2))


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
