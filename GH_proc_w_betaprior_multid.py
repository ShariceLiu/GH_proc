"""2D example of beta/f inference with known SE parameters.

The imported Multivariate_GH and beta_f_inf also work for arbitrary D with
inputs shaped (N, D). Running this file makes a 2D example, saves heatmaps
and a slice with credible bands, and displays both figures.

Examples:
    python GH_proc_w_betaprior_multid.py
    python GH_proc_w_betaprior_multid.py --linear-beta --seed 5
    python GH_proc_w_betaprior_multid.py --no-show
"""

from pathlib import Path

import numpy as np

from GH_proc_w_betaprior import Multivariate_GH, beta_f_inf


def demo_2d(seed=5, grid_size=13, n_train=85, beta_prior=True,
            save_prefix="figure/betaprior_2d", show=True):
    """Simulate one joint realization, observe a subset, infer on the full grid.

    beta_prior=True draws beta from its GP prior; False generates the plane
    beta(x) = 0.25 + 0.5*x[0] - 0.25*x[1]. Inference always uses a zero-mean
    GP beta prior. All kernel parameters and the noise coefficient are fixed.
    Returns a dict with posterior, xs, fs, ys, beta, W and train_ids.
    Set save_prefix=None to skip saving, or show=False for a headless run.
    """
    if not isinstance(grid_size, (int, np.integer)) or grid_size < 3:
        raise ValueError("grid_size must be an integer >= 3")
    if not isinstance(n_train, (int, np.integer)) or not 1 <= n_train < grid_size**2:
        raise ValueError("n_train must be an integer between 1 and grid_size**2 - 1")

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = np.linspace(-3, 3, grid_size)
    x1, x2 = np.meshgrid(coordinates, coordinates, indexing="xy")
    xs = np.column_stack((x1.ravel(), x2.ravel()))  # (N, 2)
    parameters = dict(l=1, omega=1, v_0=0.2, w=[2.0, 1.2], k=0.05, mu=0,
                      sigma_v0=1, sigma_w=[0.25, 0.6])
    fs, ys, beta, W = Multivariate_GH(
        xs, **parameters, beta_prior=beta_prior, a_b=0.25, b_b=[0.5, -0.25],
        return_latents=True, seed=seed)
    rng = np.random.default_rng(seed + 1)
    train_ids = np.sort(rng.choice(len(xs), size=n_train, replace=False))
    test_ids = np.setdiff1d(np.arange(len(xs)), train_ids)
    posterior = beta_f_inf(ys[train_ids], xs[train_ids], x_pred=xs, **parameters)
    print(f"Inputs: {xs.shape}; observed: {n_train}; held out: {len(test_ids)}")
    print(f"True W: {W:.3f}; posterior mean W: {posterior['weights'] @ posterior['W']:.3f}")
    for name, truth in (("beta", beta), ("f", fs)):
        error = posterior[f"{name}_mean"][test_ids] - truth[test_ids]
        print(f"Held-out {name} RMSE: {np.sqrt(np.mean(error**2)):.3f}")
    print(f"Quadrature: {posterior['quadrature']}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (name, truth) in enumerate((("beta", beta), ("f", fs))):
        mean, std = posterior[f"{name}_mean"], posterior[f"{name}_std"]
        low, high = min(truth.min(), mean.min()), max(truth.max(), mean.max())
        panels = ((truth, f"True {name}", low, high),
                  (mean, f"Posterior mean {name}", low, high),
                  (std, f"Posterior std {name}", 0, std.max()))
        for col, (values, title, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            mesh = ax.pcolormesh(x1, x2, values.reshape(x1.shape), shading="auto",
                                 cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set(title=title, xlabel="x1", ylabel="x2", aspect="equal")
            fig.colorbar(mesh, ax=ax, shrink=0.85)
        axes[row, 1].scatter(xs[train_ids, 0], xs[train_ids, 1], s=10,
                             facecolors="none", edgecolors="white", linewidths=0.5)
    beta_kind = "GP beta" if beta_prior else "linear beta"
    fig.suptitle(f"2D inputs, {beta_kind}; circles mark observed locations", fontsize=14)

    # A fixed-x2 slice displays actual pointwise mixture intervals as bands.
    row_id = grid_size // 2
    slice_ids = np.arange(row_id * grid_size, (row_id + 1) * grid_size)
    observed_slice = np.intersect1d(slice_ids, train_ids)
    slice_fig, slice_axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7),
                                       constrained_layout=True)
    for ax, name, truth in zip(slice_axes, ("beta", "f"), (beta, fs)):
        band = posterior[f"{name}_ci"][slice_ids]
        ax.fill_between(coordinates, band[:, 0], band[:, 1], alpha=0.25,
                        label="95% pointwise credible interval")
        ax.plot(coordinates, truth[slice_ids], "k--", label=f"True {name}")
        ax.plot(coordinates, posterior[f"{name}_mean"][slice_ids], label="Posterior mean")
        ax.set_ylabel(name)
    slice_axes[1].scatter(xs[observed_slice, 0], ys[observed_slice], s=20,
                           color="tab:red", label="Observed y on slice")
    for ax in slice_axes:
        ax.legend()
    slice_axes[1].set_xlabel("x1")
    slice_fig.suptitle(f"Slice at x2 = {coordinates[row_id]:.2f}")

    if save_prefix is not None:
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(prefix.with_name(prefix.name + "_maps.png"), dpi=150)
        slice_fig.savefig(prefix.with_name(prefix.name + "_slice.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(slice_fig)
    return dict(posterior=posterior, xs=xs, fs=fs, ys=ys, beta=beta, W=W,
                train_ids=train_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--n-train", type=int, default=85)
    parser.add_argument("--linear-beta", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-prefix", default="figure/betaprior_2d")
    args = parser.parse_args()
    demo_2d(seed=args.seed, grid_size=args.grid_size, n_train=args.n_train,
            beta_prior=not args.linear_beta, save_prefix=args.save_prefix,
            show=not args.no_show)
