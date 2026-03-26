"""
Advanced Example: Parameter Optimization with Grid Search

This example demonstrates how to optimize kernel parameters and 
explore the performance of the KKF with different configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.gaussian_process.kernels import Matern, RBF, ExpSineSquared

from KKF import DynamicalSystem, KoopmanOperator
from KKF.applyKKF import apply_koopman_kalman_filter


def evaluate_filter(system, kernel, n_features, observations, initial_dist, noise_samples=50):
    """Evaluate filter performance for a given kernel."""
    try:
        koop = KoopmanOperator(kernel, system)
        solution = apply_koopman_kalman_filter(
            koopman_operator=koop,
            observations=observations,
            initial_distribution=initial_dist,
            n_features=n_features,
            optimize=False,
            noise_samples=noise_samples,
        )
        return solution
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    """Run advanced optimization example."""
    print("=" * 70)
    print("KKF Advanced Example: Parameter Exploration")
    print("=" * 70)

    # Create nonlinear system
    def f(x):
        """Nonlinear dynamics."""
        return np.array([
            0.5 * x[0] + np.sin(x[1]),
            0.7 * x[1] - 0.1 * x[0] ** 2
        ])

    def g(x):
        """Nonlinear observation."""
        return np.array([x[0] + 0.1 * x[1] ** 2])

    nx, ny = 2, 1
    X_dist = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
    dyn_noise = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
    obs_noise = stats.multivariate_normal(mean=np.zeros(ny), cov=0.02 * np.eye(ny))

    system = DynamicalSystem(
        nx=nx,
        ny=ny,
        f=f,
        g=g,
        dist_X=X_dist,
        dist_dyn=dyn_noise,
        dist_obs=obs_noise,
        discrete_time=True,
    )

    print(f"\nNonlinear System created:")
    print(f"  Dynamics: x[k+1] = [0.5*x[0] + sin(x[1]), 0.7*x[1] - 0.1*x[0]²]")
    print(f"  Observation: y[k] = x[0] + 0.1*x[1]²")

    # Generate data
    print(f"\nGenerating synthetic data...")
    n_timesteps = 80
    x_true = np.zeros((n_timesteps, nx))
    y_meas = np.zeros((n_timesteps, ny))

    x_true[0] = np.array([1.0, 0.5])
    y_meas[0] = g(x_true[0]) + obs_noise.rvs()

    np.random.seed(42)
    for t in range(1, n_timesteps):
        x_true[t] = f(x_true[t - 1]) + dyn_noise.rvs()
        y_meas[t] = g(x_true[t]) + obs_noise.rvs()

    print(f"  ✓ Generated {n_timesteps} timesteps")

    # Kernel parameter exploration
    print(f"\nExploring different kernel configurations...")

    kernels_to_test = {
        "RBF (0.5)": RBF(length_scale=0.5),
        "RBF (1.0)": RBF(length_scale=1.0),
        "RBF (2.0)": RBF(length_scale=2.0),
        "Matern (0.5, ν=0.5)": Matern(length_scale=0.5, nu=0.5),
        "Matern (1.0, ν=0.5)": Matern(length_scale=1.0, nu=0.5),
        "Matern (1.0, ν=1.5)": Matern(length_scale=1.0, nu=1.5),
        "Matern (1.0, ν=2.5)": Matern(length_scale=1.0, nu=2.5),
        "ExpSineSquared": ExpSineSquared(length_scale=1.0, periodicity=1.0),
    }

    n_features_list = [10, 20, 30]
    results = {}

    for n_feat in n_features_list:
        results[n_feat] = {}
        print(f"\n  Testing with {n_feat} features:")

        for kernel_name, kernel in kernels_to_test.items():
            print(f"    Testing {kernel_name}...", end=" ")

            solution = evaluate_filter(
                system,
                kernel,
                n_feat,
                y_meas,
                stats.multivariate_normal(mean=x_true[0], cov=0.1 * np.eye(nx)),
                noise_samples=50,
            )

            if solution is not None:
                error = np.mean(np.linalg.norm(solution.x_plus - x_true, axis=1))
                results[n_feat][kernel_name] = error
                print(f"✓ Error: {error:.6f}")
            else:
                results[n_feat][kernel_name] = np.nan
                print(f"✗ Failed")

    # Print summary
    print(f"\n" + "=" * 70)
    print("Summary of Results (Mean Absolute Error):")
    print("=" * 70)
    print(f"{'Kernel Config':<30} {'10 Features':<15} {'20 Features':<15} {'30 Features':<15}")
    print("-" * 75)

    for kernel_name in kernels_to_test.keys():
        e10 = results[10].get(kernel_name, np.nan)
        e20 = results[20].get(kernel_name, np.nan)
        e30 = results[30].get(kernel_name, np.nan)

        print(
            f"{kernel_name:<30} {e10:<15.6f} {e20:<15.6f} {e30:<15.6f}"
        )

    # Find best configuration
    best_config = None
    best_error = np.inf

    for n_feat, kernel_results in results.items():
        for kernel_name, error in kernel_results.items():
            if not np.isnan(error) and error < best_error:
                best_error = error
                best_config = (n_feat, kernel_name)

    print("\n" + "=" * 70)
    if best_config:
        print(f"Best Configuration: {best_config[1]} with {best_config[0]} features")
        print(f"Mean Absolute Error: {best_error:.6f}")
    print("=" * 70)

    # Visualization of results
    print(f"\nGenerating comparison plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error vs number of features for different kernels
    for kernel_name in ["RBF (1.0)", "Matern (1.0, ν=0.5)", "Matern (1.0, ν=2.5)"]:
        errors = []
        features = sorted(results.keys())
        for n_feat in features:
            err = results[n_feat].get(kernel_name, np.nan)
            errors.append(err if not np.isnan(err) else None)

        # Filter out None values
        valid_idx = [i for i, e in enumerate(errors) if e is not None]
        if valid_idx:
            axes[0].plot(
                [features[i] for i in valid_idx],
                [errors[i] for i in valid_idx],
                "o-",
                label=kernel_name,
                linewidth=2,
                markersize=8,
            )

    axes[0].set_xlabel("Number of Features")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].set_title("Filter Performance vs Number of Features")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Heatmap of errors
    kernel_names = list(kernels_to_test.keys())
    error_matrix = np.zeros((len(kernel_names), len(n_features_list)))

    for i, kernel_name in enumerate(kernel_names):
        for j, n_feat in enumerate(n_features_list):
            error_matrix[i, j] = results[n_feat].get(kernel_name, np.nan)

    im = axes[1].imshow(error_matrix, cmap="RdYlGn_r", aspect="auto")
    axes[1].set_xticks(np.arange(len(n_features_list)))
    axes[1].set_yticks(np.arange(len(kernel_names)))
    axes[1].set_xticklabels(n_features_list)
    axes[1].set_yticklabels(kernel_names, fontsize=9)
    axes[1].set_xlabel("Number of Features")
    axes[1].set_title("Error Heatmap (lower is better)")

    # Add text annotations
    for i in range(len(kernel_names)):
        for j in range(len(n_features_list)):
            text = axes[1].text(
                j,
                i,
                f"{error_matrix[i, j]:.4f}",
                ha="center",
                va="center",
                color="black" if error_matrix[i, j] < np.nanmax(error_matrix) / 2 else "white",
                fontsize=8,
            )

    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label("Mean Absolute Error")

    plt.tight_layout()
    plt.savefig("kkf_parameter_exploration.png", dpi=100, bbox_inches="tight")
    print("  Figure saved as 'kkf_parameter_exploration.png'")

    plt.show()

    print("\n" + "=" * 70)
    print("Advanced example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
