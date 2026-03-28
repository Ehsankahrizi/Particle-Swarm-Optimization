"""
Particle Swarm Optimization (PSO) for Eggholder and Schwefel Functions
Complete implementation with all 10 variants × 5 seeds for each function.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import json
import math

# ============================================================
# OBJECTIVE FUNCTIONS
# ============================================================

def eggholder(x):
    """Eggholder function (2D). x = [x1, x2], domain: [-512, 512]
    Global min: f(512, 404.2319) = -959.6407"""
    x1, x2 = x[0], x[1]
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return term1 + term2

def schwefel(x):
    """Schwefel function (n-D). domain: [-512, 512]
    Global min at (420.968746, ..., 420.968746) = 0"""
    n = len(x)
    alpha = 418.982887
    total = 0.0
    for i in range(n):
        total += -x[i] * np.sin(np.sqrt(np.abs(x[i])))
    return total + alpha * n


# ============================================================
# PSO IMPLEMENTATION
# ============================================================

class PSO:
    def __init__(self, obj_func, n_dims, bounds, n_particles=30, max_iter=500,
                 phi1=2.0, phi2=2.0, use_inertia=False, use_constriction=False,
                 topology='star', model='full', seed=42):
        """
        Parameters:
        -----------
        obj_func : callable - objective function to minimize
        n_dims : int - number of dimensions
        bounds : tuple (lb, ub) - same bounds for all dims
        n_particles : int - swarm size
        max_iter : int - max iterations
        phi1, phi2 : float - cognitive and social coefficients
        use_inertia : bool - whether to use inertia weight
        use_constriction : bool - whether to use constriction coefficient
        topology : 'star' or 'ring' - neighborhood topology
        model : 'full', 'cognition', 'social', 'selfless'
        seed : int - random seed
        """
        self.obj_func = obj_func
        self.n_dims = n_dims
        self.lb, self.ub = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.phi1 = phi1
        self.phi2 = phi2
        self.use_inertia = use_inertia
        self.use_constriction = use_constriction
        self.topology = topology
        self.model = model
        self.seed = seed

        # Velocity limits based on boundary range
        self.v_max = (self.ub - self.lb) / 2  # = 512 for [-512,512]
        self.v_min = -self.v_max

        # Constriction coefficient K
        self.K = 1.0
        if use_constriction:
            phi = phi1 + phi2
            if phi > 4:
                self.K = 2.0 / abs(2 - phi - math.sqrt(phi**2 - 4*phi))
            else:
                self.K = 1.0  # fallback

        # Inertia parameters
        self.w_start = 0.9
        self.w_end = 0.4

    def _get_neighbors(self, idx):
        """Get neighbor indices for particle idx based on topology."""
        if self.topology == 'star':
            return list(range(self.n_particles))
        elif self.topology == 'ring':
            left = (idx - 1) % self.n_particles
            right = (idx + 1) % self.n_particles
            return [left, idx, right]

    def _get_best_neighbor(self, idx, p_best_fitness):
        """Get index of best neighbor (excluding self for selfless model)."""
        neighbors = self._get_neighbors(idx)
        if self.model == 'selfless':
            # Exclude self from consideration for the best particle
            candidates = [n for n in neighbors if n != idx]
            if not candidates:
                candidates = neighbors
            best_n = candidates[0]
            for n in candidates[1:]:
                if p_best_fitness[n] < p_best_fitness[best_n]:
                    best_n = n
            return best_n
        else:
            best_n = neighbors[0]
            for n in neighbors[1:]:
                if p_best_fitness[n] < p_best_fitness[best_n]:
                    best_n = n
            return best_n

    def run(self):
        """Execute PSO. Returns (best_position, best_fitness, history)."""
        np.random.seed(self.seed)

        # Initialize positions uniformly in bounds
        positions = np.random.uniform(self.lb, self.ub, (self.n_particles, self.n_dims))

        # Initialize velocities in [-v_max/10, v_max/10] (small initial velocities)
        velocities = np.random.uniform(-5, 5, (self.n_particles, self.n_dims))

        # Evaluate initial fitness
        fitness = np.array([self.obj_func(positions[i]) for i in range(self.n_particles)])

        # Personal best
        p_best = positions.copy()
        p_best_fitness = fitness.copy()

        # Global best
        g_best_idx = np.argmin(p_best_fitness)
        g_best = p_best[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]

        history = [g_best_fitness]

        for t in range(self.max_iter):
            # Inertia weight (linear decrease)
            if self.use_inertia:
                w = self.w_start - (self.w_start - self.w_end) * (t / self.max_iter)
            else:
                w = 1.0  # No inertia: weight = 1.0

            for i in range(self.n_particles):
                r1 = np.random.uniform(0, 1, self.n_dims)
                r2 = np.random.uniform(0, 1, self.n_dims)

                # Get best neighbor for social component
                g_idx = self._get_best_neighbor(i, p_best_fitness)

                # Compute velocity components based on model
                cognitive = self.phi1 * r1 * (p_best[i] - positions[i])
                social = self.phi2 * r2 * (p_best[g_idx] - positions[i])

                if self.model == 'cognition':
                    # phi2 = 0 effectively
                    new_v = w * velocities[i] + cognitive
                elif self.model == 'social':
                    # phi1 = 0 effectively
                    new_v = w * velocities[i] + social
                elif self.model == 'selfless':
                    # phi1 = 0, g != i (already handled in _get_best_neighbor)
                    new_v = w * velocities[i] + social
                else:  # full model
                    new_v = w * velocities[i] + cognitive + social

                # Apply constriction
                if self.use_constriction:
                    new_v = self.K * new_v

                # If NOT using inertia (and not using constriction alone),
                # we need to handle velocity differently for the base case
                if not self.use_inertia and not self.use_constriction:
                    # Base PSO: v = v + cognitive + social (no w multiplier)
                    if self.model == 'cognition':
                        new_v = velocities[i] + cognitive
                    elif self.model in ('social', 'selfless'):
                        new_v = velocities[i] + social
                    else:
                        new_v = velocities[i] + cognitive + social

                # Restrict velocity
                for d in range(self.n_dims):
                    if new_v[d] > self.v_max:
                        new_v[d] = new_v[d] / abs(new_v[d]) * self.v_max
                    elif new_v[d] < self.v_min:
                        new_v[d] = new_v[d] / abs(new_v[d]) * self.v_max

                velocities[i] = new_v

                # Update position
                new_pos = positions[i] + velocities[i]

                # Boundary handling: clamp to bounds, reset velocity on that dim
                for d in range(self.n_dims):
                    if new_pos[d] < self.lb:
                        new_pos[d] = self.lb
                        velocities[i][d] = 0
                    elif new_pos[d] > self.ub:
                        new_pos[d] = self.ub
                        velocities[i][d] = 0

                positions[i] = new_pos

            # Evaluate fitness
            fitness = np.array([self.obj_func(positions[i]) for i in range(self.n_particles)])

            # Update personal bests
            for i in range(self.n_particles):
                if fitness[i] < p_best_fitness[i]:
                    p_best[i] = positions[i].copy()
                    p_best_fitness[i] = fitness[i]

            # Update global best
            current_best_idx = np.argmin(p_best_fitness)
            if p_best_fitness[current_best_idx] < g_best_fitness:
                g_best = p_best[current_best_idx].copy()
                g_best_fitness = p_best_fitness[current_best_idx]

            history.append(g_best_fitness)

        return g_best, g_best_fitness, history


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = [1, 2, 3, 4, 5]
DEFAULT_PARTICLES = 30
MAX_ITER = 500

def get_experiment_configs(default_particles=30):
    """Return list of experiment configurations for all 10 variants."""
    configs = [
        {
            'name': '1. Full Model (Star)',
            'short': 'Full_Star',
            'phi1': 2.0, 'phi2': 2.0,
            'use_inertia': False, 'use_constriction': False,
            'topology': 'star', 'model': 'full',
            'n_particles': default_particles
        },
        {
            'name': '2. Cognition Only (φ₂=0)',
            'short': 'Cognition',
            'phi1': 2.0, 'phi2': 0.0,
            'use_inertia': False, 'use_constriction': False,
            'topology': 'star', 'model': 'cognition',
            'n_particles': default_particles
        },
        {
            'name': '3. Social Only (φ₁=0)',
            'short': 'Social',
            'phi1': 0.0, 'phi2': 2.0,
            'use_inertia': False, 'use_constriction': False,
            'topology': 'star', 'model': 'social',
            'n_particles': default_particles
        },
        {
            'name': '4. Selfless (φ₁=0, g≠i)',
            'short': 'Selfless',
            'phi1': 0.0, 'phi2': 2.0,
            'use_inertia': False, 'use_constriction': False,
            'topology': 'star', 'model': 'selfless',
            'n_particles': default_particles
        },
        {
            'name': '5. Ring Topology (Full)',
            'short': 'Ring_Full',
            'phi1': 2.0, 'phi2': 2.0,
            'use_inertia': False, 'use_constriction': False,
            'topology': 'ring', 'model': 'full',
            'n_particles': default_particles
        },
        {
            'name': '6. Inertia (Star, Full)',
            'short': 'Inertia',
            'phi1': 2.0, 'phi2': 2.0,
            'use_inertia': True, 'use_constriction': False,
            'topology': 'star', 'model': 'full',
            'n_particles': default_particles
        },
        {
            'name': '7. Constriction (Star, Full)',
            'short': 'Constriction',
            'phi1': 2.05, 'phi2': 2.05,
            'use_inertia': False, 'use_constriction': True,
            'topology': 'star', 'model': 'full',
            'n_particles': default_particles
        },
        {
            'name': '8. Inertia + Constriction',
            'short': 'Inertia_Constr',
            'phi1': 2.05, 'phi2': 2.05,
            'use_inertia': True, 'use_constriction': True,
            'topology': 'star', 'model': 'full',
            'n_particles': default_particles
        },
        {
            'name': '9a. More Particles (Inertia+Constr)',
            'short': 'More_Particles',
            'phi1': 2.05, 'phi2': 2.05,
            'use_inertia': True, 'use_constriction': True,
            'topology': 'star', 'model': 'full',
            'n_particles': default_particles * 2  # double
        },
        {
            'name': '9b. Fewer Particles (Inertia+Constr)',
            'short': 'Fewer_Particles',
            'phi1': 2.05, 'phi2': 2.05,
            'use_inertia': True, 'use_constriction': True,
            'topology': 'star', 'model': 'full',
            'n_particles': max(10, default_particles // 2)  # half
        },
    ]
    return configs


def run_all_experiments(obj_func, n_dims, bounds, func_name,
                        default_particles=30, max_iter=500, output_dir='results'):
    """Run all 10 PSO variants × 5 seeds for a given objective function."""
    os.makedirs(output_dir, exist_ok=True)
    configs = get_experiment_configs(default_particles)
    all_results = []

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}  (particles={cfg['n_particles']})")
        print(f"{'='*60}")
        for seed in SEEDS:
            t0 = time.perf_counter()
            pso = PSO(
                obj_func=obj_func,
                n_dims=n_dims,
                bounds=bounds,
                n_particles=cfg['n_particles'],
                max_iter=max_iter,
                phi1=cfg['phi1'],
                phi2=cfg['phi2'],
                use_inertia=cfg['use_inertia'],
                use_constriction=cfg['use_constriction'],
                topology=cfg['topology'],
                model=cfg['model'],
                seed=seed
            )
            best_pos, best_fit, history = pso.run()
            runtime = time.perf_counter() - t0

            result = {
                'experiment': cfg['name'],
                'short': cfg['short'],
                'seed': seed,
                'n_particles': cfg['n_particles'],
                'phi1': cfg['phi1'],
                'phi2': cfg['phi2'],
                'inertia': cfg['use_inertia'],
                'constriction': cfg['use_constriction'],
                'topology': cfg['topology'],
                'model': cfg['model'],
                'best_position': best_pos.tolist(),
                'best_fitness': best_fit,
                'history': history,
                'runtime': runtime
            }
            all_results.append(result)
            pos_str = ', '.join([f'{p:.4f}' for p in best_pos])
            print(f"  Seed {seed}: f = {best_fit:12.4f}  at ({pos_str})  [{runtime:.2f}s]")

    return all_results


def generate_plots(all_results, func_name, output_dir, known_optimum=None):
    """Generate comprehensive plots for all experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    configs = get_experiment_configs()
    exp_names = [c['name'] for c in configs]
    exp_shorts = [c['short'] for c in configs]

    # ---- 1. Convergence curves per experiment ----
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    axes = axes.flatten()
    colors_seeds = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for idx, name in enumerate(exp_names):
        if idx >= 10:
            break
        ax = axes[idx]
        subset = [r for r in all_results if r['experiment'] == name]
        for i, r in enumerate(subset):
            ax.plot(r['history'], color=colors_seeds[i], alpha=0.8,
                    linewidth=1.2, label=f"Seed {r['seed']}")
        if known_optimum is not None:
            ax.axhline(known_optimum, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel('Best Fitness', fontsize=9)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='-.', alpha=0.4)

    for idx in range(10, 12):
        axes[idx].axis('off')

    plt.suptitle(f'{func_name} — Convergence Curves (All Variants)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/convergence_all.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---- 2. Summary boxplot ----
    fig, ax = plt.subplots(figsize=(14, 6))
    data_box = []
    labels_box = []
    for name, short in zip(exp_names, exp_shorts):
        subset = [r for r in all_results if r['experiment'] == name]
        data_box.append([r['best_fitness'] for r in subset])
        labels_box.append(short)

    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True)
    box_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12',
                  '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax.set_title(f'{func_name} — Best Fitness by PSO Variant', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', linestyle='-.', alpha=0.5)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_variants.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---- 3. Comparison bar chart (mean ± std) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    means = []
    stds = []
    for name in exp_names:
        subset = [r['best_fitness'] for r in all_results if r['experiment'] == name]
        means.append(np.mean(subset))
        stds.append(np.std(subset))
    x_pos = np.arange(len(exp_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=box_colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_shorts, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean Best Fitness', fontsize=12, fontweight='bold')
    ax.set_title(f'{func_name} — Mean Best Fitness ± Std (5 Seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', linestyle='-.', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mean_std_bar.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---- 4. Particle count comparison (variants 8, 9a, 9b) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    particle_variants = ['8. Inertia + Constriction', '9a. More Particles (Inertia+Constr)',
                         '9b. Fewer Particles (Inertia+Constr)']
    particle_labels = []
    for name in particle_variants:
        subset = [r for r in all_results if r['experiment'] == name]
        if subset:
            n_p = subset[0]['n_particles']
            particle_labels.append(f'N={n_p}')
            for i, r in enumerate(subset):
                ax.plot(r['history'], color=colors_seeds[i], alpha=0.7,
                        linewidth=1.0,
                        linestyle=['-', '--', ':'][particle_variants.index(name)],
                        label=f'N={n_p}, Seed {r["seed"]}' if i == 0 else '')
    if known_optimum is not None:
        ax.axhline(known_optimum, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax.set_title(f'{func_name} — Effect of Swarm Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='-.', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/particle_count_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---- 5. Individual convergence per experiment (for report) ----
    for idx, name in enumerate(exp_names):
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = [r for r in all_results if r['experiment'] == name]
        for i, r in enumerate(subset):
            ax.plot(r['history'], color=colors_seeds[i], alpha=0.8,
                    linewidth=1.5, label=f"Seed {r['seed']}")
        if known_optimum is not None:
            ax.axhline(known_optimum, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='Known Optimum')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        ax.set_title(f'{func_name} — {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='-.', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/conv_{exp_shorts[idx]}.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"All plots saved to {output_dir}/")


def generate_surface_plot(func, func_name, bounds, output_dir, known_min_pos=None):
    """Generate 3D surface and contour plots for 2D functions."""
    os.makedirs(output_dir, exist_ok=True)
    lb, ub = bounds
    x1 = np.linspace(lb, ub, 300)
    x2 = np.linspace(lb, ub, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func([X1[i, j], X2[i, j]])

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('x₁'); ax1.set_ylabel('x₂'); ax1.set_zlabel('f(x)')
    ax1.set_title(f'{func_name} — 3D Surface', fontsize=12, fontweight='bold')

    ax2 = fig.add_subplot(122)
    c = ax2.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(c, ax=ax2)
    if known_min_pos:
        ax2.plot(known_min_pos[0], known_min_pos[1], 'r*', markersize=15,
                 label=f'Global min ({known_min_pos[0]}, {known_min_pos[1]})')
        ax2.legend()
    ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')
    ax2.set_title(f'{func_name} — Contour Plot', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/surface_plot.png', dpi=200, bbox_inches='tight')
    plt.close()


def print_summary_table(all_results, func_name):
    """Print a formatted summary table of all results."""
    configs = get_experiment_configs()
    exp_names = [c['name'] for c in configs]

    print(f"\n{'='*100}")
    print(f"  SUMMARY TABLE — {func_name}")
    print(f"{'='*100}")
    print(f"{'Variant':<40} {'Mean':>12} {'Std':>12} {'Best':>12} {'Worst':>12} {'Particles':>10}")
    print(f"{'-'*100}")

    for name in exp_names:
        subset = [r['best_fitness'] for r in all_results if r['experiment'] == name]
        n_p = [r['n_particles'] for r in all_results if r['experiment'] == name][0]
        mean_f = np.mean(subset)
        std_f = np.std(subset)
        best_f = np.min(subset)
        worst_f = np.max(subset)
        print(f"{name:<40} {mean_f:>12.4f} {std_f:>12.4f} {best_f:>12.4f} {worst_f:>12.4f} {n_p:>10}")

    print(f"{'='*100}")

    # Overall best
    best_r = min(all_results, key=lambda r: r['best_fitness'])
    pos_str = ', '.join([f'{p:.4f}' for p in best_r['best_position']])
    print(f"\nOverall best: {best_r['experiment']}, Seed={best_r['seed']}")
    print(f"  f* = {best_r['best_fitness']:.6f}  at ({pos_str})")


def save_results_csv(all_results, func_name, output_dir):
    """Save results to CSV format."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = f'{output_dir}/results_{func_name.lower().replace(" ", "_")}.csv'
    with open(filepath, 'w') as f:
        f.write('Experiment,Seed,Particles,Phi1,Phi2,Inertia,Constriction,Topology,Model,Best_Fitness,Position,Runtime\n')
        for r in all_results:
            pos_str = ';'.join([f'{p:.6f}' for p in r['best_position']])
            f.write(f"{r['short']},{r['seed']},{r['n_particles']},{r['phi1']},{r['phi2']},"
                    f"{r['inertia']},{r['constriction']},{r['topology']},{r['model']},"
                    f"{r['best_fitness']:.6f},{pos_str},{r['runtime']:.3f}\n")
    print(f"Results saved to {filepath}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    BASE_DIR = '/sessions/zen-busy-maxwell/PSO_Results'
    os.makedirs(BASE_DIR, exist_ok=True)

    # ============================================================
    # PART 1: EGGHOLDER FUNCTION (2D)
    # ============================================================
    print("\n" + "="*70)
    print("  PART 1: EGGHOLDER FUNCTION (2D)")
    print("  Global minimum: f(512, 404.2319) = -959.6407")
    print("="*70)

    egg_dir = f'{BASE_DIR}/Eggholder'
    os.makedirs(egg_dir, exist_ok=True)

    # Surface plot
    generate_surface_plot(eggholder, 'Eggholder', (-512, 512), egg_dir,
                          known_min_pos=[512, 404.2319])

    # Run all experiments: 30 particles, 500 iterations
    egg_results = run_all_experiments(
        obj_func=eggholder, n_dims=2, bounds=(-512, 512),
        func_name='Eggholder', default_particles=30,
        max_iter=500, output_dir=egg_dir
    )

    # Generate plots
    generate_plots(egg_results, 'Eggholder', egg_dir, known_optimum=-959.6407)
    print_summary_table(egg_results, 'Eggholder')
    save_results_csv(egg_results, 'Eggholder', egg_dir)

    # ============================================================
    # PART 2: SCHWEFEL FUNCTION (4D) — BONUS
    # ============================================================
    print("\n" + "="*70)
    print("  PART 2: SCHWEFEL FUNCTION (4D) — BONUS")
    print("  Global minimum: f(420.9687, ..., 420.9687) = 0")
    print("="*70)

    schw_dir = f'{BASE_DIR}/Schwefel'
    os.makedirs(schw_dir, exist_ok=True)

    # Surface plot (2D slice for visualization)
    generate_surface_plot(schwefel, 'Schwefel (2D slice)', (-512, 512), schw_dir,
                          known_min_pos=[420.9687, 420.9687])

    # Run all experiments: 50 particles for 4D, 500 iterations, SAME SEEDS
    schw_results = run_all_experiments(
        obj_func=schwefel, n_dims=4, bounds=(-512, 512),
        func_name='Schwefel', default_particles=50,
        max_iter=500, output_dir=schw_dir
    )

    # Generate plots
    generate_plots(schw_results, 'Schwefel (4D)', schw_dir, known_optimum=0.0)
    print_summary_table(schw_results, 'Schwefel (4D)')
    save_results_csv(schw_results, 'Schwefel', schw_dir)

    # Save all results as JSON for report generation
    def make_serializable(results):
        out = []
        for r in results:
            r2 = dict(r)
            r2['best_position'] = [float(x) for x in r2['best_position']]
            r2['history'] = [float(x) for x in r2['history']]
            out.append(r2)
        return out

    with open(f'{BASE_DIR}/egg_results.json', 'w') as f:
        json.dump(make_serializable(egg_results), f)
    with open(f'{BASE_DIR}/schw_results.json', 'w') as f:
        json.dump(make_serializable(schw_results), f)

    print("\n\nAll experiments complete!")
    print(f"Total Eggholder runs: {len(egg_results)}")
    print(f"Total Schwefel runs: {len(schw_results)}")
    print(f"Grand total: {len(egg_results) + len(schw_results)} runs")
