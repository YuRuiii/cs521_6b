"""
Bitcoin Selfish Mining Simulator
================================
Entry point for running experiments and generating visualizations.

This simulator reproduces the key results from:
  'Majority is not Enough: Bitcoin Mining is Vulnerable'
  (Eyal & Sirer, 2014)

Usage:
    python main.py                  # Run all experiments
    python main.py --experiment 1   # Run a specific experiment (1-5)
"""

import argparse
import numpy as np

from simulator import (
    SelfishMiningSimulator,
    theoretical_selfish_revenue,
    selfish_mining_threshold,
)
from strategies import (
    MiningPool,
    Miner,
    simulate_pool_mining,
    create_hash_rate_distribution,
)
from visualization import (
    plot_selfish_mining_revenue,
    plot_threshold_vs_gamma,
    plot_revenue_heatmap,
    plot_pool_comparison,
    plot_convergence,
)

SEED = 42


def _section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


def experiment_1_revenue_curves():
    _section("Experiment 1: Selfish Mining Revenue vs. Hash Rate")

    alphas = np.linspace(0.01, 0.49, 25)
    gammas = [0.0, 0.25, 0.5, 1.0]

    print(f"  alpha: {alphas[0]:.2f}..{alphas[-1]:.2f} ({len(alphas)} points)")
    print(f"  gamma: {gammas}")
    print(f"  rounds: 100,000")

    print("\n  profitability thresholds:")
    for g in gammas:
        t = selfish_mining_threshold(g)
        print(f"    gamma={g:.2f} -> alpha* = {t:.4f} ({t*100:.1f}%)")

    plot_selfish_mining_revenue(
        alpha_values=alphas,
        gamma_values=gammas,
        num_rounds=100_000,
        seed=SEED,
        save_path="fig1_revenue_curves.png",
    )


def experiment_2_threshold_analysis():
    """
    Experiment 2: How the profitability threshold varies with gamma.
    """
    print("=" * 60)
    print("Experiment 2: Profitability Threshold vs. Gamma")
    print("=" * 60)

    print("  gamma=0.0 (no network advantage): alpha* = 1/3 = 33.3%")
    print("  gamma=0.5 (moderate advantage):   alpha* = 1/4 = 25.0%")
    print("  gamma=1.0 (full advantage):        alpha* -> 0%")

    plot_threshold_vs_gamma(save_path="fig2_threshold_vs_gamma.png")


def experiment_3_heatmap():
    """
    Experiment 3: Revenue advantage heatmap across (alpha, gamma) space.
    """
    print("=" * 60)
    print("Experiment 3: Revenue Advantage Heatmap")
    print("=" * 60)

    print("  Scanning alpha x gamma parameter space...")
    print("  Green = honest mining better, Red = selfish mining better")

    plot_revenue_heatmap(
        num_rounds=50_000,
        seed=42,
        save_path="fig3_heatmap.png",
    )


def experiment_4_pool_strategies():
    """
    Experiment 4: Compare PPS vs PPLNS pool reward distribution.
    """
    print("=" * 60)
    print("Experiment 4: Mining Pool Strategy Comparison (PPS vs PPLNS)")
    print("=" * 60)

    pool_results = {}

    for strategy in ["pps", "pplns"]:
        pool = MiningPool(name=f"Pool_{strategy.upper()}", strategy=strategy, pplns_window=200)
        miners = create_hash_rate_distribution(
            num_miners=5,
            distribution="zipf",
            selfish_fraction=0.3,
            seed=42,
        )
        for m in miners:
            pool.add_member(m)

        result = simulate_pool_mining(
            pool=pool,
            num_rounds=20_000,
            block_reward=6.25,
            seed=42,
        )
        pool_results[strategy.upper()] = result

        print(f"\n  {strategy.upper()} strategy results:")
        for name, data in result.items():
            print(f"    {name}: hash_rate={data['hash_rate']:.3f}, "
                  f"revenue={data['revenue']:.2f}")

    plot_pool_comparison(pool_results, save_path="fig4_pool_comparison.png")


def experiment_5_convergence():
    _section("Experiment 5: Simulation Convergence")

    alpha, gamma = 0.3, 0.5
    theo = theoretical_selfish_revenue(alpha, gamma)
    print(f"  alpha={alpha}, gamma={gamma}")
    print(f"  theoretical = {theo:.4f}")
    print(f"  honest      = {alpha:.4f}")
    print(f"  advantage   = {theo - alpha:.4f}")

    plot_convergence(
        alpha=alpha, gamma=gamma,
        max_rounds=500_000, checkpoints=50,
        seed=SEED, save_path="fig5_convergence.png",
    )


def run_quick_demo():
    print("\n" + "=" * 60)
    print("Quick Demo: Selfish Mining Simulation")
    print("=" * 60)

    cases = [
        (0.10, 0.0), (0.10, 0.5),
        (0.25, 0.0), (0.25, 0.5),
        (0.33, 0.0), (0.33, 0.5),
        (0.40, 0.0), (0.40, 0.5),
        (0.45, 0.0), (0.45, 0.5),
    ]

    header = f"\n{'alpha':>6} {'gamma':>6} {'Simulated':>10} {'Theoretical':>12} {'Honest':>8} {'Profit?':>9}"
    print(header)
    print("-" * 60)

    for alpha, gamma in cases:
        sim = SelfishMiningSimulator(alpha, gamma, seed=SEED)
        r = sim.run(num_rounds=200_000)
        theo = theoretical_selfish_revenue(alpha, gamma)
        profit = "YES" if alpha > selfish_mining_threshold(gamma) else "no"

        print(f"{alpha:>6.2f} {gamma:>6.1f} "
              f"{r.selfish_revenue:>10.4f} {theo:>12.4f} "
              f"{alpha:>8.4f} {profit:>9}")


def main():
    parser = argparse.ArgumentParser(
        description="Bitcoin Selfish Mining Simulator"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a specific experiment (1-5). Default: run all.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick text-only demo without plots.",
    )
    args = parser.parse_args()

    if args.demo:
        run_quick_demo()
        return

    experiments = {
        1: experiment_1_revenue_curves,
        2: experiment_2_threshold_analysis,
        3: experiment_3_heatmap,
        4: experiment_4_pool_strategies,
        5: experiment_5_convergence,
    }

    if args.experiment:
        experiments[args.experiment]()
    else:
        run_quick_demo()
        for exp_fn in experiments.values():
            exp_fn()


if __name__ == "__main__":
    main()
