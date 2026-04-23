"""Plots for the selfish mining experiments."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from simulator import (
    SelfishMiningSimulator,
    selfish_mining_threshold,
    theoretical_selfish_revenue,
)

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
MARKERS = ["o", "s", "^", "D"]


def _save_and_show(save_path: Optional[str]):
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.show()


def plot_selfish_mining_revenue(
    alpha_values: np.ndarray,
    gamma_values: List[float],
    num_rounds: int = 100_000,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(10, 7))

    # honest miner baseline
    ax.plot(alpha_values, alpha_values, "k--", linewidth=2, label="Honest (revenue = α)")

    for i, gamma in enumerate(gamma_values):
        sim_vals = []
        theo_vals = []
        for a in alpha_values:
            sim = SelfishMiningSimulator(a, gamma, seed=seed)
            sim_vals.append(sim.run(num_rounds).selfish_revenue)
            theo_vals.append(theoretical_selfish_revenue(a, gamma))

        c = COLORS[i % len(COLORS)]
        ax.plot(alpha_values, theo_vals, color=c, linewidth=2, label=f"Theoretical (γ={gamma})")
        ax.scatter(
            alpha_values, sim_vals,
            color=c, marker=MARKERS[i % len(MARKERS)],
            s=40, alpha=0.7, zorder=5, label=f"Simulated (γ={gamma})",
        )

        t = selfish_mining_threshold(gamma)
        ax.axvline(x=t, color=c, linestyle=":", alpha=0.5, linewidth=1)
        ax.annotate(f"α*={t:.3f}", xy=(t, 0.02 + i * 0.03),
                    fontsize=8, color=c, ha="center")

    ax.set_xlabel("Selfish Miner Hash Rate (α)", fontsize=13)
    ax.set_ylabel("Relative Revenue", fontsize=13)
    ax.set_title("Selfish Mining: Revenue vs. Hash Rate\n(Eyal & Sirer, 2014)", fontsize=14)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_and_show(save_path)

def plot_threshold_vs_gamma(
    save_path: Optional[str] = None,
):
    """
    Plot the selfish mining profitability threshold as a function of gamma.
    Shows that with better network connectivity (higher gamma),
    the attacker needs less hash power.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    gamma_values = np.linspace(0, 1, 200)
    thresholds = [selfish_mining_threshold(g) for g in gamma_values]

    ax.plot(gamma_values, thresholds, "b-", linewidth=2.5)
    ax.fill_between(
        gamma_values, thresholds, 0.5,
        alpha=0.15, color="red", label="Selfish mining profitable"
    )
    ax.fill_between(
        gamma_values, 0, thresholds,
        alpha=0.15, color="green", label="Honest mining optimal"
    )

    # Mark key points
    ax.plot(0, 1/3, "ko", markersize=8)
    ax.annotate("γ=0 → α*=1/3", xy=(0.02, 1/3 + 0.01), fontsize=10)

    ax.plot(1, 0, "ko", markersize=8)
    ax.annotate("γ=1 → α*=0", xy=(0.85, 0.02), fontsize=10)

    ax.set_xlabel("γ (Network Propagation Advantage)", fontsize=13)
    ax.set_ylabel("Threshold α*", fontsize=13)
    ax.set_title("Selfish Mining Profitability Threshold", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()


def plot_revenue_heatmap(
    num_rounds: int = 50_000,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """
    Heatmap showing the selfish miner's relative revenue advantage
    across (alpha, gamma) parameter space.
    """
    alphas = np.linspace(0.01, 0.49, 30)
    gammas = np.linspace(0, 1, 30)
    advantage = np.zeros((len(gammas), len(alphas)))

    for i, gamma in enumerate(gammas):
        for j, alpha in enumerate(alphas):
            sim = SelfishMiningSimulator(alpha=alpha, gamma=gamma, seed=seed)
            result = sim.run(num_rounds=num_rounds)
            # Advantage = selfish revenue - honest revenue (alpha)
            advantage[i, j] = result.selfish_revenue - alpha

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        advantage,
        extent=[alphas[0], alphas[-1], gammas[0], gammas[-1]],
        origin="lower",
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=-0.05,
        vmax=0.15,
    )

    # Overlay the theoretical threshold curve
    gamma_line = np.linspace(0, 1, 200)
    threshold_line = [(1 - g) / (3 - 2 * g) for g in gamma_line]
    ax.plot(threshold_line, gamma_line, "k--", linewidth=2, label="Threshold α*")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Revenue Advantage (selfish - honest)", fontsize=11)

    ax.set_xlabel("α (Selfish Miner Hash Rate)", fontsize=13)
    ax.set_ylabel("γ (Network Propagation Advantage)", fontsize=13)
    ax.set_title("Selfish Mining Revenue Advantage Heatmap", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()

def plot_pool_comparison(
    pool_results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing miner revenues under different pool strategies (PPS vs PPLNS).

    Parameters
    ----------
    pool_results : dict
        Keys are strategy names ('PPS', 'PPLNS'), values are dicts
        mapping miner names to their result dicts.
    """
    fig, axes = plt.subplots(1, len(pool_results), figsize=(6 * len(pool_results), 5),
                              sharey=True)
    if len(pool_results) == 1:
        axes = [axes]

    for ax, (strategy_name, miners) in zip(axes, pool_results.items()):
        names = list(miners.keys())
        revenues = [miners[n]["revenue"] for n in names]
        hash_rates = [miners[n]["hash_rate"] for n in names]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, hash_rates, width, label="Hash Rate Share",
                       color="#3498db", alpha=0.85)
        bars2 = ax.bar(x + width/2, [r / max(sum(revenues), 1e-9) for r in revenues],
                       width, label="Revenue Share", color="#e74c3c", alpha=0.85)

        ax.set_xlabel("Miner", fontsize=11)
        ax.set_ylabel("Share", fontsize=11)
        ax.set_title(f"Pool Strategy: {strategy_name}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()

def plot_convergence(
    alpha: float = 0.3,
    gamma: float = 0.5,
    max_rounds: int = 500_000,
    checkpoints: int = 50,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    round_counts = np.linspace(1000, max_rounds, checkpoints, dtype=int)
    sim_vals = [
        SelfishMiningSimulator(alpha, gamma, seed=seed).run(int(n)).selfish_revenue
        for n in round_counts
    ]
    theo = theoretical_selfish_revenue(alpha, gamma)

    ax.plot(round_counts, sim_vals, "b-o", markersize=4, label="Simulated revenue")
    ax.axhline(y=theo, color="r", linestyle="--", linewidth=2,
               label=f"Theoretical = {theo:.4f}")
    ax.axhline(y=alpha, color="gray", linestyle=":", linewidth=1.5,
               label=f"Honest revenue = {alpha}")

    ax.set_xlabel("Number of Simulation Rounds", fontsize=13)
    ax.set_ylabel("Selfish Miner Relative Revenue", fontsize=13)
    ax.set_title(f"Convergence Test (α={alpha}, γ={gamma})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_and_show(save_path)
