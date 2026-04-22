"""Plots for the selfish mining experiments."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

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
