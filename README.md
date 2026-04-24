# Bitcoin Selfish Mining Simulator

Reproduces results from Eyal & Sirer (2014), "Majority is not Enough".

## Setup

```
pip install numpy matplotlib
```

## Run

```
python main.py                 # run all 5 experiments + demo
python main.py --experiment 1  # run one experiment (1..5)
python main.py --demo          # text-only sanity check, no plots
```

Figures are saved as `fig1_*.png` ... `fig5_*.png` in the current directory.

## Files

| File | Contents |
|---|---|
| `simulator.py` | `SelfishMiningSimulator` state machine, theoretical formulas |
| `strategies.py` | `MiningPool` with PPS / PPLNS reward schemes |
| `visualization.py` | Plotting functions for each figure |
| `main.py` | Experiments 1–5 and CLI entry point |

## Experiments

1. Selfish mining revenue vs. hash rate (fig1)
2. Profitability threshold α\* vs. γ (fig2)
3. Revenue advantage heatmap over (α, γ) (fig3)
4. Pool strategy comparison, PPS vs PPLNS (fig4)
5. Simulation convergence at α=0.3, γ=0.5 (fig5)
