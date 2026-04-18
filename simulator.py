"""Selfish mining simulator based on Eyal & Sirer (2014)."""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationResult:
    alpha: float
    gamma: float
    total_rounds: int
    selfish_blocks: int = 0
    honest_blocks: int = 0

    @property
    def selfish_revenue(self) -> float:
        total = self.selfish_blocks + self.honest_blocks
        return self.selfish_blocks / total if total else 0.0

    @property
    def honest_revenue(self) -> float:
        total = self.selfish_blocks + self.honest_blocks
        return self.honest_blocks / total if total else 0.0


@dataclass
class MiningState:
    private_lead: int = 0
    fork_active: bool = False   # True when we're in the 0' competing state


class SelfishMiningSimulator:
    def __init__(self, alpha: float, gamma: float, seed: Optional[int] = None):
        assert 0 < alpha < 0.5, "alpha must be in (0, 0.5)"
        assert 0 <= gamma <= 1, "gamma must be in [0, 1]"
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)

    def run(self, num_rounds: int = 100_000) -> SimulationResult:
        state = MiningState()
        result = SimulationResult(self.alpha, self.gamma, num_rounds)

        for _ in range(num_rounds):
            if self.rng.random() < self.alpha:
                self._on_selfish_block(state, result)
            else:
                self._on_honest_block(state, result)

        return result

    def _on_selfish_block(self, state, result):
        if state.fork_active:
            # selfish miner wins the fork race outright
            result.selfish_blocks += 2
            state.private_lead = 0
            state.fork_active = False
        else:
            state.private_lead += 1

    def _on_honest_block(self, state, result):
        if state.fork_active:
            # fork gets resolved by this block; gamma decides which side it joins
            if self.rng.random() < self.gamma:
                result.selfish_blocks += 1
                result.honest_blocks += 1
            else:
                result.honest_blocks += 2
            state.private_lead = 0
            state.fork_active = False
            return

        lead = state.private_lead
        if lead == 0:
            result.honest_blocks += 1
        elif lead == 1:
            # publish the secret block, wait for next round to settle
            state.fork_active = True
            state.private_lead = 0
        elif lead == 2:
            # override the honest block with the private chain
            result.selfish_blocks += 2
            state.private_lead = 0
        else:
            # lead >= 3: match honest and keep the remaining lead
            result.selfish_blocks += 1
            state.private_lead -= 1


def theoretical_selfish_revenue(alpha: float, gamma: float) -> float:
    """Closed-form relative revenue from Eyal & Sirer (2014)."""
    if alpha == 0:
        return 0.0
    a = alpha
    num = a * (1 - a) ** 2 * (4 * a + gamma * (1 - 2 * a)) - a ** 3
    den = 1 - a * (1 + a * (2 - a))
    return num / den if den else 0.0


def selfish_mining_threshold(gamma: float) -> float:
    return (1 - gamma) / (3 - 2 * gamma)
