"""
Mining pool reward strategies: PPLNS and PPS.
Also includes multi-miner simulation with configurable hash rate distribution.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Miner:
    """Represents a single miner or mining pool."""
    name: str
    hash_rate: float        # fraction of total network hash rate
    is_selfish: bool = False
    blocks_won: int = 0
    revenue: float = 0.0


@dataclass
class ShareRecord:
    """A record of a share submitted by a miner within a pool."""
    miner_name: str
    round_number: int


class MiningPool:
    """
    Simulates a mining pool with configurable reward distribution strategy.

    Supported strategies:
    - PPS (Pay Per Share): miners get a fixed payout per share,
      regardless of whether the pool finds a block.
    - PPLNS (Pay Per Last N Shares): reward is split among the last N shares
      when a block is found.
    """

    def __init__(self, name: str, strategy: str = "pps", pplns_window: int = 100):
        if strategy not in ("pps", "pplns"):
            raise ValueError(f"Unknown strategy: {strategy}. Use 'pps' or 'pplns'.")

        self.name = name
        self.strategy = strategy
        self.pplns_window = pplns_window

        self.members: List[Miner] = []
        self.shares: List[ShareRecord] = []
        self.blocks_found: int = 0
        self.total_pool_hash_rate: float = 0.0

    def add_member(self, miner: Miner):
        self.members.append(miner)
        self.total_pool_hash_rate += miner.hash_rate

    def submit_share(self, miner: Miner, round_number: int):
        """Record a share submission from a miner."""
        self.shares.append(ShareRecord(miner_name=miner.name, round_number=round_number))

    def block_found(self, block_reward: float = 1.0):
        """Distribute reward when the pool finds a block."""
        self.blocks_found += 1

        if self.strategy == "pps":
            self._distribute_pps(block_reward)
        else:
            self._distribute_pplns(block_reward)

    def _distribute_pps(self, block_reward: float):
        """
        PPS: each miner gets paid proportional to their hash rate,
        regardless of luck. The pool operator absorbs variance.
        """
        for miner in self.members:
            share = miner.hash_rate / self.total_pool_hash_rate
            miner.revenue += block_reward * share

    def _distribute_pplns(self, block_reward: float):
        """
        PPLNS: reward is distributed based on the last N shares submitted.
        This discourages pool hopping.
        """
        recent_shares = self.shares[-self.pplns_window:]
        if not recent_shares:
            return

        share_counts: dict[str, int] = {}
        for record in recent_shares:
            share_counts[record.miner_name] = share_counts.get(record.miner_name, 0) + 1

        total_shares = len(recent_shares)
        for miner in self.members:
            count = share_counts.get(miner.name, 0)
            miner.revenue += block_reward * (count / total_shares)


def simulate_pool_mining(
    pool: MiningPool,
    num_rounds: int = 10_000,
    block_reward: float = 6.25,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate mining within a pool over multiple rounds.

    Each round, miners submit shares proportional to their hash rate,
    and blocks are found probabilistically based on pool hash rate.

    Returns a summary dict with each miner's revenue.
    """
    rng = random.Random(seed)

    for round_num in range(num_rounds):
        # Each miner submits shares proportional to hash rate
        for miner in pool.members:
            if rng.random() < miner.hash_rate:
                pool.submit_share(miner, round_num)

        # Pool finds a block with probability proportional to total hash rate
        if rng.random() < pool.total_pool_hash_rate:
            pool.block_found(block_reward)

    return {
        miner.name: {
            "hash_rate": miner.hash_rate,
            "revenue": miner.revenue,
            "revenue_per_hash": miner.revenue / miner.hash_rate if miner.hash_rate > 0 else 0,
        }
        for miner in pool.members
    }


def create_hash_rate_distribution(
    num_miners: int,
    distribution: str = "uniform",
    selfish_fraction: float = 0.0,
    seed: Optional[int] = None,
) -> List[Miner]:
    """
    Create a list of miners with a given hash rate distribution.

    Parameters
    ----------
    num_miners : int
        Number of miners.
    distribution : str
        'uniform' - equal hash rates
        'zipf' - power-law distribution (realistic)
    selfish_fraction : float
        Total hash rate fraction controlled by the selfish miner (miner 0).
    seed : int or None
        Random seed.
    """
    rng = random.Random(seed)
    miners = []

    if distribution == "uniform":
        remaining = 1.0 - selfish_fraction
        honest_rate = remaining / max(num_miners - 1, 1)

        miners.append(Miner(name="selfish_0", hash_rate=selfish_fraction, is_selfish=True))
        for i in range(1, num_miners):
            miners.append(Miner(name=f"honest_{i}", hash_rate=honest_rate))

    elif distribution == "zipf":
        # Generate Zipf-like weights for honest miners
        weights = [1.0 / (i + 1) for i in range(num_miners - 1)]
        total_weight = sum(weights)
        remaining = 1.0 - selfish_fraction

        miners.append(Miner(name="selfish_0", hash_rate=selfish_fraction, is_selfish=True))
        for i, w in enumerate(weights):
            rate = remaining * (w / total_weight)
            miners.append(Miner(name=f"honest_{i+1}", hash_rate=rate))

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return miners
