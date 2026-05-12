"""
Topology-derived gamma for selfish mining.

The original Eyal-Sirer analysis treats gamma — the share of honest hash
power that adopts the selfish branch during a fork race — as a free
parameter. In a real deployment gamma is the output of (a) the gossip
topology, (b) how the selfish miner is peered into it, and (c) propagation
delays. This module measures gamma empirically by simulating fork races
over a small scale-free network, then exposes the result so it can be fed
back into the selfish-mining revenue formula.
"""

import math
import random
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class GossipNetwork:
    n_honest: int
    edges: List[Tuple[int, int]]
    hash_rate: List[float]
    neighbors: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.neighbors:
            self.neighbors = [[] for _ in range(self.n_honest)]
            for a, b in self.edges:
                self.neighbors[a].append(b)
                self.neighbors[b].append(a)


def build_scale_free(n: int, m: int = 3, seed: int = 42) -> GossipNetwork:
    """Barabasi-Albert preferential-attachment graph with Zipf hash rates."""
    rng = random.Random(seed)
    edges: List[Tuple[int, int]] = []
    degrees = [0] * n

    # Seed with a small fully-connected core
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edges.append((i, j))
            degrees[i] += 1
            degrees[j] += 1

    for new in range(m + 1, n):
        targets: set = set()
        while len(targets) < m:
            pool = [c for c in range(new) if c not in targets]
            weights = [degrees[c] + 1 for c in pool]
            chosen = rng.choices(pool, weights=weights, k=1)[0]
            targets.add(chosen)
        for t in targets:
            edges.append((new, t))
            degrees[new] += 1
            degrees[t] += 1

    raw = [1.0 / (i + 1) for i in range(n)]
    rng.shuffle(raw)
    total = sum(raw)
    hash_rate = [w / total for w in raw]

    return GossipNetwork(n_honest=n, edges=edges, hash_rate=hash_rate)


def propagation_times(source: int, net: GossipNetwork, per_edge_ms: float = 100.0) -> List[float]:
    """Uniform-weight Dijkstra (== BFS scaled by per_edge_ms)."""
    dist = [math.inf] * net.n_honest
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v in net.neighbors[u]:
            nd = d + per_edge_ms
            if nd < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))
    return dist


def gamma_from_topology(
    net: GossipNetwork,
    selfish_peers: List[int],
    per_edge_ms: float = 100.0,
    selfish_to_peer_ms: float = 50.0,
    trials: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Estimate gamma by averaging fork-race outcomes over many honest publishers."""
    if not selfish_peers:
        return 0.0, 0.0

    rng = random.Random(seed)

    selfish_times = [math.inf] * net.n_honest
    for peer in selfish_peers:
        peer_times = propagation_times(peer, net, per_edge_ms)
        for v in range(net.n_honest):
            t = selfish_to_peer_ms + peer_times[v]
            if t < selfish_times[v]:
                selfish_times[v] = t

    gammas: List[float] = []
    for _ in range(trials):
        publisher = rng.choices(range(net.n_honest), weights=net.hash_rate, k=1)[0]
        honest_times = propagation_times(publisher, net, per_edge_ms)

        adopted = 0.0
        for v in range(net.n_honest):
            st = selfish_times[v]
            ht = honest_times[v]
            if st < ht:
                adopted += net.hash_rate[v]
            elif st == ht:
                adopted += 0.5 * net.hash_rate[v]
        gammas.append(adopted)

    return float(np.mean(gammas)), float(np.std(gammas))


def select_peers(net: GossipNetwork, k: int, strategy: str, seed: int = 42) -> List[int]:
    """Three peer-selection policies for the selfish miner."""
    rng = random.Random(seed)
    n = net.n_honest
    k = min(k, n)
    if strategy == "random":
        return rng.sample(range(n), k)
    if strategy == "high_degree":
        return sorted(range(n), key=lambda i: -len(net.neighbors[i]))[:k]
    if strategy == "high_hashrate":
        return sorted(range(n), key=lambda i: -net.hash_rate[i])[:k]
    raise ValueError(f"unknown strategy: {strategy}")


def sweep_peer_counts(
    net: GossipNetwork,
    peer_counts: List[int],
    strategies: Tuple[str, ...] = ("random", "high_degree"),
    trials: int = 200,
    seed: int = 42,
) -> Dict[str, List[Dict[str, float]]]:
    """Sweep over peer counts for each strategy and return (gamma_mean, gamma_std)."""
    out: Dict[str, List[Dict[str, float]]] = {}
    for s in strategies:
        rows: List[Dict[str, float]] = []
        for k in peer_counts:
            peers = select_peers(net, k, s, seed)
            mean, std = gamma_from_topology(net, peers, trials=trials, seed=seed)
            rows.append({"k": float(k), "gamma_mean": mean, "gamma_std": std})
        out[s] = rows
    return out
