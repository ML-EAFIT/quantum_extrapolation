"""Active learning for choosing which circuits to run on real hardware (paper Section 5.3).

Greedy sampling on the input domain (GSx, Wu et al. 2019; paper Algorithm 1):

1. min-max normalize the global features and the graph features separately to [0, 1];
2. pick the sample closest to the centroid of the pool;
3. repeatedly pick the sample whose distance to its nearest already-selected sample is
   largest, where the distance between samples n and m is (Eq. 3)

       d(n, m) = alpha * ||g_n - g_m|| + (1 - alpha) * ||h_n - h_m||

   with g the global features and h the graph features (alpha = 0.5).

Each candidate is a (circuit, device) pair, because the compiled graph (and T1/T2)
differ per device while the global features do not. Graphs have a variable number of
nodes; their fixed-size representation here is the mean of the node feature vectors
(the authors' code zero-pads every node matrix to the largest graph instead, which is
equivalent in spirit but needs memory proportional to N x max_nodes x 178).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .dataset import Sample


def sample_size(population: int, confidence: float = 0.95, margin: float = 0.05, p: float = 0.5) -> int:
    """Cochran's sample size with finite-population correction (3020 -> 341 at 95 % / 5 %)."""
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - confidence) / 2)
    n0 = z * z * p * (1 - p) / (margin * margin)
    return int(math.ceil(n0 / (1 + (n0 - 1) / population)))


def minmax(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def greedy_sampling_x(G: np.ndarray, H: np.ndarray, k: int, alpha: float = 0.5) -> list[int]:
    """GSx selection of ``k`` row indices from the pool given global (G) and graph (H) features."""
    G, H = minmax(G), minmax(H)
    n = G.shape[0]
    if k >= n:
        return list(range(n))

    def dist_to(m: int) -> np.ndarray:
        return alpha * np.linalg.norm(G - G[m], axis=1) + (1 - alpha) * np.linalg.norm(H - H[m], axis=1)

    cg, ch = G.mean(axis=0), H.mean(axis=0)
    to_centroid = alpha * np.linalg.norm(G - cg, axis=1) + (1 - alpha) * np.linalg.norm(H - ch, axis=1)
    first = int(np.argmin(to_centroid))
    selected = [first]
    nearest = dist_to(first)  # distance of every sample to its closest selected sample
    nearest[first] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(nearest))
        selected.append(nxt)
        nearest = np.minimum(nearest, dist_to(nxt))
        nearest[selected] = -np.inf
    return selected


def pool_matrices(samples: Sequence[Sample], max_qubits: int = 127) -> tuple[np.ndarray, np.ndarray]:
    G = np.stack([s.global_raw for s in samples])
    G = G[:, np.abs(G).sum(axis=0) > 0]
    H = np.stack([s.graph.mean_vector(max_qubits) for s in samples])
    return G, H


def select(samples: Sequence[Sample], k: int | None = None, alpha: float = 0.5, max_qubits: int = 127) -> list[int]:
    """Indices of the samples to label (run on hardware). ``k`` defaults to :func:`sample_size`."""
    k = k or sample_size(len(samples))
    G, H = pool_matrices(samples, max_qubits)
    return greedy_sampling_x(G, H, k, alpha)
