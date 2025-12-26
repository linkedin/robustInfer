"""
K-Partners Pair Sampling for U-Statistics

This module implements efficient pair sampling strategies for large-scale U-statistics
computation, particularly designed for DRGU and similar pairwise models.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np


class KPartnersSampler:
    """
    One-sided k-partners pair sampler.

    For each i in [0..n-1], sample k distinct partners j != i.
    Keep pair only if proposer(i,j) == i (default: i < j) to avoid double counting.
    HT weight: w_ij = 2*(n-1)/k (unbiased for uniform partners).

    Improvements:
    - Provides unbiased estimates with proper Horvitz-Thompson weights
    - Scales linearly O(n*k) instead of quadratically O(n^2)
    - Allows for stratified sampling extensions
    - Maintains proper statistical properties
    """

    def __init__(
        self,
        n: int,
        k: int,
        seed: int = 0,
        proposer: str = 'id',
        anchor_based: bool = False,
        s: int | None = None,
    ):
        """
        Initialize k-partners sampler (simplified version).

        Args:
            n: Number of observations
            k: Number of partners to sample for each observation
               (or m=k partners per anchor if anchor_based=True)
            seed: Random seed for reproducibility
            proposer: Strategy for avoiding double counting (simplified to "id" only)
                     "id": keep pair (i,j) only if i < j
            anchor_based: If True, use anchor-based sampling (s anchors, m=k partners each)
            s: Number of anchors (only used if anchor_based=True, defaults to n//10 if None)
        """
        assert n >= 2, 'Need n >= 2'
        assert k >= 1 and k < n, 'Need 1 <= k < n'

        self.n = int(n)
        self.k = int(k)  # m = k for anchor-based sampling
        self.seed = int(seed)

        # Anchor-based sampling configuration
        self.anchor_based = anchor_based
        if anchor_based:
            self.s = (
                s if s is not None else max(1, n // 10)
            )  # Default: 10% of observations as anchors
            assert 1 <= self.s <= n, f'Need 1 <= s <= n, got s={self.s}'
            assert self.k < n, f'Need k < n for partner sampling, got k={self.k}, n={n}'

        # Simplified: only support "id" proposer for now
        if proposer != 'id':
            raise ValueError(f"Only 'id' proposer supported currently, got '{proposer}'.")
        self.proposer = 'id'

        # Consider add hash-based proposer in future
        # self._h = ... hash computation ...

    def iter_pairs(
        self, epoch: int = 0, timeout: float | None = None
    ) -> Iterator[tuple[int, int, float]] | Iterator[tuple[int, int, float, int]]:
        """
        Generate pairs for one epoch with timeout protection.

        Args:
            epoch: Epoch number (affects random seed for different epochs)
            timeout: Maximum time in seconds before raising TimeoutError (None = no timeout)

        Yields:
            If anchor_based=False: Tuple of (i, j, weight) where:
            - i, j are observation indices
            - weight is Horvitz-Thompson weight for unbiased estimation

            If anchor_based=True: Tuple of (i, j, weight, anchor_id) where:
            - i, j are observation indices (i is anchor, j is partner)
            - weight is 1.0 (no reweighting needed for anchor-based)
            - anchor_id is the position of anchor i in the list (0 to s-1)

        Raises:
            TimeoutError: If sampling takes longer than timeout seconds
        """
        start_time = time.time() if timeout is not None else None

        # Different random seed per epoch for proper statistical sampling
        # Ensure seed stays within valid range for numpy (0 to 2^32 - 1)
        epoch_seed = (self.seed ^ (epoch * 0x9E3779B1)) & ((1 << 32) - 1)
        rng = np.random.RandomState(epoch_seed)

        if self.anchor_based:
            # Anchor-based sampling: sample s anchors, then k partners for each
            anchor_indices = rng.choice(self.n, size=self.s, replace=False)

            for anchor_id, anchor_idx in enumerate(anchor_indices):
                # Sample k partners for this anchor
                candidates = list(range(self.n))
                candidates.remove(anchor_idx)

                if self.k <= len(candidates):
                    partner_indices = rng.choice(candidates, size=self.k, replace=False)
                else:
                    partner_indices = candidates

                # Yield all anchor-partner pairs (no ordering constraint needed)
                for partner_idx in partner_indices:
                    yield (int(anchor_idx), int(partner_idx), 1.0, anchor_id)
        else:
            # Original k-partners sampling
            # Horvitz-Thompson weight for unbiased U-statistic estimation
            # Each pair has probability k/(n-1) of being selected, so weight = (n-1)/k
            # Factor of 2 accounts for the fact we only keep one direction of each pair
            w = 2.0 * (self.n - 1) / self.k

            for i in range(self.n):
                # Choose sampling strategy based on efficiency
                if self.k > (self.n - 1) * 0.5 or self.n <= 1000:
                    # Use direct sampling when k is large relative to n, or n is small
                    # This avoids inefficient rejection sampling in these cases
                    all_candidates = list(range(self.n))
                    all_candidates.remove(i)  # Remove self
                    partners = set(rng.choice(all_candidates, size=self.k, replace=False))
                else:
                    # Use rejection sampling for small k relative to n (memory efficient)
                    partners = set()
                    attempts = 0
                    while len(partners) < self.k:
                        # Check for timeout periodically to break infinite loops
                        # (only if timeout enabled)
                        if (
                            timeout is not None and attempts % 5000 == 0
                        ):  # Check every 5000 attempts (less frequent)
                            elapsed = time.time() - start_time
                            if elapsed > timeout:
                                raise TimeoutError(
                                    f'Sampling timeout after {elapsed:.2f}s (limit: {timeout}s). '
                                    f'Stuck at observation {i}, '
                                    f'found {len(partners)}/{self.k} partners. '
                                    f'This may indicate an infinite loop or numerical issue.'
                                )

                        j = int(rng.randint(0, self.n - 1))
                        if j == i:
                            attempts += 1
                            continue  # Can't partner with self
                        partners.add(j)
                        attempts += 1

                # Emit pairs, but avoid double counting
                for j in partners:
                    # Simplified: only use lexicographic ordering (i < j)
                    if i < j:
                        yield (i, j, w)

    def expected_pairs_per_epoch(self) -> float:
        """
        Expected number of pairs generated per epoch.

        Returns:
            Expected number of pairs (useful for batch size planning)
        """
        if self.anchor_based:
            # Each of s anchors generates k partners = s * k total pairs
            return self.s * self.k
        else:
            # Each observation generates k partners
            # With probability 0.5 (on average) each pair is kept due to ordering
            # So expected pairs ~= n * k * 0.5
            return self.n * self.k * 0.5


# Consider add stratified/block/reservoir samplers in future
