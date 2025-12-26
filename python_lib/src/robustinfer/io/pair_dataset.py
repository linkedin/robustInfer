"""
Pair Batch Dataset for Streaming Mini-Batch Processing

This module provides PyTorch IterableDataset for streaming pair mini-batches
from row-level data using efficient k-partners sampling.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset

from .samplers import KPartnersSampler


class PairBatchIterableDataset(IterableDataset):
    """
    Streams mini-batches of pairs from row-level tensors.

    Improvement:
    - Uses PyTorch's native IterableDataset interface
    - Efficient streaming without loading all pairs into memory
    - Proper integration with DataLoader for multi-worker support
    - Automatic batching with configurable batch sizes
    - Clean separation between sampling and data loading

    Expected input tensors:
      X: [n, p] features (float32/float64)
      y: [n] labels/response
      z: [n] treatment/indicator (optional; pass None if unused)

    Yields dict with keys: xi, xj, yi, yj, zi, zj, w_ij
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None,
        sampler: KPartnersSampler,
        pairs_per_batch: int = 131072,
        device: torch.device | str = 'cpu',
        epoch: int = 0,
        shuffle_pairs: bool = True,
        infinite: bool = False,
        anchors_per_batch: int = 1,
    ):
        """
        Initialize pair batch dataset.

        Args:
            X: Feature matrix [n, p]
            y: Response vector [n]
            z: Treatment vector [n] (optional)
            sampler: K-partners sampler instance
            pairs_per_batch: Number of pairs per batch
            device: Device to move tensors to
            epoch: Current epoch (affects random sampling)
            shuffle_pairs: Whether to shuffle pairs within batches
            infinite: If True, repeat the same epoch indefinitely
        """
        super().__init__()

        # Validate inputs
        assert X.shape[0] == y.shape[0], 'X and y must have same number of observations'
        if z is not None:
            assert z.shape[0] == y.shape[0], 'z must have same length as y'

        self.X = X
        self.y = y
        self.z = z
        self.sampler = sampler
        self.m = int(pairs_per_batch)
        self.device = torch.device(device)
        self.epoch = int(epoch)
        self.shuffle_pairs = shuffle_pairs
        self.infinite = infinite
        self.anchors_per_batch = anchors_per_batch
        # Store original dtypes for proper tensor creation
        self.X_dtype = X.dtype
        self.y_dtype = y.dtype
        self.z_dtype = z.dtype if z is not None else None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Iterate over pair batches.
        """
        if self.infinite:
            while True:
                yield from self._iter_one_epoch()
        else:
            yield from self._iter_one_epoch()

    def _iter_one_epoch(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Iterate over pair batches.

        For anchor_based=True: Yields one batch per anchor containing all its partners
        For anchor_based=False: Yields batches of size pairs_per_batch

        Yields:
            Dictionary containing batched pair data with keys:
            - xi, xj: Feature matrices for pairs [batch_size, p]
            - yi, yj: Response vectors for pairs [batch_size]
            - zi, zj: Treatment vectors for pairs [batch_size] (if z provided)
            - w_ij: Horvitz-Thompson weights [batch_size]
            - anchor_id: Anchor identifier [batch_size] (only if anchor_based=True)
        """
        if getattr(self.sampler, 'anchor_based', False):
            # Anchor-based: yield batches of multiple anchors
            current_anchor_id = None
            xi, xj, yi, yj, zi, zj, wij, anchor_ids = [], [], [], [], [], [], [], []
            anchors_in_current_batch = 0

            for i, j, w, anchor_id in self.sampler.iter_pairs(epoch=self.epoch):
                # Check if we've moved to a new anchor
                if current_anchor_id is not None and anchor_id != current_anchor_id:
                    # We've finished collecting pairs for one anchor
                    anchors_in_current_batch += 1

                    # Check if we should yield the current batch
                    if anchors_in_current_batch >= self.anchors_per_batch:
                        yield self._flush_batch(xi, xj, yi, yj, zi, zj, wij, anchor_ids)
                        xi, xj, yi, yj, zi, zj, wij, anchor_ids = [], [], [], [], [], [], [], []
                        anchors_in_current_batch = 0

                # Accumulate pairs for current anchor
                current_anchor_id = anchor_id
                xi.append(self.X[i])
                xj.append(self.X[j])
                yi.append(self.y[i])
                yj.append(self.y[j])

                if self.z is not None:
                    zi.append(self.z[i])
                    zj.append(self.z[j])

                wij.append(w)
                anchor_ids.append(anchor_id)

            # Emit final anchor batch if any pairs remain
            if wij:
                yield self._flush_batch(xi, xj, yi, yj, zi, zj, wij, anchor_ids)
        else:
            # Original k-partners: accumulate pairs until batch size is reached
            xi, xj, yi, yj, zi, zj, wij = [], [], [], [], [], [], []

            for i, j, w in self.sampler.iter_pairs(epoch=self.epoch):
                # Extract features and responses for this pair
                xi.append(self.X[i])
                xj.append(self.X[j])
                yi.append(self.y[i])
                yj.append(self.y[j])

                # Extract treatment if provided
                if self.z is not None:
                    zi.append(self.z[i])
                    zj.append(self.z[j])

                # Store weight
                wij.append(w)

                # Emit batch when we have enough pairs
                if len(wij) >= self.m:
                    yield self._flush_batch(xi, xj, yi, yj, zi, zj, wij)
                    xi, xj, yi, yj, zi, zj, wij = [], [], [], [], [], [], []

            # Emit final partial batch if any pairs remain
            if wij:
                yield self._flush_batch(xi, xj, yi, yj, zi, zj, wij)

    def _flush_batch(self, xi, xj, yi, yj, zi, zj, wij, anchor_ids=None) -> dict[str, torch.Tensor]:
        """
        Convert accumulated pairs into a tensor batch.

        Args:
            xi, xj, yi, yj, zi, zj, wij: Lists of pair data
            anchor_ids: List of anchor IDs (for anchor-based sampling)

        Returns:
            Dictionary of batched tensors
        """
        dev = self.device

        # Create batch dictionary with required tensors
        batch = {
            'xi': torch.stack(xi).to(device=dev, dtype=self.X_dtype),
            'xj': torch.stack(xj).to(device=dev, dtype=self.X_dtype),
            'yi': torch.stack(yi).to(device=dev, dtype=self.y_dtype),
            'yj': torch.stack(yj).to(device=dev, dtype=self.y_dtype),
            'w_ij': torch.tensor(wij, dtype=self.X_dtype, device=dev),
        }

        # Add treatment tensors if available
        if self.z is not None and zi and zj:
            batch['zi'] = torch.stack(zi).to(device=dev, dtype=self.z_dtype)
            batch['zj'] = torch.stack(zj).to(device=dev, dtype=self.z_dtype)

        # Add anchor IDs if available (for anchor-based sampling)
        if anchor_ids is not None:
            batch['anchor_id'] = torch.tensor(anchor_ids, dtype=torch.long, device=dev)

        # Optionally shuffle pairs within batch for better randomization
        # Skip shuffling for anchor-based sampling to maintain anchor structure
        if self.shuffle_pairs and len(wij) > 1 and anchor_ids is None:
            perm = torch.randperm(len(wij), device=dev)
            for key in batch:
                batch[key] = batch[key][perm]

        return batch

    def set_epoch(self, epoch: int):
        """
        Set epoch for proper randomization across epochs.

        Args:
            epoch: New epoch number
        """
        self.epoch = epoch

    def estimate_batches_per_epoch(self) -> int:
        """
        Estimate number of batches per epoch.

        Returns:
            Estimated number of batches (useful for progress tracking)
        """
        if getattr(self.sampler, 'anchor_based', False):
            # For anchor-based sampling, number of batches with multiple anchors per batch
            import math

            return math.ceil(self.sampler.s / self.anchors_per_batch)
        else:
            # For regular sampling, estimate based on pairs per batch
            expected_pairs = self.sampler.expected_pairs_per_epoch()
            return max(1, int(expected_pairs / self.m))


# class CachedPairBatchDataset(PairBatchIterableDataset):
#     """Cached version for repeated epochs - caches pair indices in memory"""
