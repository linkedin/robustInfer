"""
Tests for the io module: samplers and pair datasets
"""

from collections import defaultdict

import pytest
import torch

from robustinfer.io.pair_dataset import PairBatchIterableDataset
from robustinfer.io.samplers import KPartnersSampler


class TestKPartnersSampler:
    """Test the KPartnersSampler for both regular and anchor-based modes."""

    def test_regular_sampling_basic(self):
        """Test basic k-partners sampling functionality."""
        n, k = 10, 3
        sampler = KPartnersSampler(n=n, k=k, seed=42)

        assert sampler.n == n
        assert sampler.k == k
        assert not sampler.anchor_based
        assert sampler.expected_pairs_per_epoch() == n * k * 0.5

    def test_regular_sampling_pairs(self):
        """Test that regular sampling generates valid pairs."""
        n, k = 20, 5
        sampler = KPartnersSampler(n=n, k=k, seed=42)

        pairs = list(sampler.iter_pairs(epoch=0))

        # Check basic properties
        assert len(pairs) > 0
        for i, j, w in pairs:
            assert 0 <= i < n and 0 <= j < n
            assert i != j  # No self-pairs
            assert i < j  # Lexicographic ordering
            assert w > 0  # Positive weight

        # Check that weights are consistent
        weights = [w for _, _, w in pairs]
        expected_weight = 2.0 * (n - 1) / k
        assert all(abs(w - expected_weight) < 1e-10 for w in weights)

    def test_anchor_based_initialization(self):
        """Test anchor-based sampler initialization."""
        n, k, s = 100, 10, 20
        sampler = KPartnersSampler(n=n, k=k, anchor_based=True, s=s, seed=42)

        assert sampler.n == n
        assert sampler.k == k  # m = k
        assert sampler.s == s
        assert sampler.anchor_based
        assert sampler.expected_pairs_per_epoch() == s * k

    def test_anchor_based_default_s(self):
        """Test anchor-based sampler with default s."""
        n, k = 100, 10
        sampler = KPartnersSampler(n=n, k=k, anchor_based=True, seed=42)

        expected_s = max(1, n // 10)
        assert sampler.s == expected_s

    def test_anchor_based_sampling_structure(self):
        """Test that anchor-based sampling maintains anchor structure."""
        n, k, s = 50, 5, 10
        sampler = KPartnersSampler(n=n, k=k, anchor_based=True, s=s, seed=42)

        pairs = list(sampler.iter_pairs(epoch=0))

        # Group pairs by anchor
        anchor_groups = defaultdict(list)
        for i, j, w, anchor_id in pairs:
            assert 0 <= i < n and 0 <= j < n
            assert i != j  # No self-pairs
            assert w == 1.0  # Anchor-based uses weight 1.0
            assert 0 <= anchor_id < s
            anchor_groups[anchor_id].append((i, j))

        # Check that we have exactly s anchors
        assert len(anchor_groups) == s

        # Check that each anchor has exactly k partners (or less if not enough candidates)
        for _anchor_id, anchor_pairs in anchor_groups.items():
            anchor_indices = [i for i, j in anchor_pairs]
            partner_indices = [j for i, j in anchor_pairs]

            # All pairs for this anchor should have the same anchor index
            assert len(set(anchor_indices)) == 1
            anchor_idx = anchor_indices[0]

            # All partner indices should be different and not equal to anchor
            assert len(set(partner_indices)) == len(partner_indices)  # All unique
            assert anchor_idx not in partner_indices

            # Should have k partners (or n-1 if k >= n-1)
            expected_partners = min(k, n - 1)
            assert len(anchor_pairs) == expected_partners

    def test_anchor_based_reproducibility(self):
        """Test that anchor-based sampling is reproducible with same seed."""
        n, k, s = 30, 4, 6

        # Generate pairs twice with same seed
        sampler1 = KPartnersSampler(n=n, k=k, anchor_based=True, s=s, seed=123)
        pairs1 = list(sampler1.iter_pairs(epoch=0))

        sampler2 = KPartnersSampler(n=n, k=k, anchor_based=True, s=s, seed=123)
        pairs2 = list(sampler2.iter_pairs(epoch=0))

        assert pairs1 == pairs2

    def test_anchor_based_different_epochs(self):
        """Test that different epochs produce different samples."""
        n, k, s = 30, 4, 6
        sampler = KPartnersSampler(n=n, k=k, anchor_based=True, s=s, seed=123)

        pairs_epoch0 = list(sampler.iter_pairs(epoch=0))
        pairs_epoch1 = list(sampler.iter_pairs(epoch=1))

        # Should be different (very unlikely to be identical by chance)
        assert pairs_epoch0 != pairs_epoch1

    def test_validation_errors(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test n too small
        with pytest.raises(AssertionError):
            KPartnersSampler(n=1, k=1)

        # Test k too large
        with pytest.raises(AssertionError):
            KPartnersSampler(n=10, k=10)

        # Test invalid s for anchor-based
        with pytest.raises(AssertionError):
            KPartnersSampler(n=10, k=3, anchor_based=True, s=15)


class TestPairBatchIterableDataset:
    """Test the PairBatchIterableDataset for both regular and anchor-based modes."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n, p = 20, 3
        X = torch.randn(n, p, dtype=torch.float32)
        y = torch.randn(n, dtype=torch.float32)
        z = torch.randint(0, 2, (n,), dtype=torch.long)
        return X, y, z

    def test_regular_dataset_basic(self, sample_data):
        """Test basic functionality of regular pair dataset."""
        X, y, z = sample_data
        n = len(y)

        sampler = KPartnersSampler(n=n, k=5, seed=42)
        dataset = PairBatchIterableDataset(X=X, y=y, z=z, sampler=sampler, pairs_per_batch=10)

        batches = list(dataset)
        assert len(batches) > 0

        # Check first batch structure
        batch = batches[0]
        required_keys = {'xi', 'xj', 'yi', 'yj', 'zi', 'zj', 'w_ij'}
        assert required_keys.issubset(batch.keys())
        assert 'anchor_id' not in batch  # Should not have anchor_id for regular sampling

        # Check batch shapes
        batch_size = len(batch['yi'])
        assert batch['xi'].shape == (batch_size, X.shape[1])
        assert batch['xj'].shape == (batch_size, X.shape[1])
        assert batch['yi'].shape == (batch_size,)
        assert batch['yj'].shape == (batch_size,)
        assert batch['zi'].shape == (batch_size,)
        assert batch['zj'].shape == (batch_size,)
        assert batch['w_ij'].shape == (batch_size,)

    def test_anchor_based_dataset_structure(self, sample_data):
        """Test that anchor-based dataset maintains anchor structure."""
        X, y, z = sample_data
        n = len(y)
        k, s = 4, 5

        sampler = KPartnersSampler(n=n, k=k, anchor_based=True, s=s, seed=42)
        dataset = PairBatchIterableDataset(
            X=X,
            y=y,
            z=z,
            sampler=sampler,
            pairs_per_batch=100,  # Large batch size
        )

        batches = list(dataset)

        # Should have exactly s batches (one per anchor)
        assert len(batches) == s

        for _batch_idx, batch in enumerate(batches):
            # Check required keys including anchor_id
            required_keys = {'xi', 'xj', 'yi', 'yj', 'zi', 'zj', 'w_ij', 'anchor_id'}
            assert required_keys.issubset(batch.keys())

            # All anchor_ids in this batch should be the same
            anchor_ids = batch['anchor_id']
            assert torch.all(anchor_ids == anchor_ids[0])

            # Batch size should be k (or n-1 if k >= n-1)
            expected_batch_size = min(k, n - 1)
            assert len(batch['yi']) == expected_batch_size

            # All xi should be the same (same anchor repeated)
            xi_values = batch['xi']
            assert torch.allclose(xi_values, xi_values[0:1].expand_as(xi_values))

            # All yi should be the same (same anchor repeated)
            yi_values = batch['yi']
            assert torch.allclose(yi_values, yi_values[0:1].expand_as(yi_values))

            # All zi should be the same (same anchor repeated)
            zi_values = batch['zi']
            assert torch.all(zi_values == zi_values[0])

            # All weights should be 1.0 for anchor-based
            assert torch.allclose(batch['w_ij'], torch.ones_like(batch['w_ij']))

            # xj, yj, zj should all be different (different partners)
            xj_values = batch['xj']
            batch['yj']
            batch['zj']

            if expected_batch_size > 1:
                # Partners should be different from each other
                for i in range(expected_batch_size):
                    for j in range(i + 1, expected_batch_size):
                        assert not torch.allclose(xj_values[i], xj_values[j])

    def test_anchor_based_no_shuffling(self, sample_data):
        """Test that anchor-based datasets don't shuffle pairs within batches."""
        X, y, z = sample_data
        n = len(y)

        sampler = KPartnersSampler(n=n, k=3, anchor_based=True, s=3, seed=42)
        dataset = PairBatchIterableDataset(
            X=X,
            y=y,
            z=z,
            sampler=sampler,
            shuffle_pairs=True,  # Try to enable shuffling
        )

        batches1 = list(dataset)

        # Create another dataset with same seed
        dataset2 = PairBatchIterableDataset(X=X, y=y, z=z, sampler=sampler, shuffle_pairs=True)
        dataset2.set_epoch(0)  # Same epoch
        batches2 = list(dataset2)

        # Batches should be identical (no shuffling occurred)
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2, strict=False):
            for key in b1:
                assert torch.allclose(b1[key], b2[key])

    def test_without_treatment_variable(self, sample_data):
        """Test dataset without treatment variable."""
        X, y, _ = sample_data
        n = len(y)

        sampler = KPartnersSampler(n=n, k=3, anchor_based=True, s=3, seed=42)
        dataset = PairBatchIterableDataset(X=X, y=y, z=None, sampler=sampler)

        batches = list(dataset)
        assert len(batches) > 0

        batch = batches[0]
        assert 'zi' not in batch
        assert 'zj' not in batch
        assert 'xi' in batch and 'xj' in batch
        assert 'yi' in batch and 'yj' in batch

    def test_set_epoch(self, sample_data):
        """Test epoch setting functionality."""
        X, y, z = sample_data
        n = len(y)

        sampler = KPartnersSampler(n=n, k=3, anchor_based=True, s=3, seed=42)
        dataset = PairBatchIterableDataset(X=X, y=y, z=z, sampler=sampler)

        batches_epoch0 = list(dataset)

        dataset.set_epoch(1)
        batches_epoch1 = list(dataset)

        # Should be different - compare partner indices instead of anchor_ids
        # (anchor_ids might be the same due to small s, but partners should differ)
        assert len(batches_epoch0) == len(batches_epoch1)

        # Check that at least one batch has different partners
        different = False
        for b0, b1 in zip(batches_epoch0, batches_epoch1, strict=False):
            # Compare the xj (partner features) since partners should be different
            if not torch.allclose(b0['xj'], b1['xj']):
                different = True
                break
        assert different, 'Partner selections should be different across epochs'

    def test_estimate_batches_per_epoch(self, sample_data):
        """Test batch count estimation."""
        X, y, z = sample_data
        n = len(y)

        # Regular sampling
        sampler_regular = KPartnersSampler(n=n, k=5, seed=42)
        dataset_regular = PairBatchIterableDataset(
            X=X, y=y, z=z, sampler=sampler_regular, pairs_per_batch=10
        )

        estimated = dataset_regular.estimate_batches_per_epoch()
        actual = len(list(dataset_regular))
        # Should be close (estimation may not be exact due to randomness)
        assert abs(estimated - actual) <= 2

        # Anchor-based sampling
        s = 7
        sampler_anchor = KPartnersSampler(n=n, k=3, anchor_based=True, s=s, seed=42)
        dataset_anchor = PairBatchIterableDataset(X=X, y=y, z=z, sampler=sampler_anchor)

        estimated = dataset_anchor.estimate_batches_per_epoch()
        actual = len(list(dataset_anchor))
        assert estimated == actual == s  # Should be exact for anchor-based


if __name__ == '__main__':
    pytest.main([__file__])
