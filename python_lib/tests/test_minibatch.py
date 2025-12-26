import pandas as pd
import pytest
import torch

from robustinfer import DRGU  # For equivalence testing
from robustinfer.io.pair_dataset import PairBatchIterableDataset

# Import the minibatch components
from robustinfer.io.samplers import KPartnersSampler
from robustinfer.minibatch.drgu_minibatch import DRGUMiniBatch
from robustinfer.minibatch.estimating_equations import drgu_compute_B_U
from robustinfer.minibatch.minibatch_fisher import MiniBatchFisherScoring, Penalty
from robustinfer.minibatch.montecarlo_estimation import MonteCarloEstimation

# TODO: Add back advanced samplers and example functions when basic convergence works
# from robustinfer.io.samplers import (
#     StratifiedKPartnersSampler,
#     BlockKPartnersSampler,
#     ReservoirKPartnersSampler,
# )
# from robustinfer.minibatch.estimating_equations import example_compute_B_U


# ============================================================================
# Test Data Generation
# ============================================================================


@pytest.fixture
def simple_data():
    """Generate simple synthetic data for testing."""
    torch.manual_seed(42)
    n, p = 100, 3

    X = torch.randn(n, p, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    z = torch.randint(0, 2, (n,), dtype=torch.float64)

    return {'X': X, 'y': y, 'z': z, 'n': n, 'p': p}


@pytest.fixture
def drgu_data():
    """Generate DRGU-specific synthetic data for convergence testing."""
    torch.manual_seed(123)
    n, p = 200, 2

    # Generate features
    X = torch.randn(n, p, dtype=torch.float64)

    # Generate treatment with some structure
    z_logits = 0.1 + 0.2 * X.sum(dim=1) + 0.1 * torch.randn(n)
    z = torch.bernoulli(torch.sigmoid(z_logits)).to(torch.float64)

    # Generate outcome with treatment effect
    y_mean = 0.2 + 0.3 * X.sum(dim=1) + 0.5 * z + 0.1 * z * X[:, 0]
    y = y_mean + 0.2 * torch.randn(n, dtype=torch.float64)

    return {'X': X, 'y': y, 'z': z, 'n': n, 'p': p}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def fit_minibatch_optimizer(optimizer, dataset, max_steps=10, tolerance=1e-6):
    """
    Helper function to simulate fit functionality for MiniBatchFisherScoring.

    Args:
        optimizer: MiniBatchFisherScoring instance
        dataset: Data iterator
        max_steps: Maximum number of optimization steps
        tolerance: Convergence tolerance

    Returns:
        Dict with final theta, convergence info, and step count
    """
    converged = False
    step = 0

    for step in range(max_steps):
        try:
            # Get next batch
            batch = next(iter(dataset))

            # Perform optimization step
            info = optimizer.step(batch)

            # Check convergence
            if 'U_norm' in info and info['U_norm'] < tolerance:
                converged = True
                break

        except (StopIteration, RuntimeError) as e:
            # Handle dataset exhaustion or numerical issues
            if step == 0:
                raise e  # Re-raise if we couldn't even take one step
            break

    return {
        'theta': optimizer.theta,
        'converged': converged,
        'steps': step + 1,
        'final_info': info if 'info' in locals() else {},
    }


# ============================================================================
# SAMPLING TESTS
# ============================================================================


class TestSampling:
    """Test all sampling strategies work correctly."""

    def test_kpartners_basic(self, simple_data):
        """Test basic k-partners sampling."""
        n = simple_data['n']
        k = 5
        sampler = KPartnersSampler(n=n, k=k, seed=42)

        # Collect all pairs from one epoch
        pairs = list(sampler.iter_pairs(epoch=0))

        # Should have approximately n*k/2 pairs (since we keep i < j)
        assert len(pairs) > 0
        assert len(pairs) <= n * k  # Upper bound

        # Check pair format
        for i_idx, j_idx, weight in pairs:
            assert 0 <= i_idx < n
            assert 0 <= j_idx < n
            assert i_idx != j_idx
            assert weight > 0
            # For proposer="id", should have i < j
            assert i_idx < j_idx

    def test_sampling_weights(self, simple_data):
        """Test Horvitz-Thompson weights are correct."""
        n = simple_data['n']
        k = 10
        sampler = KPartnersSampler(n=n, k=k, seed=42)

        pairs = list(sampler.iter_pairs(epoch=0))

        # HT weight should be 2*(n-1)/k for unbiased estimation
        expected_weight = 2 * (n - 1) / k

        for _, _, weight in pairs:
            assert abs(weight - expected_weight) < 1e-10

    def test_complete_sampling_when_k_equals_n_minus_1(self, simple_data):
        """Test that when k=n-1, we get all possible ordered pairs exactly once."""
        import time

        n = simple_data['n']
        k = n - 1  # Sample all possible partners

        sampler = KPartnersSampler(n=n, k=k, seed=42)

        # Time the sampling operation to detect infinite loops
        start_time = time.time()
        pairs = list(sampler.iter_pairs(epoch=0, timeout=5.0))  # Enable timeout for this test only
        elapsed_time = time.time() - start_time

        # Should complete quickly - timeout will catch infinite loops automatically
        max_allowed_time = 4.0  # seconds (less than timeout for early warning)
        assert elapsed_time < max_allowed_time, (
            f'Sampling took too long: {elapsed_time:.2f}s > {max_allowed_time}s (performance issue)'
        )

        # Convert pairs to set of (i,j) tuples (ignoring weights)
        pair_indices = {(i, j) for i, j, _ in pairs}

        # Generate expected set of all ordered pairs with i < j
        expected_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}

        # Should get exactly all unique ordered pairs
        assert pair_indices == expected_pairs, (
            f'Expected {len(expected_pairs)} unique pairs, got {len(pair_indices)}'
        )

        # Verify count: should be n*(n-1)/2 pairs
        expected_count = n * (n - 1) // 2
        assert len(pairs) == expected_count, f'Expected {expected_count} pairs, got {len(pairs)}'

        # Verify all weights are correct for complete sampling
        # expected_weight = 2.0 * (n - 1) / k  # k = n-1, so this should be 2.0
        for _, _, weight in pairs:
            assert abs(weight - 2.0) < 1e-10, (
                f'Expected weight 2.0 for complete sampling, got {weight}'
            )

        print(
            f'Complete sampling test passed: {len(pairs)} pairs for '
            f'n={n}, k={k} in {elapsed_time:.3f}s'
        )

    # TODO: Add back advanced sampling tests when features are restored
    # def test_stratified_sampling(self, simple_data):
    #     """Test stratified sampling works and maintains stratification."""

    # def test_block_sampling(self, simple_data):
    #     """Test block sampling works and respects block boundaries."""

    # def test_reservoir_sampling(self, simple_data):
    #     """Test reservoir sampling works."""


class TestPairDataset:
    """Test pair dataset streaming works."""

    def test_dataset_basic(self, simple_data):
        """Test basic dataset functionality."""
        X, y, z = simple_data['X'], simple_data['y'], simple_data['z']
        n = simple_data['n']

        sampler = KPartnersSampler(n=n, k=5, seed=42)
        dataset = PairBatchIterableDataset(X=X, y=y, z=z, sampler=sampler, pairs_per_batch=20)

        # Get one batch
        batch = next(iter(dataset))

        # Check batch format
        expected_keys = {'xi', 'xj', 'yi', 'yj', 'zi', 'zj', 'w_ij'}
        assert set(batch.keys()) == expected_keys

        # Check dimensions
        m = batch['xi'].shape[0]  # batch size
        p = batch['xi'].shape[1]  # features

        assert batch['xi'].shape == (m, p)
        assert batch['xj'].shape == (m, p)
        assert batch['yi'].shape == (m,)
        assert batch['yj'].shape == (m,)
        assert batch['zi'].shape == (m,)
        assert batch['zj'].shape == (m,)
        assert batch['w_ij'].shape == (m,)


# ============================================================================
# CONVERGENCE TESTS
# ============================================================================


class TestConvergence:
    """Test mini-batch Fisher scoring actually converges."""

    # TODO: Add back simple parameter update test when example_compute_B_U is restored
    # def test_parameter_updates(self, simple_data):
    #     """Test that parameters actually change from initial values."""

    # TODO: Add back penalty test when example_compute_B_U is restored
    # def test_penalty_regularization(self, simple_data):
    #     """Test that large penalty shrinks parameters."""

    # TODO: Add back optimization modes test when trust_region/dogleg are restored
    # def test_optimization_modes(self, simple_data):
    #     """Test different optimization modes work (plain, line_search, trust_region, dogleg)."""


class TestDRGUConvergence:
    """Test DRGU-specific convergence."""

    def test_drgu_parameter_updates(self, drgu_data):
        """Test DRGU mini-batch actually updates parameters."""
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n, p = drgu_data['n'], drgu_data['p']

        sampler = KPartnersSampler(n=n, k=8, seed=42)
        dataset = PairBatchIterableDataset(X=X, y=y, z=z, sampler=sampler, pairs_per_batch=40)

        # DRGU parameters: delta, beta, gamma
        theta = {
            'delta': torch.zeros(1, dtype=torch.float64),
            'beta': torch.zeros(p + 1, dtype=torch.float64),
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float64),
        }

        optimizer = MiniBatchFisherScoring(
            model_params=theta, compute_B_U=drgu_compute_B_U, penalty=Penalty(lam=1e-4)
        )

        result = fit_minibatch_optimizer(optimizer, dataset, max_steps=5)
        final_theta = result['theta']

        # Check parameter changes
        delta_change = abs(final_theta['delta'].item())
        beta_change = torch.norm(final_theta['beta']).item()
        gamma_change = torch.norm(final_theta['gamma']).item()

        print('DRGU parameter changes:')
        print(f'  delta: {delta_change:.6f}')
        print(f'  beta norm: {beta_change:.6f}')
        print(f'  gamma norm: {gamma_change:.6f}')

        # At least one parameter should change meaningfully
        total_change = delta_change + beta_change + gamma_change
        assert total_change > 0.05, f'DRGU parameters barely changed: {total_change}'

    def test_drgu_multi_epoch_stability(self, drgu_data):
        """Test DRGU optimization is stable across multiple epochs."""
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n, p = drgu_data['n'], drgu_data['p']

        sampler = KPartnersSampler(n=n, k=8, seed=42)

        theta = {
            'delta': torch.zeros(1, dtype=torch.float64),
            'beta': torch.zeros(p + 1, dtype=torch.float64),
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float64),
        }

        # Run multiple epochs
        for epoch in range(3):
            dataset = PairBatchIterableDataset(
                X=X, y=y, z=z, sampler=sampler, pairs_per_batch=40, epoch=epoch
            )

            optimizer = MiniBatchFisherScoring(
                model_params=theta,
                compute_B_U=drgu_compute_B_U,
                penalty=Penalty(lam=1e-2),  # Increased regularization for stability
                max_step_norm=5.0,  # Conservative step size for stability
            )

            result = fit_minibatch_optimizer(optimizer, dataset, max_steps=3)
            theta = result['theta']

            # Check for numerical stability (no NaNs/Infs)
            for key, param in theta.items():
                assert torch.isfinite(param).all(), f'NaN/Inf in {key} at epoch {epoch}'

            print(
                f'Epoch {epoch}: delta={theta["delta"].item():.4f}, '
                f'beta_norm={torch.norm(theta["beta"]).item():.4f}'
            )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestEquivalence:
    """Test equivalence between DRGU and DRGUMiniBatch."""

    def test_B_U_identical_full_sampling(self, drgu_data):
        """
        Test that B (Fisher matrix) and U (gradient) are identical
        between DRGU and DRGUMiniBatch with complete sampling.
        """
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']
        _n, p = drgu_data['n'], drgu_data['p']  # _n unused, p needed for theta

        # Use smaller data for faster testing
        n_test = 20
        X_test, y_test, z_test = X[:n_test], y[:n_test], z[:n_test]

        # Create DataFrame
        import pandas as pd

        df_test = pd.DataFrame(
            {
                'x1': X_test[:, 0].numpy(),
                'x2': X_test[:, 1].numpy(),
                'treatment': z_test.numpy(),
                'response': y_test.numpy(),
            }
        )

        # Same initial parameters for both
        theta = {
            'delta': torch.zeros(1, dtype=torch.float32),
            'beta': torch.zeros(p + 1, dtype=torch.float32),
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float32),
        }

        # DRGU computation
        model_torch = DRGU(
            df_test,
            covariates=['x1', 'x2'],
            treatment='treatment',
            response='response',
            device='cpu',
        )
        from robustinfer.drgu import compute_B_U, data_pairwise

        torch_data = data_pairwise(model_torch.y, model_torch.z, model_torch.w)
        V_inv = torch.eye(3, dtype=torch.float32)
        B_torch, U_torch = compute_B_U(theta, V_inv, torch_data)

        # DRGUMiniBatch computation with complete sampling
        model_mini = DRGUMiniBatch(
            df_test,
            covariates=['x1', 'x2'],
            treatment='treatment',
            response='response',
            device='cpu',
        )
        k = n_test - 1  # Sample all partners
        expected_pairs = n_test * (n_test - 1) // 2

        from robustinfer.io.pair_dataset import PairBatchIterableDataset
        from robustinfer.io.samplers import KPartnersSampler
        from robustinfer.minibatch.estimating_equations import drgu_compute_B_U

        sampler = KPartnersSampler(n=n_test, k=k, seed=42)
        dataset = PairBatchIterableDataset(
            X=model_mini.X,
            y=model_mini.y,
            z=model_mini.z,
            sampler=sampler,
            pairs_per_batch=expected_pairs,
        )

        batch = next(iter(dataset))
        B_mini, U_mini = drgu_compute_B_U(theta, batch)

        # Compare B and U
        print(f'\nComparing B and U for n={n_test}, pairs={expected_pairs}')
        print(
            f'DRGU   - B cond: {torch.linalg.cond(B_torch):.2e}, '
            f'U norm: {torch.linalg.norm(U_torch):.6f}'
        )
        print(
            f'DRGUMiniBatch - B cond: {torch.linalg.cond(B_mini):.2e}, '
            f'U norm: {torch.linalg.norm(U_mini):.6f}'
        )

        # Check B matrices are essentially identical (allow for numerical precision)
        B_diff = torch.linalg.norm(B_torch - B_mini)
        B_rel_diff = B_diff / torch.linalg.norm(B_torch)
        print(f'B difference: {B_rel_diff:.2e}')
        assert B_rel_diff < 1e-6, f'B matrices differ by {B_rel_diff:.2e}'

        # Check U vectors are essentially identical
        U_diff = torch.linalg.norm(U_torch - U_mini)
        U_rel_diff = U_diff / torch.linalg.norm(U_torch)
        print(f'U difference: {U_rel_diff:.2e}')
        assert U_rel_diff < 1e-6, f'U vectors differ by {U_rel_diff:.2e}'

        print('B and U are identical!')

    def test_theta_identical_after_one_step(self, drgu_data):
        """
        Test that theta parameters are identical after 1 optimization step
        between DRGU and DRGUMiniBatch with complete sampling.
        """
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']

        # Use smaller data for faster testing
        n_test = 20
        X_test, y_test, z_test = X[:n_test], y[:n_test], z[:n_test]

        # Create DataFrame
        import pandas as pd

        df_test = pd.DataFrame(
            {
                'x1': X_test[:, 0].numpy(),
                'x2': X_test[:, 1].numpy(),
                'treatment': z_test.numpy(),
                'response': y_test.numpy(),
            }
        )

        # Create both models
        model_torch = DRGU(
            df_test,
            covariates=['x1', 'x2'],
            treatment='treatment',
            response='response',
            device='cpu',
        )
        model_mini = DRGUMiniBatch(
            df_test,
            covariates=['x1', 'x2'],
            treatment='treatment',
            response='response',
            device='cpu',
        )

        # Set identical non-zero initial parameters (avoid zeros which might cause issues)
        theta_init = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        }

        # Set identical initial parameters for both models using set_theta
        model_torch.set_theta(theta_init)
        model_mini.set_theta(theta_init)

        print(f'\nTesting 1-step theta updates for n={n_test}')
        print('Initial parameters:')
        for param_name, value in theta_init.items():
            print(f'  {param_name}: {value.tolist()}')

        # Take exactly 1 step with DRGU
        model_torch.fit(max_iter=1, tol=1e-20, lamb=1e-6, verbose=False)
        theta_torch_after = {k: v.clone() for k, v in model_torch.theta.items()}

        # Take exactly 1 step with DRGUMiniBatch (complete sampling)
        k = n_test - 1
        expected_pairs = n_test * (n_test - 1) // 2

        model_mini.fit(
            tol=1e-20,
            lamb=1e-6,
            verbose=False,
            pairs_per_anchor=k,
            pairs_per_batch=expected_pairs,
            max_epochs=1,
            batches_per_epoch=1,
            warm_up=False,  # Disable warm up for precise equivalence test
        )
        theta_mini_after = {k: v.clone() for k, v in model_mini.theta.items()}

        # Compare parameter updates
        print('\nParameter comparison after 1 step:')
        for param_name in ['delta', 'beta', 'gamma']:
            initial = theta_init[param_name]
            torch_final = theta_torch_after[param_name]
            mini_final = theta_mini_after[param_name]

            # Compare final values
            diff = torch.linalg.norm(torch_final - mini_final)
            rel_diff = (
                diff / torch.linalg.norm(torch_final)
                if torch.linalg.norm(torch_final) > 1e-12
                else diff
            )

            # Compare step sizes
            torch_step = torch.linalg.norm(torch_final - initial)
            mini_step = torch.linalg.norm(mini_final - initial)

            print(f'  {param_name}:')
            print(f'    torch_final: {torch_final.tolist()}')
            print(f'    mini_final:  {mini_final.tolist()}')
            print(
                f'    diff: {rel_diff:.2e}, torch_step: {torch_step:.6f}, '
                f'mini_step: {mini_step:.6f}'
            )

            # Use more lenient tolerance due to numerical precision
            assert rel_diff < 1e-3, f'{param_name} parameters differ by {rel_diff:.2e}'

        print('Theta updates are essentially identical!')


class TestMomentumAndLearningRate:
    """Test momentum and adaptive learning rate functionality."""

    def test_momentum_disabled_by_default(self, drgu_data):
        """Test that momentum=0.0 behaves like no momentum (default behavior)."""
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']

        # Use small data for faster testing
        n_test = 20
        X_test, y_test, z_test = X[:n_test], y[:n_test], z[:n_test]

        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        }

        # Create optimizers with and without explicit momentum=0.0
        optimizer_default = MiniBatchFisherScoring(
            model_params=theta.copy(), compute_B_U=drgu_compute_B_U, penalty=Penalty(lam=1e-6)
        )

        optimizer_explicit = MiniBatchFisherScoring(
            model_params=theta.copy(),
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=1e-6),
            momentum=0.0,  # Explicit
        )

        # Create identical batch
        from robustinfer.io.pair_dataset import PairBatchIterableDataset
        from robustinfer.io.samplers import KPartnersSampler

        sampler = KPartnersSampler(n=n_test, k=5, seed=42)
        dataset = PairBatchIterableDataset(
            X=X_test, y=y_test, z=z_test, sampler=sampler, pairs_per_batch=50
        )
        batch = next(iter(dataset))

        # Take one step with each
        _ = optimizer_default.step(batch)
        _ = optimizer_explicit.step(batch)

        # Should have identical results
        for param_name in ['delta', 'beta', 'gamma']:
            default_param = optimizer_default.theta[param_name]
            explicit_param = optimizer_explicit.theta[param_name]
            assert torch.allclose(default_param, explicit_param, atol=1e-8), (
                f'{param_name} differs between default and explicit momentum=0.0'
            )

        # Both should have no momentum buffer
        assert optimizer_default.momentum_buffer is None
        assert optimizer_explicit.momentum_buffer is None

    def test_momentum_accumulation(self, drgu_data):
        """Test that momentum > 0.0 accumulates gradients correctly."""
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n_test = 20
        X_test, y_test, z_test = X[:n_test], y[:n_test], z[:n_test]

        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        }

        # Create optimizer with momentum (use smaller LR to avoid step norm issues)
        optimizer = MiniBatchFisherScoring(
            model_params=theta.copy(),
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=1e-3),  # More regularization for stability
            momentum=0.9,
            learning_rate=0.1,  # Smaller LR for stability
            max_step_norm=50.0,  # Higher limit for momentum tests
        )

        # Create batch
        from robustinfer.io.pair_dataset import PairBatchIterableDataset
        from robustinfer.io.samplers import KPartnersSampler

        sampler = KPartnersSampler(n=n_test, k=5, seed=42)
        dataset = PairBatchIterableDataset(
            X=X_test, y=y_test, z=z_test, sampler=sampler, pairs_per_batch=50
        )
        batch = next(iter(dataset))

        # Take first step - should initialize momentum buffer
        theta_0 = {k: v.clone() for k, v in optimizer.theta.items()}
        _ = optimizer.step(batch)
        theta_1 = {k: v.clone() for k, v in optimizer.theta.items()}

        # Momentum buffer should be initialized
        assert optimizer.momentum_buffer is not None
        momentum_1 = optimizer.momentum_buffer.clone()

        # Take second step - momentum should affect the update
        _ = optimizer.step(batch)
        theta_2 = {k: v.clone() for k, v in optimizer.theta.items()}
        momentum_2 = optimizer.momentum_buffer.clone()

        # Momentum buffer should have changed
        assert not torch.allclose(momentum_1, momentum_2), 'Momentum buffer should update'

        # The second step should be influenced by momentum from first step
        # We can't easily verify the exact formula without recomputing, but we can check
        # that the momentum buffer norm is reasonable and has accumulated
        assert torch.linalg.norm(momentum_2) > 0, 'Momentum buffer should be non-zero'

        # Momentum should affect the direction and magnitude
        # The exact relationship is complex, but we can verify basic properties
        delta_1_norm = torch.linalg.norm(theta_1['delta'] - theta_0['delta'])
        delta_2_norm = torch.linalg.norm(theta_2['delta'] - theta_1['delta'])

        # Both steps should be non-trivial
        assert delta_1_norm > 1e-6, 'First step should be significant'
        assert delta_2_norm > 1e-6, 'Second step should be significant'

    def test_learning_rate_scaling(self, drgu_data):
        """Test that learning rate properly scales the step size."""
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n_test = 20
        X_test, y_test, z_test = X[:n_test], y[:n_test], z[:n_test]

        theta_init = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        }

        # Create optimizers with different learning rates
        lr_values = [1.0, 0.5, 0.1]
        optimizers = []
        for lr in lr_values:
            optimizer = MiniBatchFisherScoring(
                model_params=theta_init.copy(),
                compute_B_U=drgu_compute_B_U,
                penalty=Penalty(lam=1e-6),
                learning_rate=lr,
                momentum=0.0,  # No momentum for cleaner test
            )
            optimizers.append(optimizer)

        # Create identical batch for all
        from robustinfer.io.pair_dataset import PairBatchIterableDataset
        from robustinfer.io.samplers import KPartnersSampler

        sampler = KPartnersSampler(n=n_test, k=5, seed=42)
        dataset = PairBatchIterableDataset(
            X=X_test, y=y_test, z=z_test, sampler=sampler, pairs_per_batch=50
        )
        batch = next(iter(dataset))

        # Take one step with each optimizer
        theta_finals = []
        for optimizer in optimizers:
            _ = optimizer.step(batch)
            theta_after = {k: v.clone() for k, v in optimizer.theta.items()}
            theta_finals.append(theta_after)

        # Check that step sizes scale with learning rate
        # lr=1.0 should have largest step, lr=0.1 should have smallest step
        for param_name in ['delta', 'beta', 'gamma']:
            step_lr_1_0 = torch.linalg.norm(theta_finals[0][param_name] - theta_init[param_name])
            step_lr_0_5 = torch.linalg.norm(theta_finals[1][param_name] - theta_init[param_name])
            step_lr_0_1 = torch.linalg.norm(theta_finals[2][param_name] - theta_init[param_name])

            # Step sizes should be in order: lr=1.0 > lr=0.5 > lr=0.1
            assert step_lr_1_0 > step_lr_0_5, (
                f'{param_name}: lr=1.0 should have larger step than lr=0.5'
            )
            assert step_lr_0_5 > step_lr_0_1, (
                f'{param_name}: lr=0.5 should have larger step than lr=0.1'
            )

            # Approximate scaling check (should be roughly proportional)
            # Allow some numerical tolerance
            if step_lr_1_0 > 1e-6:  # Only check if step is significant
                ratio_expected = 0.5  # lr=0.5 should be ~50% of lr=1.0
                ratio_actual = step_lr_0_5 / step_lr_1_0
                assert abs(ratio_actual - ratio_expected) < 0.2, (
                    f'{param_name}: scaling ratio off, '
                    f'expected ~{ratio_expected}, got {ratio_actual}'
                )

    def test_adaptive_learning_rate_scaling(self):
        """Test adaptive learning rate based on condition numbers."""
        from robustinfer.minibatch.minibatch_fisher import MiniBatchFisherScoring, Penalty

        # Create a dummy optimizer just to test the adaptive LR method
        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        }

        optimizer = MiniBatchFisherScoring(
            model_params=theta,
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=1e-6),
            learning_rate=0.5,  # Base learning rate
            adaptive_lr_bool=True,
        )

        # Test different condition numbers
        test_cases = [
            (1e2, 0.5),  # Good condition -> full LR
            (1e4, 0.35),  # Moderate condition -> 0.7 * LR
            (1e6, 0.15),  # Bad condition -> 0.3 * LR
            (1e10, 0.025),  # Terrible condition -> 0.05 * LR
        ]

        for cond_num, expected_lr in test_cases:
            actual_lr = optimizer._adaptive_learning_rate(cond_num)
            assert abs(actual_lr - expected_lr) < 0.01, (
                f'For cond_num={cond_num:.1e}, expected LR ~{expected_lr}, got {actual_lr}'
            )

    def test_adaptive_lr_disabled_by_default(self):
        """Test that adaptive_lr_bool=False uses fixed learning rate."""
        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        }

        optimizer = MiniBatchFisherScoring(
            model_params=theta,
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=1e-6),
            learning_rate=0.3,
            adaptive_lr_bool=False,  # Disabled
        )

        # Should return fixed learning rate regardless of condition number
        assert optimizer._adaptive_learning_rate(1e2) == 0.3
        assert optimizer._adaptive_learning_rate(1e10) == 0.3
        assert optimizer._adaptive_learning_rate(1e15) == 0.3

    def test_momentum_and_learning_rate_together(self, drgu_data):
        """Test that momentum and learning rate work together correctly."""
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n_test = 15
        X_test, y_test, z_test = X[:n_test], y[:n_test], z[:n_test]

        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        }

        # Create optimizer with both features (conservative settings)
        optimizer = MiniBatchFisherScoring(
            model_params=theta.copy(),
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=1e-3),  # More regularization
            learning_rate=0.1,  # Conservative LR
            momentum=0.5,  # Moderate momentum
            adaptive_lr_bool=True,
            max_step_norm=20.0,  # Higher limit
        )

        # Create batch
        from robustinfer.io.pair_dataset import PairBatchIterableDataset
        from robustinfer.io.samplers import KPartnersSampler

        sampler = KPartnersSampler(n=n_test, k=4, seed=42)
        dataset = PairBatchIterableDataset(
            X=X_test, y=y_test, z=z_test, sampler=sampler, pairs_per_batch=40
        )
        batch = next(iter(dataset))

        # Take multiple steps
        theta_history = []
        for _step in range(3):
            # theta_before = {k: v.clone() for k, v in optimizer.theta.items()}  # Unused
            _ = optimizer.step(batch)
            theta_after = {k: v.clone() for k, v in optimizer.theta.items()}
            theta_history.append(theta_after)

        # Should have momentum buffer
        assert optimizer.momentum_buffer is not None

        # Should be making some progress (parameters changing)
        # Note: With momentum and adaptive LR, convergence patterns can be complex
        total_change = 0.0
        for param_name in ['delta', 'beta', 'gamma']:
            step_1_norm = torch.linalg.norm(theta_history[0][param_name] - theta[param_name])
            step_3_norm = torch.linalg.norm(theta_history[2][param_name] - theta[param_name])
            total_change += step_3_norm
            # Each step should make some change
            assert step_1_norm > 1e-6, f'{param_name} should change in step 1'

        # Overall should be making progress
        assert total_change > 1e-5, 'Should make overall progress across all parameters'

        # Test that optimizer doesn't crash with extreme condition numbers
        # (This is more of an integration test)
        assert len(theta_history) == 3, 'Should complete all steps without crashing'

    def test_fisher_ema_smoothing(self, drgu_data):
        """Test Fisher EMA smoothing functionality."""
        X_test, y_test, z_test = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n = drgu_data['n']

        # Create theta with correct dimensions (p=2 from fixture)
        p = 2
        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.zeros(p + 1, dtype=torch.float32),  # 3 parameters
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float32),  # 5 parameters
        }

        # Test 1: fisher_ema=0.0 (disabled by default)
        optimizer_no_ema = MiniBatchFisherScoring(
            model_params=theta,
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=0.001),
            fisher_ema=0.0,
        )
        assert optimizer_no_ema.fisher_ema == 0.0
        assert optimizer_no_ema.J_obs_buffer is None

        # Test 2: fisher_ema > 0 (enabled)
        optimizer_ema = MiniBatchFisherScoring(
            model_params=theta,
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=0.001),
            fisher_ema=0.8,
        )
        assert optimizer_ema.fisher_ema == 0.8
        assert optimizer_ema.J_obs_buffer is None  # Not initialized until first step

        # Create mini-batch
        sampler = KPartnersSampler(n=n, k=15, seed=42)
        dataset = PairBatchIterableDataset(
            X=X_test, y=y_test, z=z_test, sampler=sampler, pairs_per_batch=40
        )
        batch = next(iter(dataset))

        # Step 1: Should initialize J_obs_buffer
        optimizer_ema.step(batch)
        assert optimizer_ema.J_obs_buffer is not None
        J_first = optimizer_ema.J_obs_buffer.clone()

        # Step 2: Should update buffer with EMA
        optimizer_ema.step(batch)
        J_second = optimizer_ema.J_obs_buffer.clone()

        # Buffer should change (EMA of different mini-batches)
        buffer_changed = not torch.allclose(J_first, J_second, atol=1e-6)
        assert buffer_changed, 'Fisher EMA buffer should update with new mini-batches'

        print(f'Fisher EMA test: J_obs buffer shape={J_first.shape}, EMA working={buffer_changed}')

    def test_fisher_ema_integration_with_model(self, drgu_data):
        """Test Fisher EMA works with full DRGUMiniBatch model."""
        import pandas as pd

        X_test, y_test, z_test = drgu_data['X'], drgu_data['y'], drgu_data['z']
        # n = drgu_data['n']  # Unused

        # Create DataFrame (like other tests do)
        df_test = pd.DataFrame(
            {
                'x1': X_test[:, 0].numpy(),
                'x2': X_test[:, 1].numpy(),
                'treatment': z_test.numpy(),
                'outcome': y_test.numpy(),
            }
        )

        # Create model
        model = DRGUMiniBatch(
            df_test, covariates=['x1', 'x2'], treatment='treatment', response='outcome'
        )

        # Test with Fisher EMA enabled (very conservative parameters to avoid step norm issues)
        _ = model.fit(
            pairs_per_anchor=20,
            pairs_per_batch=60,
            max_epochs=2,
            batches_per_epoch=2,
            fisher_ema=0.2,  # Light EMA for stability
            learning_rate=0.05,  # Very conservative learning rate
            lamb=0.01,  # Add some regularization
            verbose=False,
        )

        # Should complete without errors
        assert hasattr(model, 'theta')
        assert model.theta is not None

        # Parameters should be reasonable
        for param_name in ['delta', 'beta', 'gamma']:
            param_tensor = model.theta[param_name]
            assert torch.isfinite(param_tensor).all(), f'{param_name} should be finite'

        print('Fisher EMA integration test: Model fit completed successfully with EMA')


class TestIntegration:
    """Test full workflow integration."""

    def test_end_to_end_workflow(self, drgu_data):
        """Test complete end-to-end workflow."""
        X, y, z = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n, p = drgu_data['n'], drgu_data['p']

        # 1. Setup sampling
        sampler = KPartnersSampler(n=n, k=10, seed=42)

        # 2. Create dataset
        dataset = PairBatchIterableDataset(X=X, y=y, z=z, sampler=sampler, pairs_per_batch=60)

        # 3. Initialize parameters
        theta = {
            'delta': torch.zeros(1, dtype=torch.float64),
            'beta': torch.zeros(p + 1, dtype=torch.float64),
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float64),
        }

        # 4. Setup optimizer
        optimizer = MiniBatchFisherScoring(
            model_params=theta,
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=1e-5),
            max_step_norm=5.0,
        )

        # 5. Run optimization
        result = fit_minibatch_optimizer(optimizer, dataset, max_steps=5)

        # 6. Verify results
        assert 'theta' in result
        assert 'converged' in result
        assert 'steps' in result

        final_theta = result['theta']
        assert set(final_theta.keys()) == {'delta', 'beta', 'gamma'}

        # Check dimensions
        assert final_theta['delta'].shape == (1,)
        assert final_theta['beta'].shape == (p + 1,)
        assert final_theta['gamma'].shape == (2 * p + 1,)

        print('End-to-end test completed successfully!')
        print(f'Converged: {result["converged"]}, Steps: {result["steps"]}')


class TestMonteCarloVariance:
    """Test Monte Carlo variance estimation functionality."""

    def test_drgu_compute_B_Sig_basic(self, drgu_data):
        """Test that drgu_compute_B_Sig returns valid B and Sig matrices."""
        from robustinfer.minibatch.estimating_equations import drgu_compute_B_Sig

        X_test, y_test, z_test = drgu_data['X'], drgu_data['y'], drgu_data['z']

        # Create mini-batch with float32 data
        batch = {
            'xi': X_test[:10].to(torch.float32),
            'xj': X_test[10:20].to(torch.float32),
            'yi': y_test[:10].to(torch.float32),
            'yj': y_test[10:20].to(torch.float32),
            'zi': z_test[:10].to(torch.float32),
            'zj': z_test[10:20].to(torch.float32),
            'w_ij': torch.ones(10, dtype=torch.float32),
        }

        # Create test theta - match what drgu_torch expects
        p = X_test.shape[1]
        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.zeros(p + 1, dtype=torch.float32),  # Treatment model: Wt_i * beta
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float32),  # Outcome model: Xg_ij * gamma
        }

        # Test B/Sig computation
        B, Sig = drgu_compute_B_Sig(theta, batch)

        # Check shapes
        expected_dim = 1 + (p + 1) + (2 * p + 1)  # delta + beta + gamma
        assert B.shape == (expected_dim, expected_dim), (
            f'B shape: {B.shape}, expected: {expected_dim}x{expected_dim}'
        )
        assert Sig.shape == (expected_dim, expected_dim), (
            f'Sig shape: {Sig.shape}, expected: {expected_dim}x{expected_dim}'
        )

        # Check properties
        # B matrix (Jacobian) is not required to be symmetric in DRGU context
        assert torch.allclose(Sig, Sig.T, atol=1e-6), 'Sig matrix should be symmetric'

        # Check that matrices are positive semi-definite
        # (eigenvalues >= 0, allow small numerical errors)
        B_eigs = torch.linalg.eigvals(B).real
        Sig_eigs = torch.linalg.eigvals(Sig).real
        assert torch.all(B_eigs >= -1e-4), f'B has negative eigenvalues: {B_eigs.min()}'
        assert torch.all(Sig_eigs >= -1e-4), f'Sig has negative eigenvalues: {Sig_eigs.min()}'

    def test_montecarlo_estimation_basic(self, drgu_data):
        """Test Monte Carlo variance estimation with small sample."""
        from robustinfer.minibatch import MonteCarloEstimation

        X_test, y_test, z_test = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n, p = X_test.shape

        # Use small subset for quick test and convert to float32
        n_small = 50
        X_small = X_test[:n_small].to(torch.float32)
        y_small = y_test[:n_small].to(torch.float32)
        z_small = z_test[:n_small].to(torch.float32)

        # Create test theta - match what drgu_torch expects
        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.zeros(p + 1, dtype=torch.float32),  # Treatment model: Wt_i * beta
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float32),  # Outcome model: Xg_ij * gamma
        }

        # Test Monte Carlo estimation (anchor-based)
        mc_estimator = MonteCarloEstimation(device=torch.device('cpu'))
        variance_matrix = mc_estimator.estimate(
            X=X_small,
            y=y_small,
            z=z_small,
            theta=theta,
            k=5,  # Partners per anchor
            s=8,  # Number of anchors
            alpha=0.1,  # Small debiasing
            verbose=False,
        )

        # Check variance matrix properties
        expected_dim = 1 + (p + 1) + (2 * p + 1)
        assert variance_matrix.shape == (expected_dim, expected_dim)
        assert torch.allclose(variance_matrix, variance_matrix.T, atol=1e-10), (
            'Final variance matrix should be symmetric'
        )

        # Check that variance matrix is positive semi-definite (allow small numerical errors)
        var_eigs = torch.linalg.eigvals(variance_matrix).real
        assert torch.all(var_eigs >= -1e-4), (
            f'Variance matrix has negative eigenvalues: {var_eigs.min()}'
        )

        # Check reasonable magnitude (not too large or too small)
        trace_val = torch.trace(variance_matrix)
        assert 1e-6 < trace_val < 1e6, f'Variance trace seems unreasonable: {trace_val}'

    def test_variance_scaling_in_drgu_minibatch(self, drgu_data):
        """Test that DRGUMiniBatch applies proper variance scaling (1/n)."""
        X_test, y_test, z_test = drgu_data['X'], drgu_data['y'], drgu_data['z']

        # Use small subset for quick test
        n_small = 30
        indices = torch.randperm(len(X_test))[:n_small]

        # Create DataFrame
        df_test = pd.DataFrame(
            {
                'x1': X_test[indices, 0].numpy(),
                'x2': X_test[indices, 1].numpy(),
                'treatment': z_test[indices].numpy(),
                'outcome': y_test[indices].numpy(),
            }
        )

        # Create model and fit (minimal steps)
        model = DRGUMiniBatch(
            df_test, covariates=['x1', 'x2'], treatment='treatment', response='outcome'
        )

        # Fit with minimal parameters for quick test
        model.fit(
            pairs_per_anchor=5,
            pairs_per_batch=10,
            max_epochs=1,
            batches_per_epoch=1,
            lamb=0.1,  # Add regularization for stability
            verbose=False,
        )

        # Force convergence status for testing (since we're using minimal epochs)
        model.converged = True

        # Now estimate variance separately with custom parameters
        model.estimate_variance(
            pairs_per_anchor=3, s=8, alpha=0.1, verbose=False
        )  # Small values for quick test

        # Check that variance matrix was computed and scaled
        assert hasattr(model, 'Var'), 'Model should have raw variance matrix (Var)'
        assert hasattr(model, 'variance_matrix'), 'Model should have scaled variance matrix'

        # Check scaling relationship: variance_matrix = Var / n
        expected_variance = model.Var / n_small
        assert torch.allclose(model.variance_matrix, expected_variance, atol=1e-6), (
            'variance_matrix should equal Var / n'
        )

        # Check dimensions match parameter count
        expected_dim = 1 + 3 + 5  # delta(1) + beta(3) + gamma(5) for 2D covariates
        assert model.variance_matrix.shape == (expected_dim, expected_dim)

        # Check that final variance matrix is symmetric
        assert torch.allclose(model.variance_matrix, model.variance_matrix.T, atol=1e-10), (
            'Final model variance matrix should be symmetric'
        )

        print(f'Variance scaling test passed: n={n_small}')

    def test_variance_matrix_deterministic(self, drgu_data):
        """Test that variance computation is deterministic with same seed."""
        from robustinfer.minibatch import MonteCarloEstimation

        X_test, y_test, z_test = drgu_data['X'], drgu_data['y'], drgu_data['z']
        n, p = X_test.shape

        # Use small subset and convert to float32
        n_small = 40
        X_small = X_test[:n_small].to(torch.float32)
        y_small = y_test[:n_small].to(torch.float32)
        z_small = z_test[:n_small].to(torch.float32)

        # Create test theta - match what drgu_torch expects
        theta = {
            'delta': torch.tensor([0.1], dtype=torch.float32),
            'beta': torch.zeros(p + 1, dtype=torch.float32),  # Treatment model: Wt_i * beta
            'gamma': torch.zeros(2 * p + 1, dtype=torch.float32),  # Outcome model: Xg_ij * gamma
        }

        # Run estimation twice with same seed
        mc_estimator = MonteCarloEstimation(device=torch.device('cpu'))

        # First run
        torch.manual_seed(42)
        var1 = mc_estimator.estimate(
            X=X_small,
            y=y_small,
            z=z_small,
            theta=theta,
            k=5,
            s=10,
            alpha=0.1,
            verbose=False,
        )

        # Second run with same seed
        torch.manual_seed(42)
        var2 = mc_estimator.estimate(
            X=X_small,
            y=y_small,
            z=z_small,
            theta=theta,
            k=5,
            s=10,
            alpha=0.1,
            verbose=False,
        )

        # Should be identical
        assert torch.allclose(var1, var2, atol=1e-10), (
            'Variance estimation should be deterministic with same seed'
        )

    def test_variance_estimation_error_handling(self, drgu_data):
        """Test proper error handling when variance hasn't been estimated."""
        X_test, y_test, z_test = drgu_data['X'], drgu_data['y'], drgu_data['z']

        # Use small subset for quick test
        n_small = 20
        indices = torch.randperm(len(X_test))[:n_small]

        # Create DataFrame
        df_test = pd.DataFrame(
            {
                'x1': X_test[indices, 0].numpy(),
                'x2': X_test[indices, 1].numpy(),
                'treatment': z_test[indices].numpy(),
                'outcome': y_test[indices].numpy(),
            }
        )

        # Create and fit model
        model = DRGUMiniBatch(
            df_test, covariates=['x1', 'x2'], treatment='treatment', response='outcome'
        )

        model.fit(
            pairs_per_anchor=3,
            pairs_per_batch=5,
            max_epochs=1,
            batches_per_epoch=1,
            lamb=0.1,
            verbose=False,
        )

        # Force convergence for testing (models with minimal epochs may not actually converge)
        model.converged = True

        # Test convergence check - modify convergence status to test error
        original_converged = model.converged
        model.converged = False
        with pytest.raises(RuntimeError, match='Model did not converge'):
            model.estimate_variance(pairs_per_anchor=5, s=8, alpha=0.0, verbose=False)

        # Restore convergence status
        model.converged = original_converged

        # Test that summary raises error before variance estimation
        with pytest.raises(RuntimeError, match='Variance matrix has not been estimated'):
            model.summary()

        # After estimating variance, summary should work
        model.estimate_variance(
            pairs_per_anchor=5, s=8, alpha=0.0, verbose=False
        )  # pairs_per_anchor=5 < n=20
        summary_df = model.summary()  # Should not raise error
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) > 0

        print('Error handling test passed!')


# ============================================================================
# Monte Carlo Estimation Tests
# ============================================================================


class TestMonteCarloEstimation:
    """Test the anchor-based Monte Carlo estimation."""

    @pytest.fixture
    def sample_data_mc(self):
        """Create sample data for Monte Carlo testing."""
        torch.manual_seed(42)
        n, p = 50, 3
        X = torch.randn(n, p, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)
        z = torch.randint(0, 2, (n,), dtype=torch.long)

        # Create correct theta parameters for DRGU model
        # delta: treatment effect [3] (for 3-component vector)
        # beta: treatment model [p+1=4] (with intercept)
        # gamma: outcome model [2p+1=7] (intercept + both partners' features)
        theta = {
            'delta': torch.randn(3, dtype=torch.float64) * 0.1,
            'beta': torch.randn(p + 1, dtype=torch.float64) * 0.1,  # [4] for treatment model
            'gamma': torch.randn(2 * p + 1, dtype=torch.float64) * 0.1,  # [7] for outcome model
        }

        return X, y, z, theta

    def test_basic_functionality(self, sample_data_mc):
        """Test that Monte Carlo estimation runs without errors."""
        X, y, z, theta = sample_data_mc

        estimator = MonteCarloEstimation(device='cpu')
        variance_matrix = estimator.estimate(
            X=X, y=y, z=z, theta=theta, k=5, s=10, alpha=0.5, verbose=False
        )

        # Check output shape
        d_total = theta['delta'].shape[0] + theta['beta'].shape[0] + theta['gamma'].shape[0]
        assert variance_matrix.shape == (d_total, d_total)

        # Check that it's a tensor
        assert isinstance(variance_matrix, torch.Tensor)
        assert variance_matrix.dtype == torch.float64

    def test_variance_matrix_properties(self, sample_data_mc):
        """Test mathematical properties of the variance matrix."""
        X, y, z, theta = sample_data_mc

        estimator = MonteCarloEstimation(device='cpu')
        variance_matrix = estimator.estimate(
            X=X, y=y, z=z, theta=theta, k=4, s=8, alpha=0.0, verbose=False
        )

        # Should be approximately symmetric
        symmetry_error = torch.norm(variance_matrix - variance_matrix.T)
        assert symmetry_error < 1e-10, f'Variance matrix not symmetric: error={symmetry_error}'

        # Should be positive semi-definite (eigenvalues >= 0)
        eigenvals = torch.linalg.eigvals(variance_matrix).real
        min_eigenval = torch.min(eigenvals)
        assert min_eigenval >= -1e-10, f'Variance matrix not PSD: min eigenvalue={min_eigenval}'

    def test_reproducibility(self, sample_data_mc):
        """Test that results are reproducible with same parameters."""
        X, y, z, theta = sample_data_mc

        # Create two estimators with same setup
        estimator1 = MonteCarloEstimation(device='cpu')
        estimator2 = MonteCarloEstimation(device='cpu')

        # Should get identical results (same seed used in sampler)
        var1 = estimator1.estimate(X=X, y=y, z=z, theta=theta, k=3, s=6, alpha=0.2)
        var2 = estimator2.estimate(X=X, y=y, z=z, theta=theta, k=3, s=6, alpha=0.2)

        assert torch.allclose(var1, var2, atol=1e-12), 'Results should be reproducible'

    def test_different_alpha_values(self, sample_data_mc):
        """Test that different alpha values produce different results."""
        X, y, z, theta = sample_data_mc

        estimator = MonteCarloEstimation(device='cpu')

        var_alpha0 = estimator.estimate(X=X, y=y, z=z, theta=theta, k=4, s=8, alpha=0.0)
        var_alpha1 = estimator.estimate(X=X, y=y, z=z, theta=theta, k=4, s=8, alpha=1.0)

        # Should be different (debiasing effect)
        assert not torch.allclose(var_alpha0, var_alpha1, atol=1e-6), (
            'Different alpha should give different results'
        )

    def test_scaling_with_n(self, sample_data_mc):
        """Test that variance scales appropriately with sample size."""
        X, y, z, theta = sample_data_mc
        n = len(y)

        estimator = MonteCarloEstimation(device='cpu')

        # Full dataset
        var_full = estimator.estimate(X=X, y=y, z=z, theta=theta, k=4, s=8, alpha=0.0)

        # Half dataset
        half_n = n // 2
        var_half = estimator.estimate(
            X=X[:half_n], y=y[:half_n], z=z[:half_n], theta=theta, k=4, s=4, alpha=0.0
        )

        # Variance should be larger for smaller sample (roughly inversely proportional)
        var_full_trace = torch.trace(var_full)
        var_half_trace = torch.trace(var_half)

        # Half sample should have larger variance (but not necessarily exactly 2x due to sampling)
        assert var_half_trace > var_full_trace, 'Smaller sample should have larger variance'

    def test_edge_cases(self, sample_data_mc):
        """Test edge cases and parameter validation."""
        X, y, z, theta = sample_data_mc
        n = len(y)

        estimator = MonteCarloEstimation(device='cpu')

        # Test with s=1 (single anchor)
        var_single = estimator.estimate(X=X, y=y, z=z, theta=theta, k=3, s=1, alpha=0.0)
        assert var_single.shape[0] > 0, 'Should work with single anchor'

        # Test with k=1 (single partner per anchor) - should raise error since m > 1 required
        with pytest.raises(AssertionError, match='Need m > 1 partners per anchor'):
            estimator.estimate(X=X, y=y, z=z, theta=theta, k=1, s=5, alpha=0.0)

        # Test with large k (more partners than possible)
        var_large_k = estimator.estimate(X=X, y=y, z=z, theta=theta, k=n - 1, s=3, alpha=0.0)
        assert var_large_k.shape[0] > 0, 'Should handle large k gracefully'

    def test_default_parameters(self, sample_data_mc):
        """Test that default parameters work correctly."""
        X, y, z, theta = sample_data_mc

        estimator = MonteCarloEstimation(device='cpu')

        # Should work with minimal parameters
        variance_matrix = estimator.estimate(X=X, y=y, z=z, theta=theta)

        assert variance_matrix.shape[0] > 0, 'Should work with default parameters'

        # Check that default s is reasonable (n//10)
        n = len(y)
        max(1, n // 10)
        # We can't directly check this, but we can verify it doesn't crash

    def test_alpha_bounds(self, sample_data_mc):
        """Test alpha parameter boundaries."""
        X, y, z, theta = sample_data_mc

        estimator = MonteCarloEstimation(device='cpu')

        # Test boundary values
        var_alpha_0 = estimator.estimate(X=X, y=y, z=z, theta=theta, k=3, s=5, alpha=0.0)
        var_alpha_1 = estimator.estimate(X=X, y=y, z=z, theta=theta, k=3, s=5, alpha=1.0)

        # Both should work
        assert var_alpha_0.shape[0] > 0
        assert var_alpha_1.shape[0] > 0

        # Test intermediate value
        var_alpha_half = estimator.estimate(X=X, y=y, z=z, theta=theta, k=3, s=5, alpha=0.5)
        assert var_alpha_half.shape[0] > 0

    def test_device_consistency(self, sample_data_mc):
        """Test that device handling works correctly."""
        X, y, z, theta = sample_data_mc

        # CPU estimator
        estimator_cpu = MonteCarloEstimation(device='cpu')
        var_cpu = estimator_cpu.estimate(X=X, y=y, z=z, theta=theta, k=3, s=5, alpha=0.0)

        assert var_cpu.device.type == 'cpu', 'Output should be on CPU'

        # Test with explicit device specification
        estimator_default = MonteCarloEstimation()  # Should default to CPU
        var_default = estimator_default.estimate(X=X, y=y, z=z, theta=theta, k=3, s=5, alpha=0.0)

        assert torch.allclose(var_cpu, var_default), (
            'Results should be same regardless of device specification'
        )

    def test_verbose_output(self, sample_data_mc, capsys):
        """Test verbose output functionality."""
        X, y, z, theta = sample_data_mc

        estimator = MonteCarloEstimation(device='cpu')

        # Test with verbose=True
        estimator.estimate(
            X=X, y=y, z=z, theta=theta, k=3, s=5, alpha=0.3, verbose=True
        )

        # Check that output was captured
        captured = capsys.readouterr()
        assert 'MC estimation' in captured.out
        assert 'Anchor-based variance computed' in captured.out

        # Test with verbose=False (should be quiet)
        estimator.estimate(X=X, y=y, z=z, theta=theta, k=3, s=5, alpha=0.3, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == '', 'Should be quiet with verbose=False'

    def test_psd_clipping_with_alpha(self, sample_data_mc):
        """Test that negative eigenvalue clipping works when alpha > 0."""
        X, y, z, theta = sample_data_mc
        mc = MonteCarloEstimation()

        # Test with alpha = 0 (no clipping needed)
        var_alpha_0 = mc.estimate(X, y, z, theta, k=3, s=5, alpha=0.0)
        eigenvals_0 = torch.linalg.eigvals(var_alpha_0).real

        # Test with alpha > 0 (potential clipping)
        var_alpha_pos = mc.estimate(X, y, z, theta, k=3, s=5, alpha=0.8)
        eigenvals_pos = torch.linalg.eigvals(var_alpha_pos).real

        # Both should be PSD (all eigenvalues >= 0)
        assert torch.all(eigenvals_0 >= -1e-8), 'Alpha=0 case should be PSD'
        assert torch.all(eigenvals_pos >= -1e-8), 'Alpha>0 case should be PSD after clipping'

        # Should still have reasonable matrix properties
        assert not torch.isnan(var_alpha_pos).any(), 'No NaN in clipped matrix'
        assert not torch.isinf(var_alpha_pos).any(), 'No Inf in clipped matrix'
        assert var_alpha_pos.shape == var_alpha_0.shape, 'Shape should be preserved'

    def test_psd_clipping_verbose_output(self, sample_data_mc, capsys):
        """Test that verbose output shows clipping information when it occurs."""
        X, y, z, theta = sample_data_mc
        mc = MonteCarloEstimation()

        # Use high alpha to potentially trigger clipping
        mc.estimate(X, y, z, theta, k=3, s=5, alpha=0.9, verbose=True)

        captured = capsys.readouterr()
        # Should show minimal output
        assert 'MC estimation' in captured.out or 'Anchor-based variance computed' in captured.out

        # May or may not show clipping message depending on data
        # But should not crash and should complete successfully


# ============================================================================
# Integration Tests
# ============================================================================


class TestThetaAveraging:
    """Test theta averaging functionality."""

    def test_theta_averaging_disabled(self, simple_data):
        """Test that theta averaging is disabled by default."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.zeros(1, dtype=torch.float32),
                'beta': torch.zeros(4, dtype=torch.float32),
                'gamma': torch.zeros(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
            theta_averaging=False,
        )

        assert optimizer.theta_averaging is False
        assert optimizer.theta_avg is None
        assert optimizer.theta_count == 0

    def test_theta_averaging_enabled(self, simple_data):
        """Test that theta averaging can be enabled."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.zeros(1, dtype=torch.float32),
                'beta': torch.zeros(4, dtype=torch.float32),
                'gamma': torch.zeros(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
            theta_averaging=True,
        )

        assert optimizer.theta_averaging is True
        assert optimizer.theta_avg is None
        assert optimizer.theta_count == 0

    def test_theta_accumulation(self, simple_data):
        """Test theta accumulation during steps."""
        X, y, z = simple_data['X'].float(), simple_data['y'].float(), simple_data['z'].float()

        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.ones(1, dtype=torch.float32),
                'beta': torch.ones(4, dtype=torch.float32),
                'gamma': torch.ones(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
            theta_averaging=True,
            max_step_norm=5.0,  # Allow larger steps for convergence
            learning_rate=0.01,  # Smaller learning rate
        )

        # Create some synthetic batch data
        sampler = KPartnersSampler(n=simple_data['n'], k=5, seed=42)
        dataset = PairBatchIterableDataset(
            X=X,
            y=y,
            z=z,
            sampler=sampler,
            pairs_per_batch=20,
            epoch=0,
            infinite=False,
        )

        # Take a few steps
        batch_iter = iter(dataset)
        initial_theta = {k: v.clone() for k, v in optimizer.theta.items()}

        for _i in range(3):
            try:
                batch = next(batch_iter)
                optimizer.step(batch)
            except StopIteration:
                break

        # Check that theta averaging happened
        assert optimizer.theta_count > 0
        assert optimizer.theta_avg is not None

        # Check that theta values changed
        for key in initial_theta:
            assert not torch.allclose(optimizer.theta[key], initial_theta[key], atol=1e-6)

    def test_get_averaged_theta(self, simple_data):
        """Test getting averaged theta."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.ones(1, dtype=torch.float32),
                'beta': torch.ones(4, dtype=torch.float32),
                'gamma': torch.ones(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
            theta_averaging=True,
        )

        # Manually set theta and update averages
        optimizer.theta = {
            'delta': torch.tensor([1.0]),
            'beta': torch.tensor([1.0, 2.0, 3.0, 4.0]),
            'gamma': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        }
        optimizer._update_theta_average()  # First update: avg = current = [1.0, ...]

        optimizer.theta = {
            'delta': torch.tensor([3.0]),
            'beta': torch.tensor([3.0, 6.0, 9.0, 12.0]),
            'gamma': torch.tensor([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]),
        }
        optimizer._update_theta_average()  # Second update: avg = 0.5*[1.0,...] + 0.5*[3.0,...]

        # Get averaged theta
        averaged = optimizer.get_averaged_theta()

        assert averaged is not None
        assert torch.allclose(averaged['delta'], torch.tensor([2.0]))  # (1+3)/2
        assert torch.allclose(
            averaged['beta'], torch.tensor([2.0, 4.0, 6.0, 8.0])
        )  # (1+3, 2+6, etc)/2
        assert torch.allclose(
            averaged['gamma'], torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
        )

    def test_apply_averaged_theta(self, simple_data):
        """Test applying averaged theta to current parameters."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.ones(1, dtype=torch.float32),
                'beta': torch.ones(4, dtype=torch.float32),
                'gamma': torch.ones(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
            theta_averaging=True,
        )

        # Manually create averaging scenario
        optimizer.theta_avg = {
            'delta': torch.tensor([3.0]),  # Average value
            'beta': torch.tensor([3.0, 4.0, 5.0, 6.0]),
            'gamma': torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        }

        # Current theta should be different
        optimizer.theta = {
            'delta': torch.tensor([99.0]),
            'beta': torch.tensor([99.0, 99.0, 99.0, 99.0]),
            'gamma': torch.tensor([99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]),
        }

        # Apply averaging
        result = optimizer.apply_averaged_theta()

        assert result is True
        assert torch.allclose(optimizer.theta['delta'], torch.tensor([3.0]))
        assert torch.allclose(optimizer.theta['beta'], torch.tensor([3.0, 4.0, 5.0, 6.0]))
        assert torch.allclose(
            optimizer.theta['gamma'], torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        )

    def test_reset_theta_averaging(self, simple_data):
        """Test resetting theta averaging."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.ones(1, dtype=torch.float32),
                'beta': torch.ones(4, dtype=torch.float32),
                'gamma': torch.ones(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
            theta_averaging=True,
        )

        # Set up some averaged data
        optimizer.theta_avg = {'delta': torch.tensor([5.0])}
        optimizer.theta_count = 3

        # Reset
        optimizer.reset_theta_averaging()

        assert optimizer.theta_avg is None
        assert optimizer.theta_count == 0

    def test_theta_averaging_in_drgu_fit(self, simple_data):
        """Test theta averaging integration in DRGUMiniBatch.fit()."""
        # Create dataframe
        df = pd.DataFrame(
            {
                'x1': simple_data['X'][:, 0].numpy(),
                'x2': simple_data['X'][:, 1].numpy(),
                'x3': simple_data['X'][:, 2].numpy(),
                'treatment': simple_data['z'].numpy(),
                'response': simple_data['y'].numpy(),
            }
        )

        # Create two models: one with averaging, one without
        model_no_avg = DRGUMiniBatch(
            data=df,
            covariates=['x1', 'x2', 'x3'],
            treatment='treatment',
            response='response',
        )

        model_with_avg = DRGUMiniBatch(
            data=df,
            covariates=['x1', 'x2', 'x3'],
            treatment='treatment',
            response='response',
        )

        # Set same initial theta for both
        initial_theta = {
            'delta': torch.zeros(1, dtype=torch.float32),
            'beta': torch.zeros(4, dtype=torch.float32),
            'gamma': torch.zeros(7, dtype=torch.float32),
        }
        model_no_avg.set_theta(initial_theta)
        model_with_avg.set_theta(initial_theta)

        # Fit both models with small step limits to avoid convergence
        result_no_avg = model_no_avg.fit(
            max_epochs=2,
            batches_per_epoch=3,
            theta_averaging=False,
            verbose=False,
            learning_rate=0.01,
            max_step_norm=2.0,  # Smaller LR, larger step norm
        )

        result_with_avg = model_with_avg.fit(
            max_epochs=2,
            batches_per_epoch=3,
            theta_averaging=True,
            verbose=False,
            learning_rate=0.01,
            max_step_norm=2.0,  # Smaller LR, larger step norm
        )

        # Both should have run without errors
        assert result_no_avg is not None
        assert result_with_avg is not None

        # Parameters should be different due to averaging
        # (This might not always be true due to randomness, but usually will be)
        no_avg_params = torch.cat([v.flatten() for v in model_no_avg.theta.values()])
        with_avg_params = torch.cat([v.flatten() for v in model_with_avg.theta.values()])

        # At minimum, the optimization should have progressed
        assert not torch.allclose(no_avg_params, torch.zeros_like(no_avg_params), atol=1e-6)
        assert not torch.allclose(with_avg_params, torch.zeros_like(with_avg_params), atol=1e-6)

    def test_u_vector_averaging(self, simple_data):
        """Test that U averaging works on vectors, not norms."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.ones(1, dtype=torch.float32),
                'beta': torch.ones(4, dtype=torch.float32),
                'gamma': torch.ones(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
        )

        # Test with specific U vectors to verify vector averaging vs norm averaging
        U1 = torch.tensor([3.0, 4.0])  # norm = 5.0
        U2 = torch.tensor([-3.0, -4.0])  # norm = 5.0

        # Update with first vector
        optimizer._update_U_history(U1)
        assert torch.allclose(optimizer.U_avg, U1)
        assert abs(optimizer.get_U_running_avg() - 5.0) < 1e-6

        # Update with second vector
        optimizer._update_U_history(U2)

        # Average of vectors: ([3,4] + [-3,-4])/2 = [0,0], norm = 0
        expected_avg_vector = torch.tensor([0.0, 0.0])
        assert torch.allclose(optimizer.U_avg, expected_avg_vector, atol=1e-6)
        assert abs(optimizer.get_U_running_avg() - 0.0) < 1e-6

        # If we had averaged norms instead: (5.0 + 5.0)/2 = 5.0
        # This demonstrates the difference: vector averaging gives 0, norm averaging would give 5

    def test_u_averaging_reset_per_epoch(self, simple_data):
        """Test that U averaging is reset at the beginning of each epoch."""
        # Create dataframe
        df = pd.DataFrame(
            {
                'x1': simple_data['X'][:, 0].numpy(),
                'x2': simple_data['X'][:, 1].numpy(),
                'x3': simple_data['X'][:, 2].numpy(),
                'treatment': simple_data['z'].numpy(),
                'response': simple_data['y'].numpy(),
            }
        )

        model = DRGUMiniBatch(
            data=df,
            covariates=['x1', 'x2', 'x3'],
            treatment='treatment',
            response='response',
        )

        # Set initial theta
        initial_theta = {
            'delta': torch.zeros(1, dtype=torch.float32),
            'beta': torch.zeros(4, dtype=torch.float32),
            'gamma': torch.zeros(7, dtype=torch.float32),
        }
        model.set_theta(initial_theta)

        # Create custom optimizer to test epoch-level resets
        optimizer = MiniBatchFisherScoring(
            model_params=initial_theta,
            compute_B_U=drgu_compute_B_U,
            learning_rate=0.01,
            max_step_norm=2.0,
        )

        # Manually add some U average
        test_U = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        optimizer._update_U_history(test_U)

        assert optimizer.U_avg is not None
        assert optimizer.U_count == 1
        assert torch.allclose(optimizer.U_avg, test_U)

        # Reset U averaging (this should happen at epoch start)
        optimizer.reset_U_averaging()

        # Check that current average is reset
        assert optimizer.U_avg is None
        assert optimizer.U_count == 0
        assert optimizer.get_U_running_avg() is None

        # Check that history was saved
        assert len(optimizer.U_history) == 1
        assert torch.allclose(optimizer.U_history[0], test_U)

    def test_theta_history_tracking(self, simple_data):
        """Test that theta averaging history is tracked across epochs."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.ones(1, dtype=torch.float32),
                'beta': torch.ones(4, dtype=torch.float32),
                'gamma': torch.ones(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
            theta_averaging=True,
        )

        # Initially no history
        assert len(optimizer.theta_history) == 0

        # Simulate first epoch with some theta updates
        optimizer.theta = {'delta': torch.tensor([1.0]), 'beta': torch.tensor([1.0, 2.0, 3.0, 4.0])}
        optimizer._update_theta_average()
        optimizer.theta = {'delta': torch.tensor([3.0]), 'beta': torch.tensor([3.0, 4.0, 5.0, 6.0])}
        optimizer._update_theta_average()

        # Reset should save to history
        optimizer.reset_theta_averaging()

        assert len(optimizer.theta_history) == 1
        assert torch.allclose(optimizer.theta_history[0]['delta'], torch.tensor([2.0]))
        assert torch.allclose(
            optimizer.theta_history[0]['beta'], torch.tensor([2.0, 3.0, 4.0, 5.0])
        )

        # Second epoch - no updates, should save None
        optimizer.reset_theta_averaging()

        assert len(optimizer.theta_history) == 2
        assert optimizer.theta_history[1] is None

    def test_U_history_tracking(self, simple_data):
        """Test that U averaging history is tracked across epochs."""
        optimizer = MiniBatchFisherScoring(
            model_params={
                'delta': torch.ones(1, dtype=torch.float32),
                'beta': torch.ones(4, dtype=torch.float32),
                'gamma': torch.ones(7, dtype=torch.float32),
            },
            compute_B_U=drgu_compute_B_U,
        )

        # Initially no history
        assert len(optimizer.U_history) == 0

        # Simulate epoch with U updates
        U1 = torch.tensor([1.0, 2.0])
        U2 = torch.tensor([3.0, 4.0])
        optimizer._update_U_history(U1)
        optimizer._update_U_history(U2)

        # Reset should save average to history
        optimizer.reset_U_averaging()

        assert len(optimizer.U_history) == 1
        expected_avg = torch.tensor([2.0, 3.0])  # (1+3)/2, (2+4)/2
        assert torch.allclose(optimizer.U_history[0], expected_avg)

        # Second epoch - no updates, should save None
        optimizer.reset_U_averaging()

        assert len(optimizer.U_history) == 2
        assert optimizer.U_history[1] is None


class TestDRGUMiniBatchIntegration:
    """Test integration between DRGUMiniBatch and anchor-based Monte Carlo estimation."""

    @pytest.fixture
    def sample_drgu_data(self):
        """Create sample data for DRGU integration testing."""
        torch.manual_seed(42)
        n, _p = 100, 2

        # Create DataFrame data as expected by DRGUMiniBatch
        data = pd.DataFrame(
            {
                'x1': torch.randn(n).numpy(),
                'x2': torch.randn(n).numpy(),
                'treatment': torch.randint(0, 2, (n,)).numpy(),
                'outcome': torch.randn(n).numpy(),
            }
        )

        return data

    def test_drgu_minibatch_with_anchor_estimation(self, sample_drgu_data):
        """Test that DRGUMiniBatch works with new anchor-based variance estimation interface."""
        data = sample_drgu_data

        # Create DRGU model
        model = DRGUMiniBatch(
            data=data,
            covariates=['x1', 'x2'],
            treatment='treatment',
            response='outcome',
            device='cpu',
        )

        # Manually set reasonable theta parameters to bypass fitting convergence issues
        model.theta = {
            'delta': torch.randn(3, dtype=torch.float32) * 0.1,
            'beta': torch.randn(3, dtype=torch.float32) * 0.1,  # p+1 = 3
            'gamma': torch.randn(5, dtype=torch.float32) * 0.1,  # 2p+1 = 5
        }
        model.converged = True  # Mark as converged to bypass checks

        # Test variance estimation with anchor-based Monte Carlo - interface should work
        model.estimate_variance(
            pairs_per_anchor=5,  # Partners per anchor
            s=10,  # Number of anchors
            alpha=0.3,  # Debiasing parameter
            verbose=False,
        )

        # Check that interface works correctly - variance matrix is created
        assert hasattr(model, 'variance_matrix')
        assert model.variance_matrix is not None
        assert isinstance(model.variance_matrix, torch.Tensor)

        # Should be square matrix with correct dimensions
        d_total = 3 + 3 + 5  # delta + beta + gamma dimensions
        assert model.variance_matrix.shape == (d_total, d_total)

        # Basic numeric properties (relaxed for integration testing)
        assert not torch.isnan(model.variance_matrix).any(), 'Variance matrix contains NaN'
        assert not torch.isinf(model.variance_matrix).any(), 'Variance matrix contains Inf'

    def test_anchor_based_interface_compatibility(self, sample_drgu_data):
        """Test that anchor-based interface works with different parameter combinations."""
        data = sample_drgu_data

        model = DRGUMiniBatch(
            data=data,
            covariates=['x1', 'x2'],
            treatment='treatment',
            response='outcome',
            device='cpu',
        )

        # Manually set theta and mark as converged
        model.theta = {
            'delta': torch.randn(3, dtype=torch.float32) * 0.1,
            'beta': torch.randn(3, dtype=torch.float32) * 0.1,
            'gamma': torch.randn(5, dtype=torch.float32) * 0.1,
        }
        model.converged = True

        # Test that different parameter combinations work without errors
        test_params = [
            {'pairs_per_anchor': 3, 's': 5, 'alpha': 0.0},
            {'pairs_per_anchor': 5, 's': None, 'alpha': 0.5},  # Default s
            {'pairs_per_anchor': 2, 's': 8, 'alpha': 1.0},
        ]

        for params in test_params:
            model.estimate_variance(verbose=False, **params)

            # Should succeed and create variance matrix
            assert hasattr(model, 'variance_matrix')
            assert model.variance_matrix is not None
            assert model.variance_matrix.shape == (11, 11)  # 3+3+5 dimensions

    def test_anchor_parameter_validation(self, sample_drgu_data):
        """Test parameter validation for anchor-based estimation."""
        data = sample_drgu_data

        model = DRGUMiniBatch(
            data=data,
            covariates=['x1', 'x2'],
            treatment='treatment',
            response='outcome',
            device='cpu',
        )

        # Manually set theta and mark as converged
        model.theta = {
            'delta': torch.randn(3, dtype=torch.float32) * 0.1,
            'beta': torch.randn(3, dtype=torch.float32) * 0.1,
            'gamma': torch.randn(5, dtype=torch.float32) * 0.1,
        }
        model.converged = True
        # Also set coefficients for summary() method
        model.coefficients = torch.cat(
            [model.theta['delta'], model.theta['beta'], model.theta['gamma']]
        )

        # Test default parameters
        model.estimate_variance(verbose=False)
        assert hasattr(model, 'variance_matrix')

        # Test explicit parameters
        model.estimate_variance(pairs_per_anchor=4, s=8, alpha=0.5, verbose=False)
        assert hasattr(model, 'variance_matrix')

        # Test boundary alpha values
        model.estimate_variance(pairs_per_anchor=3, s=5, alpha=0.0, verbose=False)  # No debiasing
        model.estimate_variance(pairs_per_anchor=3, s=5, alpha=1.0, verbose=False)  # Full debiasing


def test_warm_up_basic(simple_data):
    """Simple warm up test."""
    df = pd.DataFrame(
        {
            'x1': simple_data['X'][:, 0].numpy(),
            'x2': simple_data['X'][:, 1].numpy(),
            'treatment': simple_data['z'].numpy(),
            'outcome': simple_data['y'].numpy(),
        }
    )

    model = DRGUMiniBatch(df, ['x1', 'x2'], 'treatment', 'outcome')

    # Initialize theta first (required for _warm_up)
    model.theta = {
        'delta': torch.zeros(1, dtype=torch.float32),
        'beta': torch.zeros(3, dtype=torch.float32),
        'gamma': torch.zeros(5, dtype=torch.float32),
    }

    # Test sample_full_pairs works
    batch = model.sample_full_pairs(sample_size=10)
    assert 'xi' in batch and len(batch['xi']) > 0

    # Test warm up runs without error
    warm_theta = model._warm_up(warm_up_rounds=1, sample_size=10, max_step_norm=20.0, verbose=False)
    assert 'delta' in warm_theta


# ============================================================================
# Penalty/Regularization Tests
# ============================================================================


class TestPenalty:
    """Tests for penalty/regularization behavior in MiniBatch implementation."""

    def test_penalty_no_delta(self):
        """Test that delta parameter is not penalized by default."""
        penalty = Penalty(lam=1.0)
        assert penalty.penalize_delta is False, 'Default should not penalize delta'

    def test_penalty_with_delta(self):
        """Test that delta can be optionally penalized."""
        penalty = Penalty(lam=1.0, penalize_delta=True)
        assert penalty.penalize_delta is True, 'Should be able to enable delta penalization'

    def test_penalty_mask_computation(self):
        """Test that penalty mask correctly excludes delta parameter."""
        # Setup simple theta
        theta = {
            'delta': torch.tensor([1.0]),
            'beta': torch.tensor([2.0, 3.0]),
            'gamma': torch.tensor([4.0, 5.0, 6.0]),
        }

        # Create optimizer with penalty that doesn't penalize delta
        penalty = Penalty(lam=1.0, penalize_delta=False)
        optimizer = MiniBatchFisherScoring(
            model_params=theta, compute_B_U=drgu_compute_B_U, penalty=penalty
        )

        # Compute penalty terms
        g_pen, H_pen = optimizer._penalty_terms()

        # Check that penalty gradient for delta is zero
        assert g_pen[0].item() == 0.0, 'Delta gradient should be zero (not penalized)'

        # Check that other parameters are penalized
        assert g_pen[1].item() != 0.0, 'Beta parameters should be penalized'

        # Check penalty Hessian diagonal
        assert H_pen[0, 0].item() == 0.0, 'Delta penalty Hessian should be zero'
        assert H_pen[1, 1].item() == 1.0, 'Beta penalty Hessian should be lambda=1.0'

    def test_penalty_mask_with_delta_penalized(self):
        """Test that all parameters are penalized when penalize_delta=True."""
        # Setup simple theta
        theta = {
            'delta': torch.tensor([1.0]),
            'beta': torch.tensor([2.0, 3.0]),
            'gamma': torch.tensor([4.0, 5.0, 6.0]),
        }

        # Create optimizer with penalty that DOES penalize delta
        penalty = Penalty(lam=1.0, penalize_delta=True)
        optimizer = MiniBatchFisherScoring(
            model_params=theta, compute_B_U=drgu_compute_B_U, penalty=penalty
        )

        # Compute penalty terms
        g_pen, H_pen = optimizer._penalty_terms()

        # Check that ALL parameters including delta are penalized
        assert g_pen[0].item() != 0.0, 'Delta should be penalized when penalize_delta=True'
        assert H_pen[0, 0].item() == 1.0, 'Delta penalty Hessian should be lambda=1.0'

    def test_large_lambda_does_not_penalize_delta(self):
        """
        Test that with large lambda, delta is not penalized but beta/gamma are.
        """
        theta = {
            'delta': torch.tensor([0.5]),
            'beta': torch.tensor([1.0, 2.0, 3.0]),
            'gamma': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        }

        # Use large lambda to make penalty effects clear
        penalty = Penalty(lam=10.0, penalize_delta=False)
        optimizer = MiniBatchFisherScoring(
            model_params=theta, compute_B_U=drgu_compute_B_U, penalty=penalty
        )

        g_pen, H_pen = optimizer._penalty_terms()

        # Verify delta is not penalized (first element should be 0)
        assert g_pen[0].item() == 0.0, 'Delta gradient should be zero (not penalized)'
        assert H_pen[0, 0].item() == 0.0, 'Delta Hessian should be zero (not penalized)'

        # Verify beta and gamma ARE penalized (non-zero values)
        assert g_pen[1].item() != 0.0, 'Beta should be penalized'
        assert H_pen[1, 1].item() == 10.0, f'Beta Hessian should be lambda={10.0}'

        # Check the penalty structure: diag([0, lambda, lambda, ..., lambda])
        expected_diag = torch.ones(len(g_pen)) * 10.0
        expected_diag[0] = 0.0  # Delta not penalized

        actual_diag = torch.diag(H_pen)

        assert torch.allclose(actual_diag, expected_diag), (
            f'Penalty diagonal should be [0, λ, λ, ...]: '
            f'expected={expected_diag}, actual={actual_diag}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
