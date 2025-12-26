"""
DRGU Mini-Batch Model Implementation.

This module contains the DRGUMiniBatch class, which provides a mini-batch implementation
of the DRGU model using the EstimatingEquation interface.
"""

from __future__ import annotations

import pandas as pd
import torch
from scipy.stats import norm

from ..drgu import data_pairwise
from ..ee import EstimatingEquation
from ..io.pair_dataset import PairBatchIterableDataset
from ..io.samplers import KPartnersSampler
from .estimating_equations import drgu_compute_B_U
from .minibatch_fisher import MiniBatchFisherScoring, Penalty


class DRGUMiniBatch(EstimatingEquation):
    """
    Mini-batch DRGU implementation with EstimatingEquation interface.

    This class provides the same API as DRGUTorch but uses mini-batch Fisher scoring
    for scalable computation on large datasets.
    """

    def __init__(self, data, covariates, treatment, response, device='cpu'):
        """
        Initialize the mini-batch DRGU model.

        :param data: pandas.DataFrame, the dataset
        :param covariates: list, names of covariate columns
        :param treatment: str, name of the treatment variable
        :param response: str, name of the response variable
        :param device: str, device to use for PyTorch tensors ('cpu' or 'cuda')
        """
        super().__init__(data, covariates, treatment, response)
        self.device = device

        # Convert data to PyTorch tensors
        self.X = torch.tensor(self.data[self.covariates].values, dtype=torch.float32, device=device)
        self.z = torch.tensor(self.data[self.treatment].values, dtype=torch.float32, device=device)
        self.y = torch.tensor(self.data[self.response].values, dtype=torch.float32, device=device)

        # Store dimensions
        self.n, self.p = self.X.shape

        # Initialize parameters (will be set after fitting or via set_theta)
        self.theta = None
        self._theta_initialized = False
        self.final_optimizer = None
        self.variance_matrix = None  # Will be set after variance estimation
        self.converged = None  # Will be set after fitting

    def set_theta(self, theta_dict):
        """
        Set theta parameters explicitly. Useful for testing and custom initialization.

        Args:
            theta_dict: Dictionary with 'delta', 'beta', 'gamma' parameters
        """
        # Validate input
        required_keys = {'delta', 'beta', 'gamma'}
        if not all(key in theta_dict for key in required_keys):
            raise ValueError(f'theta_dict must contain keys: {required_keys}')

        # Initialize theta with proper device and dtype
        self.theta = {
            'delta': theta_dict['delta'].to(device=self.device, dtype=torch.float32).clone(),
            'beta': theta_dict['beta'].to(device=self.device, dtype=torch.float32).clone(),
            'gamma': theta_dict['gamma'].to(device=self.device, dtype=torch.float32).clone(),
        }
        self._theta_initialized = True

    def _run_batch_epoch(self, optimizer, dataset, batches_per_epoch, tol, verbose):
        """Run one epoch with batch-level updates."""
        epoch_converged = False
        delta_norm = float('inf')
        total_steps = 0
        iter_dataset = iter(dataset)

        for step in range(batches_per_epoch):
            try:
                batch = next(iter_dataset)
                info = optimizer.step(batch)
                total_steps += 1

                # Extract convergence metrics
                delta_norm = info.get('delta_norm', 0.0)
                u_norm = info.get('U_norm', 0.0)

                # Print step info (every 5 steps or on convergence)
                if verbose and (step % 5 == 0 or step == 0 or delta_norm < tol):
                    print(f'  Step {step + 1}: delta={delta_norm:.3e}, u={u_norm:.3e}')

                # Check convergence
                if delta_norm < tol:
                    epoch_converged = True
                    if verbose:
                        print(f'  CONVERGED at step {step + 1}')
                    break

            except (StopIteration, RuntimeError):
                if verbose and step == 0:
                    print('  Warning: data iteration failed')
                break

        return epoch_converged, delta_norm, total_steps

    def _run_epoch_epoch(self, optimizer, dataset, batches_per_epoch, tol, verbose):
        """Run one epoch with epoch-level updates."""
        epoch_converged = False
        delta_norm = float('inf')
        total_steps = 0
        iter_dataset = iter(dataset)

        # Accumulate B,U across batches
        for step in range(batches_per_epoch):
            try:
                batch = next(iter_dataset)
                optimizer.step(batch)  # This accumulates B,U
                total_steps += 1

            except (StopIteration, RuntimeError):
                if verbose and step == 0:
                    print('  Warning: data iteration failed')
                break

        # Apply epoch-level Fisher step
        if optimizer.batch_count > 0:  # Only if we accumulated some data
            epoch_info = optimizer.epoch_step()
            delta_norm = epoch_info.get('delta_norm', 0.0)
            u_norm = epoch_info.get('U_norm', 0.0)

            if verbose:
                print(f'  Epoch step: delta={delta_norm:.3e}, u={u_norm:.3e}')

            # Check convergence on epoch step
            if delta_norm < tol:
                epoch_converged = True
                if verbose:
                    print('  CONVERGED after epoch step')
        else:
            delta_norm = float('inf')

        return epoch_converged, delta_norm, total_steps

    def _finalize_training(self, optimizer, converged, total_steps, delta_norm, verbose):
        """Finalize training and prepare results."""
        # Store results
        self.theta = optimizer.theta
        self.final_optimizer = optimizer

        # Adjust delta (matching original DRGU implementation)
        self.theta['delta'] = self.theta['delta'] + 0.5

        # Set coefficients
        self.coefficients = torch.cat([v.flatten() for v in self.theta.values()])

        # Final convergence summary
        if verbose:
            status = 'CONVERGED' if converged else 'NOT CONVERGED'
            print(f'{status} after {total_steps} steps (final delta={delta_norm:.3e})')
            if hasattr(self, 'final_optimizer') and self.final_optimizer:
                u_final = self.final_optimizer.get_U_running_avg()
                if u_final is not None:
                    print(f'  Final u_norm: {u_final:.3e}')

        # Store convergence status for variance estimation
        self.converged = converged

        return {
            'delta': self.theta['delta'],
            'beta': self.theta['beta'],
            'gamma': self.theta['gamma'],
            'converged': converged,
            'iterations': total_steps,
            'final_delta_norm': delta_norm,
        }

    def fit(
        self,
        tol=1e-6,
        lamb=0.0,
        pairs_per_anchor=20,
        pairs_per_batch=2000,
        max_epochs=10,
        batches_per_epoch=-1,
        learning_rate=1.0,
        momentum=0.0,
        adaptive_lr_bool=False,
        fisher_ema=0.0,
        max_step_norm=5.0,
        update_mode='batch',
        theta_averaging=False,
        warm_up=False,
        option='plain',
        verbose=False,
    ):
        """
        Fit the simplified mini-batch DRGU model to the data.

        Args:
            tol: Convergence tolerance
            lamb: Regularization parameter (L2 penalty)
            option: Optimization method (only 'plain' supported for now)
            verbose: Whether to print convergence information
            pairs_per_anchor: Number of partners to sample for each anchor
            pairs_per_batch: Number of pairs per mini-batch
            max_epochs: Maximum number of epochs
            batches_per_epoch: Maximum batches to process per epoch
            learning_rate: Base learning rate scaling factor (1.0 = no scaling)
            momentum: Momentum coefficient for gradient updates (0.0 = no momentum)
            adaptive_lr_bool: Enable adaptive learning rate based on condition number
            fisher_ema: EMA coefficient for Fisher matrix smoothing (0.0 = no EMA)
            theta_averaging: Enable theta averaging across batches within epochs
            warm_up: Whether to run warm up phase for better initialization (default True)
        """
        # Setup and validation
        if option != 'plain':
            raise ValueError(f'Unsupported option: {option}')

        if batches_per_epoch == -1:
            batches_per_epoch = max(1, self.n * pairs_per_anchor // pairs_per_batch)

        if verbose:
            print(
                f'DRGU: n={self.n}, epochs={max_epochs}, tol={tol:.1e}, '
                f'batches_per_epoch={batches_per_epoch}'
            )

        # Initialize parameters (use existing if already set)
        if not self._theta_initialized:
            theta = {
                'delta': torch.zeros(1, dtype=torch.float32, device=self.device),
                'beta': torch.zeros(self.p + 1, dtype=torch.float32, device=self.device),
                'gamma': torch.zeros(2 * self.p + 1, dtype=torch.float32, device=self.device),
            }
            self.theta = theta
            self._theta_initialized = True
        else:
            theta = self.theta

        # Warm up the model (if enabled)
        if warm_up:
            theta = self._warm_up(
                tol=1e-5,
                lamb=lamb,
                max_step_norm=max_step_norm,
                sample_size=500,
                max_steps=20,
                verbose=verbose,
            )

        # Setup sampler and optimizer
        sampler = KPartnersSampler(n=self.n, k=pairs_per_anchor, seed=42)

        optimizer = MiniBatchFisherScoring(
            model_params=theta,
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=lamb),
            max_step_norm=max_step_norm,
            learning_rate=learning_rate,
            momentum=momentum,
            adaptive_lr_bool=adaptive_lr_bool,
            fisher_ema=fisher_ema,
            update_mode=update_mode,
            theta_averaging=theta_averaging,
        )

        # Main training loop
        converged = False
        total_steps = 0
        delta_norm = float('inf')
        epochs_run = 0

        for epoch in range(max_epochs):
            if verbose:
                print(f'Epoch {epoch + 1}/{max_epochs}')

            # Create dataset for this epoch
            dataset = PairBatchIterableDataset(
                X=self.X,
                y=self.y,
                z=self.z,
                sampler=sampler,
                pairs_per_batch=pairs_per_batch,
                epoch=epoch,
                infinite=True,
            )

            # Reset U averaging for this epoch
            optimizer.reset_U_averaging()

            # Run epoch with appropriate update mode
            if update_mode == 'batch':
                epoch_converged, delta_norm, epoch_steps = self._run_batch_epoch(
                    optimizer, dataset, batches_per_epoch, tol, verbose
                )
            elif update_mode == 'epoch':
                epoch_converged, delta_norm, epoch_steps = self._run_epoch_epoch(
                    optimizer, dataset, batches_per_epoch, tol, verbose
                )
            else:
                raise ValueError(f'Unknown update_mode: {update_mode}')

            total_steps += epoch_steps
            epochs_run = epoch + 1

            # Handle theta averaging
            if theta_averaging:
                averaged_applied = optimizer.apply_averaged_theta()
                if verbose and averaged_applied:
                    print('  Applied theta averaging')
                optimizer.reset_theta_averaging()

            # Check for convergence
            if epoch_converged:
                converged = True
                break

        # Finalize training and return results
        result = self._finalize_training(optimizer, converged, total_steps, delta_norm, verbose)
        result['epochs_run'] = epochs_run
        return result

    def _warm_up(
        self,
        tol=1e-6,
        lamb=0.0,
        warm_up_rounds=3,
        sample_size=500,
        max_step_norm=5.0,
        max_steps=20,
        verbose=False,
    ):
        """
        Warm up the model by fitting on full pairs of sampled data multiple times,
        and return average theta.
        """
        if verbose:
            print(
                f'Warm up: rounds={warm_up_rounds}, sample_size={sample_size}, '
                f'max_step_norm={max_step_norm}'
            )

        theta_sum = {k: torch.zeros_like(v) for k, v in self.theta.items()}
        theta_count = 0

        for i in range(warm_up_rounds):
            if verbose:
                print(f'Warm up round {i + 1}/{warm_up_rounds}')
            result = self.fit_on_full_pairs(
                tol=tol,
                lamb=lamb,
                sample_size=sample_size,
                max_step_norm=max_step_norm,
                max_steps=max_steps,
                verbose=verbose,
            )
            if result['converged']:
                for k in theta_sum:
                    theta_sum[k] = theta_sum[k] + result['theta'][k]
                theta_count += 1

        if theta_count == 0:
            if verbose:
                print('Warning: No warm up rounds converged, using initial theta')
            return {k: v.clone() for k, v in self.theta.items()}

        theta_avg = {k: theta_sum[k] / theta_count for k in theta_sum}
        if verbose:
            print(f'Warm up complete: {theta_count}/{warm_up_rounds} rounds converged')
        return theta_avg

    def fit_on_full_pairs(
        self, tol=1e-6, lamb=0.0, sample_size=500, max_step_norm=5.0, max_steps=20, verbose=False
    ):
        """
        Fit the model on full pairs of sampled data.
        """
        if verbose:
            print(
                f'Fitting on full pairs: sample_size={sample_size}, '
                f'max_step_norm={max_step_norm}, max_steps={max_steps}'
            )

        # Sample full pairs
        full_pairs = self.sample_full_pairs(sample_size)

        # Create optimizer with current theta as starting point
        optimizer = MiniBatchFisherScoring(
            model_params={
                k: v.clone() for k, v in self.theta.items()
            },  # Clone to avoid modifying original
            compute_B_U=drgu_compute_B_U,
            penalty=Penalty(lam=lamb),
            max_step_norm=max_step_norm,
        )

        # Run optimizer until convergence or max steps
        converged = False
        delta_norm = float('inf')
        steps = 0

        while steps < max_steps and not converged:
            # Run optimization step
            result = optimizer.step(full_pairs)
            steps += 1
            delta_norm = result['delta_norm']  # Get from step result
            converged = delta_norm < tol

            if verbose:
                print(f'  Step {steps}: delta={delta_norm:.3e}')

            if converged:
                break

        return {
            'theta': {k: v.clone() for k, v in optimizer.theta.items()},  # Proper deep copy
            'converged': converged,
            'steps': steps,
            'delta_norm': delta_norm,
        }

    def sample_full_pairs(self, sample_size=500):
        """Sample full pairs from the dataset using efficient DRGU pairwise logic."""

        # Limit sample size to available data
        n_sample = min(sample_size, len(self.X))

        # Randomly sample indices if needed
        if n_sample < len(self.X):
            indices = torch.randperm(len(self.X))[:n_sample]
            X_sample = self.X[indices]
            y_sample = self.y[indices]
            z_sample = self.z[indices] if self.z is not None else None
        else:
            # Use full data
            X_sample = self.X
            y_sample = self.y
            z_sample = self.z

        # Use existing efficient pairwise data creation (creates all i < j pairs)
        drgu_pairs = data_pairwise(y_sample, z_sample, X_sample)

        # Convert DRGU format to minibatch format expected by optimizer
        # DRGU format has 'wi', 'wj', 'yi', 'yj', 'zi', 'zj'
        # Minibatch format needs 'xi', 'xj', 'yi', 'yj', 'zi', 'zj', 'w_ij'
        batch = {
            'xi': drgu_pairs['wi'],  # [m, p] features for i-th elements
            'xj': drgu_pairs['wj'],  # [m, p] features for j-th elements
            'yi': drgu_pairs['yi'],  # [m] outcomes for i-th elements
            'yj': drgu_pairs['yj'],  # [m] outcomes for j-th elements
            'zi': drgu_pairs['zi'],  # [m] treatments for i-th elements
            'zj': drgu_pairs['zj'],  # [m] treatments for j-th elements
            'w_ij': torch.ones_like(drgu_pairs['yi']),  # [m] equal weights
        }

        return batch

    def estimate_variance(
        self,
        pairs_per_anchor: int = 20,
        s: int | None = None,
        alpha: float = 0.0,
        anchors_per_batch: int = 1,
        verbose: bool = True,
    ):
        """
        Estimate variance matrix using anchor-based Monte Carlo integration.

        Must be called after fit() to compute confidence intervals and hypothesis tests.

        Args:
            pairs_per_anchor: Number of partners per anchor (m in Algorithm 1)
            s: Number of anchors (defaults to n if None)
            alpha: De-bias parameter in [0,1] for anchor-based estimation
            anchors_per_batch: Number of anchors to process per batch (for efficiency)
            verbose: Print variance computation details
        """
        # Check that model has been fitted
        if not hasattr(self, 'theta') or self.theta is None:
            raise RuntimeError('Must call fit() before estimating variance')

        if not hasattr(self, 'converged'):
            raise RuntimeError('Convergence status not available. Call fit() first.')

        if not self.converged:
            raise RuntimeError(
                'Model did not converge. Cannot estimate reliable variance matrix for '
                'non-converged model.'
            )

        if verbose:
            s_display = s if s is not None else self.n // 10
            print(f'Variance estimation: s={s_display}, k={pairs_per_anchor}')

        # Compute variance matrix using anchor-based Monte Carlo integration
        from .montecarlo_estimation import MonteCarloEstimation

        # Use anchor-based Monte Carlo estimation
        mc_estimator = MonteCarloEstimation(device=self.device)
        raw_variance = mc_estimator.estimate(
            X=self.X,
            y=self.y,
            z=self.z,
            theta=self.theta,
            k=pairs_per_anchor,
            s=s,
            alpha=alpha,
            anchors_per_batch=anchors_per_batch,
            verbose=verbose,
        )

        # Monte Carlo returns 4 * B_inv @ Sigma @ B_inv.T
        # Apply consistent scaling: Var = raw, variance_matrix = Var / n
        self.Var = raw_variance  # Raw variance: 4 * B_inv @ Sigma @ B_inv.T
        self.variance_matrix = self.Var / self.n  # Final variance: (4/n) * B_inv @ Sigma @ B_inv.T

        if verbose:
            print('Variance matrix computed')

    def summary(self):
        """
        Generate a summary of the model fit, including coefficients, standard errors,
        z-scores, and p-values.

        Returns:
            pandas.DataFrame with model summary
        """
        if self.coefficients is None:
            raise ValueError('Model has not been fitted yet.')

        if self.variance_matrix is None:
            raise RuntimeError(
                'Variance matrix has not been estimated yet. '
                'Call estimate_variance() before generating summary.'
            )

        # Compute standard errors from variance matrix
        standard_errors = torch.sqrt(torch.diag(self.variance_matrix))

        # Compute z-scores
        null_hypothesis = torch.zeros_like(self.coefficients)
        null_hypothesis[0] = 0.5  # delta null hypothesis
        z_scores = (self.coefficients - null_hypothesis) / standard_errors

        # Compute p-values
        p_values = 2 * (1 - norm.cdf(torch.abs(z_scores).cpu().numpy()))

        # Create a summary table
        row_names = (
            ['delta']
            + [f'beta_{i}' for i in range(len(self.theta['beta']))]
            + [f'gamma_{i}' for i in range(len(self.theta['gamma']))]
        )

        summary = pd.DataFrame(
            {
                'Names': row_names,
                'Coefficient': self.coefficients.cpu().numpy(),
                'Null_Hypothesis': null_hypothesis.cpu().numpy(),
                'Std_Error': standard_errors.cpu().numpy(),
                'Z_Score': z_scores.cpu().numpy(),
                'P_Value': p_values,
            }
        )

        return summary
