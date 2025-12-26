"""
Anchor-based Monte Carlo estimation for DRGU.

This module provides anchor-based Monte Carlo estimation capabilities for computing
variance using the algorithm specified in the paper.
"""

import torch

from ..io.pair_dataset import PairBatchIterableDataset
from ..io.samplers import KPartnersSampler


class MonteCarloEstimation:
    """
    Anchor-based Monte Carlo estimation for sandwich variance computation.

    Implements Algorithm 1 for anchor-based Monte Carlo variance estimation
    using anchor sampling and within-anchor covariance debiasing.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize anchor-based Monte Carlo estimation.

        Args:
            device: PyTorch device for computations
        """
        self.device = device or torch.device('cpu')

    def estimate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        theta: dict[str, torch.Tensor],
        k: int = 10,
        s: int | None = None,
        alpha: float = 0.0,
        anchors_per_batch: int = 1,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Anchor-based Monte Carlo estimation following Algorithm 1.

        Args:
            X: Covariate matrix [n, p]
            y: Response vector [n]
            z: Treatment vector [n]
            theta: Parameter estimates {'delta', 'beta', 'gamma'}
            k: Number of partners per anchor (m in the algorithm)
            s: Number of anchors (defaults to n if None)
            alpha: De-bias parameter in [0,1]
            anchors_per_batch: Number of anchors to process per batch (for efficiency)
            verbose: Print progress information

        Returns:
            Variance matrix: Var(theta_hat) = (4/n)B_hat^(-1)Sigma_hat B_hat^(-1)^T
        """
        return self._estimate_anchor_based(X, y, z, theta, k, s, alpha, anchors_per_batch, verbose)

    def _estimate_anchor_based(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        theta: dict[str, torch.Tensor],
        k: int,  # m = number of partners per anchor
        s: int | None,
        alpha: float,
        anchors_per_batch: int,
        verbose: bool,
    ) -> torch.Tensor:
        """
        Anchor-based Monte Carlo estimation following Algorithm 1.

        Args:
            X: Covariate matrix [n, p]
            y: Response vector [n]
            z: Treatment vector [n]
            theta: Parameter estimates {'delta', 'beta', 'gamma'}
            k: Number of partners per anchor (m in the algorithm)
            s: Number of anchors (defaults to n if None)
            alpha: De-bias parameter in [0,1]
            verbose: Print progress information

        Returns:
            Variance matrix: Var(theta_hat) = (4/n)B_hat^(-1)Sigma_hat B_hat^(-1)^T
        """
        from ..drgu import _compute_B_u_ij
        from ..minibatch.estimating_equations import convert_minibatch_to_drgu_format

        n = len(y)
        m = k  # Partners per anchor
        s = s if s is not None else n

        # Assert algorithm requirements
        assert m > 1, (
            f'Need m > 1 partners per anchor for meaningful within-anchor covariance, got m={m}'
        )
        assert s >= 1, f'Need s >= 1 anchors, got s={s}'

        if verbose:
            print(f'MC estimation: n={n}, s={s}, m={m}')

        # Create anchor-based sampler and dataset
        sampler = KPartnersSampler(n=n, k=m, anchor_based=True, s=s, seed=42)
        dataset = PairBatchIterableDataset(
            X=X, y=y, z=z, sampler=sampler, device=self.device, anchors_per_batch=anchors_per_batch
        )

        # Initialize accumulators for running averages
        B_total = None  # Running sum of B matrices
        Sigma_between_running = None  # Running sum of u_bar_i u_bar_i^T
        Sigma_within_running = None  # Running sum of Sigma_hat_i^within

        # Use identity V_inv - standard for DRGU
        V_inv = torch.eye(3, dtype=theta['delta'].dtype, device=theta['delta'].device)

        # Compute parameter dimension once (constant across all batches)
        d_total = len(theta['delta']) + len(theta['beta']) + len(theta['gamma'])

        # Process anchor batches from dataset (dataset handles anchor batching)
        batch_count = 0
        for batch in dataset:
            # Convert mini-batch format to DRGU format
            drgu_data = convert_minibatch_to_drgu_format(batch)

            # Ensure data types match theta
            target_dtype = theta['delta'].dtype
            target_device = theta['delta'].device
            drgu_data = {
                k: v.to(dtype=target_dtype, device=target_device)
                for k, v in drgu_data.items()
                if isinstance(v, torch.Tensor)
            }

            # Efficient: Compute B_ij and u_ij for ALL pairs in this batch at once
            B_batch, u_ij_batch = _compute_B_u_ij(theta, V_inv, drgu_data)

            # B matrix is consistent across pairs for given theta
            # _compute_B_u_ij already returns averaged B matrix [d_total, d_total]

            # Accumulate B matrices: B_hat = (1/batch_count)sum_batches B_batch
            if B_total is None:
                B_total = B_batch.clone()
            else:
                B_total += B_batch

            # Per-anchor statistics using efficient tensor reshaping and batch operations
            anchor_ids = batch.get('anchor_id', None)
            if anchor_ids is not None:
                # Since data is contiguous for anchors, we can reshape efficiently
                total_pairs = u_ij_batch.shape[0]

                # We know pairs_per_anchor = m = k from the algorithm
                num_anchors = total_pairs // m

                # Reshape to [num_anchors, m, d_total] for batch operations
                u_ij_reshaped = u_ij_batch.view(num_anchors, m, d_total)

                # Compute anchor means efficiently: u_bar_i = mean over pairs dimension
                u_bar_anchors = torch.mean(u_ij_reshaped, dim=1)  # [num_anchors, d_total]

                # Compute within-anchor covariances using batch matrix multiplication
                # Center the data: u_ij - u_bar_i
                u_centered = u_ij_reshaped - u_bar_anchors.unsqueeze(1)  # [num_anchors, m, d_total]

                # Batch covariance computation: u_centered^T @ u_centered for each anchor
                # u_centered.transpose(-1, -2): [num_anchors, d_total, m]
                # torch.bmm result: [num_anchors, d_total, d_total]
                # We know m > 1 from assertion, so no conditional needed
                Sigma_within_anchors = torch.bmm(u_centered.transpose(-1, -2), u_centered) / (
                    m - 1
                )  # [num_anchors, d_total, d_total]

                # Compute Sigma_between for all anchors using batch outer products
                # u_bar_anchors: [num_anchors, d_total] -> [num_anchors, d_total, d_total]
                u_bar_outer_batch = torch.bmm(
                    u_bar_anchors.unsqueeze(-1), u_bar_anchors.unsqueeze(-2)
                )  # [num_anchors, d_total, d_total]

                # Sum over anchors to accumulate statistics (no loop needed!)
                Sigma_between_batch = u_bar_outer_batch.sum(dim=0)  # [d_total, d_total]
                Sigma_within_batch_sum = Sigma_within_anchors.sum(dim=0)  # [d_total, d_total]

                # Accumulate across multiple batches
                if Sigma_between_running is None:
                    Sigma_between_running = Sigma_between_batch.clone()
                    Sigma_within_running = Sigma_within_batch_sum.clone()
                else:
                    Sigma_between_running += Sigma_between_batch
                    Sigma_within_running += Sigma_within_batch_sum

            batch_count += 1
            if verbose and batch_count % max(1, (s // anchors_per_batch) // 5) == 0:
                print(f'  Batch {batch_count}')

                # Step 8: Estimate Jacobian B_hat = (1/s)sum_i B_i
        if B_total is None:
            raise RuntimeError(
                f'No batches were processed - B_total is None. '
                f'Check sampling parameters: n={n}, s={s}, m={m}'
            )

        B_hat = B_total / batch_count  # Use actual batch count instead of s

        # Step 9: Estimate variance component with debiasing
        # Sigma_hat = (1/s)sum u_bar_i u_bar_i^T - alpha((1/m)-(1/(n-1)))(1/s)sum Sigma_hat_i^within

        # Convert running sums to averages
        Sigma_between = Sigma_between_running / s  # (1/s)sum u_bar_i u_bar_i^T
        Sigma_within_avg = Sigma_within_running / s  # (1/s)sum Sigma_hat_i^within

        # Debiasing factor
        debias_factor = alpha * ((1.0 / m) - (1.0 / (n - 1)))

        # Final variance estimate
        Sigma_hat = Sigma_between - debias_factor * Sigma_within_avg

        # Ensure PSD when alpha > 0 by clipping negative eigenvalues
        if alpha > 0:
            eigenvals, eigenvecs = torch.linalg.eigh(Sigma_hat)
            eigenvals_clipped = torch.clamp(eigenvals, min=0.0)
            Sigma_hat = eigenvecs @ torch.diag(eigenvals_clipped) @ eigenvecs.T

            if verbose:
                neg_eigenvals = (eigenvals < 0).sum().item()
                if neg_eigenvals > 0:
                    print(f'  Clipped {neg_eigenvals} negative eigenvalues')

        # Step 10: Output Var(theta_hat) = (4/n)B_hat^(-1)Sigma_hat B_hat^(-1)^T
        try:
            B_inv = torch.linalg.inv(B_hat)
        except torch.linalg.LinAlgError:
            if verbose:
                print('  Warning: using pseudo-inverse')
            B_inv = torch.linalg.pinv(B_hat)

        # Return variance matrix with factor of 4 (caller applies 1/n scaling)
        variance_matrix = 4.0 * B_inv @ Sigma_hat @ B_inv.T

        if verbose:
            print('Anchor-based variance computed')

        return variance_matrix
