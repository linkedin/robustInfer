from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

# Type aliases for clarity
Tensor = torch.Tensor
Theta = dict[str, Tensor]
ComputeBU = Callable[[Theta, dict[str, Tensor]], tuple[Tensor, Tensor]]


@dataclass
class Penalty:
    """
    Penalty/regularization configuration.

    Supports L2 penalty with optional centering:
    penalty = lambda/2 * ||theta - theta_0||^2 where theta_0 is the center (default: 0)

    Note: For DRGU, we typically don't penalize delta (the parameter of interest for hypothesis
    testing). Set penalize_delta=False to match the Scala implementation.
    """

    lam: float
    center: dict[str, Tensor] | None = None  # shrink-to theta_0 (optional)
    penalize_delta: bool = False  # whether to penalize delta parameter (default: False)


# ---------- Utility functions ----------


def _I(d: int, like: Tensor) -> Tensor:
    """Create identity matrix with same dtype/device as reference tensor."""
    return torch.eye(d, dtype=like.dtype, device=like.device)


def fisher_solve(J: Tensor, rhs: Tensor) -> Tensor:
    """
    Robust solver for Fisher scoring linear system.

    Solves J * delta = rhs with multiple fallbacks for numerical stability:
    1. Cholesky decomposition (fastest for PD matrices)
    2. Least squares (for singular/ill-conditioned)
    3. Pseudo-inverse (final fallback)

    Note: Use penalty/regularization in the calling code rather than damping here.

    Args:
        J: Fisher information matrix [d, d] (should already include any regularization)
        rhs: Right-hand side vector [d]

    Returns:
        Solution vector delta [d]
    """
    # Use the matrix as provided (regularization should be applied upstream)
    Js = J

    try:
        # Try Cholesky (fastest for PD matrices)
        L = torch.linalg.cholesky(Js)
        return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    except Exception:
        try:
            # Try least squares
            return torch.linalg.lstsq(J, rhs.unsqueeze(-1)).solution.squeeze(-1)
        except Exception:
            # Final fallback: pseudo-inverse
            return torch.linalg.pinv(J) @ rhs


# Consider add trust region methods in future


# ---------- Mini-batch Fisher scoring ----------


class MiniBatchFisherScoring:
    """
    Mini-batch Fisher scoring.

    TODO: Add advanced methods (trust_region, dogleg, line_search)
    """

    def __init__(
        self,
        model_params: Theta,
        compute_B_U: ComputeBU,
        penalty: Penalty | None = None,
        max_step_norm: float = 10.0,  # reduced max step norm for stability
        learning_rate: float = 1.0,  # base learning rate (1.0 = no scaling)
        momentum: float = 0.0,  # momentum coefficient (0.0 = no momentum)
        adaptive_lr_bool: bool = False,  # enable adaptive learning rate
        fisher_ema: float = 0.0,  # EMA coefficient for Fisher matrix (0.0 = no EMA)
        update_mode: str = 'batch',  # 'batch' or 'epoch' - when to apply Fisher updates
        theta_averaging: bool = False,  # enable theta averaging across batches within epochs
        device: torch.device | str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize simplified mini-batch Fisher scoring.

        Args:
            model_params: Initial parameter values
            compute_B_U: Function that computes Fisher matrix B and gradient U from batch
            penalty: Penalty/regularization configuration (use this instead of damping)
            max_step_norm: Maximum allowed step norm for stability
            device: Device for computations
            dtype: Data type for computations
        """
        # Copy and prepare parameters
        self.theta = {
            k: v.to(device=device, dtype=dtype).detach().clone().requires_grad_(False)
            for k, v in model_params.items()
        }

        self.compute_B_U = compute_B_U
        self.penalty = penalty
        self.max_step_norm = float(max_step_norm)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.adaptive_lr_bool = adaptive_lr_bool
        self.fisher_ema = float(fisher_ema)

        # U monitoring (always enabled) - running average of U vectors within each epoch
        self.U_avg = None  # Running average of U vectors (reset each epoch)
        self.U_count = 0  # Number of U updates within current epoch
        self.U_history = []  # List of U averages at end of each epoch

        # Update mode
        self.update_mode = update_mode
        if update_mode not in ['batch', 'epoch']:
            raise ValueError(f"update_mode must be 'batch' or 'epoch', got {update_mode}")

        # For epoch-level accumulation
        self.B_accumulator = None
        self.U_accumulator = None
        self.batch_count = 0

        # Theta averaging across batches within epochs
        self.theta_averaging = theta_averaging
        self.theta_avg = None  # Running average of theta parameters within epoch
        self.theta_count = 0  # Number of theta updates within epoch
        self.theta_history = []  # List of theta averages at end of each epoch

        # Initialize momentum buffer (only used if momentum > 0)
        self.momentum_buffer = None

        # Initialize Fisher EMA buffer (only used if fisher_ema > 0)
        self.J_obs_buffer = None

        # Compute total parameter dimension
        self._d = sum(v.numel() for v in self.theta.values())

    def _penalty_terms(self) -> tuple[Tensor, Tensor]:
        """
        Compute penalty gradient and Hessian.

        For penalty lambda/2 * ||theta - theta_0||^2:
        - Gradient: lambda(theta - theta_0)
        - Hessian: lambda*I

        If penalize_delta=False, the first parameter (delta) is not penalized,

        Returns:
            Tuple of (penalty_gradient, penalty_hessian)
        """
        dev = next(iter(self.theta.values())).device
        dt = next(iter(self.theta.values())).dtype

        if self.penalty is None or self.penalty.lam <= 0:
            return torch.zeros(self._d, device=dev, dtype=dt), torch.zeros(
                (self._d, self._d), device=dev, dtype=dt
            )

        lam = self.penalty.lam
        theta_vec = torch.cat([v.reshape(-1) for v in self.theta.values()], dim=0)

        if self.penalty.center is not None:
            center_vec = torch.cat([v.reshape(-1) for v in self.penalty.center.values()], dim=0)
            diff = theta_vec - center_vec
        else:
            diff = theta_vec

        # Create penalty mask: don't penalize delta if penalize_delta=False
        penalty_mask = torch.ones(self._d, dtype=dt, device=dev)
        if not self.penalty.penalize_delta:
            # Assume delta is the first parameter (theta['delta'] is concatenated first)
            delta_size = self.theta['delta'].numel()
            penalty_mask[:delta_size] = 0.0

        g = lam * diff * penalty_mask  # lambda * (theta - theta_0) with mask
        H = lam * torch.diag(penalty_mask)  # lambda * diag(mask)

        return g, H

    def step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """
        Perform one simplified Fisher scoring step on a mini-batch.

        Args:
            batch: Mini-batch of pairs with weights

        Returns:
            Dictionary with step information and diagnostics
        """
        # Compute Fisher matrix B and score vector U
        B, U = self.compute_B_U(self.theta, batch)  # [d,d], [d]

        # Enhancement 1: Track U (always enabled)
        self._update_U_history(U)

        # Enhancement 2: Handle different update modes
        if self.update_mode == 'batch':
            # Current behavior: immediate Fisher update
            return self._apply_fisher_step(B, U)
        elif self.update_mode == 'epoch':
            # Accumulate B, U for epoch-level update
            self._accumulate_BU(B, U, batch)

            # Return exact average from queue if available, otherwise current U norm
            avg = self.get_U_running_avg()
            U_norm_to_return = avg if avg is not None else float(torch.linalg.norm(U))

            return {
                'mode': 'accumulating',
                'U_norm': U_norm_to_return,
                'batch_count': self.batch_count,
            }

    def _apply_fisher_step(self, B: Tensor, U: Tensor) -> dict[str, float]:
        """Apply Fisher scoring step with given B and U matrices."""
        # Observed Fisher matrix (negative Hessian)
        J_obs = -B

        # Apply EMA smoothing to Fisher matrix if enabled
        if self.fisher_ema > 0.0:
            if self.J_obs_buffer is None:
                # First step: initialize buffer
                self.J_obs_buffer = J_obs.clone()
            else:
                # EMA update: J_smooth = alpha * J_old + (1-alpha) * J_new
                self.J_obs_buffer = (
                    self.fisher_ema * self.J_obs_buffer + (1.0 - self.fisher_ema) * J_obs
                )
            J_obs = self.J_obs_buffer

        # Add penalty terms (match DRGUTorch regularization approach)
        g_pen, H_pen = self._penalty_terms()
        rhs = -(U - g_pen)  # Match DRGUTorch: -(U - lamb*theta)
        J_total = J_obs - H_pen  # Match DRGUTorch: J - lamb*I (subtracts regularization)

        # Basic Fisher scoring step
        delta = fisher_solve(J_total, rhs)

        # Apply momentum and learning rate
        learning_rate = self.learning_rate
        if self.adaptive_lr_bool:
            cond_num = float(self._cond(J_total))
            learning_rate = self._adaptive_learning_rate(cond_num)

        if self.momentum > 0.0:
            if self.momentum_buffer is None:
                self.momentum_buffer = torch.zeros_like(delta)
            self.momentum_buffer = self.momentum * self.momentum_buffer + learning_rate * delta
            delta = self.momentum_buffer
        else:
            delta = learning_rate * delta

        # Check for excessive step norm (numerical stability), normalized by number of parameters
        step_norm = float(torch.linalg.norm(delta)) / math.sqrt(self._d)
        if step_norm > self.max_step_norm:
            raise ValueError(f'Step norm {step_norm} exceeds max_step_norm {self.max_step_norm}')

        # Apply step
        self._apply_delta(delta)

        # Update theta running average if enabled
        if self.theta_averaging:
            self._update_theta_average()

        # Return basic diagnostics
        # Return exact average from queue if available, otherwise current U norm
        avg = self.get_U_running_avg()
        U_norm_to_return = avg if avg is not None else float(torch.linalg.norm(U))

        return {
            'mode': 'batch' if self.update_mode == 'batch' else 'epoch',
            'U_norm': U_norm_to_return,
            'delta_norm': float(step_norm),
            'condJ': float(self._cond(J_total)),
        }

    def _apply_delta(self, delta_vec: Tensor) -> None:
        """Apply parameter update to all parameters."""
        keys = list(self.theta.keys())
        offset = 0
        for k in keys:
            n = self.theta[k].numel()
            self.theta[k].add_(delta_vec[offset : offset + n].view_as(self.theta[k]))
            offset += n

    def _adaptive_learning_rate(self, cond_num: float) -> float:
        """Compute adaptive learning rate based on Fisher matrix conditioning."""
        # If adaptive LR is disabled, always return base learning rate
        if not self.adaptive_lr_bool:
            return self.learning_rate

        # Apply condition-number based scaling
        if cond_num > 1e8:
            return 0.05 * self.learning_rate  # very conservative for ill-conditioned
        elif cond_num > 1e5:
            return 0.3 * self.learning_rate  # conservative
        elif cond_num > 1e3:
            return 0.7 * self.learning_rate  # moderate
        else:
            return self.learning_rate  # full learning rate for well-conditioned

    @staticmethod
    def _cond(M: Tensor) -> float:
        """Compute condition number of matrix."""
        try:
            s = torch.linalg.svdvals(M)
            return (s.max() / s.min().clamp_min(1e-12)).item()
        except Exception:
            return math.inf

    # U monitoring methods
    def _update_U_history(self, U: Tensor) -> None:
        """Update running average of U vectors."""
        if self.U_avg is None:
            # Initialize average with deep copy of current U
            self.U_avg = U.detach().clone()
            self.U_count = 1
        else:
            # Update running average: avg = (count-1)/count * avg + 1/count * current
            self.U_count += 1
            alpha = 1.0 / self.U_count
            self.U_avg = (1.0 - alpha) * self.U_avg + alpha * U.detach()

    def get_U_running_avg(self) -> float | None:
        """Return norm of running average U vector."""
        if self.U_avg is None:
            return None
        return float(torch.linalg.norm(self.U_avg))

    # Enhancement 2: Batch accumulation methods
    def _accumulate_BU(self, B: Tensor, U: Tensor, batch: dict) -> None:
        """Accumulate B and U matrices across batches."""
        batch_size = len(batch['w_ij'])

        if self.B_accumulator is None:
            self.B_accumulator = B.clone() * batch_size
            self.U_accumulator = U.clone() * batch_size
        else:
            self.B_accumulator += B * batch_size
            self.U_accumulator += U * batch_size

        self.batch_count += batch_size

    def epoch_step(self) -> dict[str, float]:
        """Apply accumulated Fisher step at end of epoch."""
        if self.update_mode != 'epoch':
            raise ValueError("epoch_step() only available in 'epoch' update mode")

        if self.B_accumulator is None:
            raise ValueError('No accumulated B,U available for epoch step')

        # Normalize by total batch count for proper averaging
        B_avg = self.B_accumulator / self.batch_count
        U_avg = self.U_accumulator / self.batch_count

        # Apply Fisher step using accumulated B, U
        result = self._apply_fisher_step(B_avg, U_avg)
        result['batch_count'] = self.batch_count

        # Reset accumulators
        self._reset_accumulators()
        return result

    def _reset_accumulators(self) -> None:
        """Reset B, U accumulators for next epoch."""
        self.B_accumulator = None
        self.U_accumulator = None
        self.batch_count = 0

    # Theta averaging methods
    def _update_theta_average(self) -> None:
        """Update running average of theta within epoch."""
        self.theta_count += 1

        if self.theta_avg is None:
            # Initialize average with deep copy of current theta
            self.theta_avg = {k: v.clone() for k, v in self.theta.items()}
        else:
            # Update running average: avg = (count-1)/count * avg + 1/count * current
            alpha = 1.0 / self.theta_count
            for k in self.theta:
                self.theta_avg[k] = (1.0 - alpha) * self.theta_avg[k] + alpha * self.theta[k]

    def get_averaged_theta(self) -> Theta | None:
        """Get current averaged theta."""
        return self.theta_avg

    def apply_averaged_theta(self) -> bool:
        """Apply averaged theta as current theta. Returns True if averaging was applied."""
        if not self.theta_averaging or self.theta_avg is None:
            return False

        # Replace current theta with averaged values
        for k, v in self.theta_avg.items():
            self.theta[k].copy_(v)
        return True

    def reset_theta_averaging(self) -> None:
        """Reset theta averaging for next epoch."""
        # Save current average to history
        if self.theta_avg is not None:
            self.theta_history.append({k: v.clone() for k, v in self.theta_avg.items()})
        else:
            self.theta_history.append(None)

        # Reset for next epoch
        self.theta_avg = None
        self.theta_count = 0

    def reset_U_averaging(self) -> None:
        """Reset U vector averaging for next epoch."""
        # Save current average to history
        if self.U_avg is not None:
            self.U_history.append(self.U_avg.clone())
        else:
            self.U_history.append(None)

        # Reset for next epoch
        self.U_avg = None
        self.U_count = 0


# ============================================================================
