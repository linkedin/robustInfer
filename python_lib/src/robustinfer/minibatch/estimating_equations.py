"""
Simplified Estimating Equations for DRGU and Other Models

This module provides the interface between mini-batch Fisher scoring and
specific model estimating equations, leveraging the main drgu_torch implementation.
"""

from __future__ import annotations

import torch

from ..drgu import compute_B_U, make_Xg

Tensor = torch.Tensor
Theta = dict[str, Tensor]


def convert_minibatch_to_drgu_format(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Convert mini-batch format to main DRGU format.

    Mini-batch format:
    - xi, xj: [m, p] features for pairs (i,j)
    - yi, yj: [m] outcomes for pairs
    - zi, zj: [m] treatments for pairs
    - w_ij: [m] Horvitz-Thompson weights (optional)

    DRGU format:
    - Wt_i, Wt_j: [m, p+1] treatment model features (with intercept)
    - Xg_ij, Xg_ji: [m, 2p+1] outcome model features
    - yi, yj, zi, zj: [m] outcomes and treatments
    """
    xi, xj = batch['xi'], batch['xj']
    yi, yj = batch['yi'], batch['yj']
    zi, zj = batch['zi'], batch['zj']

    m, p = xi.shape
    device, dtype = xi.device, xi.dtype

    # Add intercept for treatment model: Wt = [1, x]
    ones = torch.ones(m, 1, dtype=dtype, device=device)
    Wt_i = torch.cat([ones, xi], dim=1)  # [m, p+1]
    Wt_j = torch.cat([ones, xj], dim=1)  # [m, p+1]

    # Create outcome model features: Xg = [1, wi, wj]
    Xg_ij = make_Xg(xi, xj)  # [m, 2p+1]
    Xg_ji = make_Xg(xj, xi)  # [m, 2p+1]

    return {
        'Wt_i': Wt_i,
        'Wt_j': Wt_j,
        'Xg_ij': Xg_ij,
        'Xg_ji': Xg_ji,
        'yi': yi,
        'yj': yj,
        'zi': zi,
        'zj': zj,
    }


def drgu_compute_B_U(theta: Theta, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """
    Simplified DRGU kernel - ignoring weights for debugging.

    Args:
        theta: Model parameters {'delta', 'beta', 'gamma'}
        batch: Mini-batch data {'xi', 'xj', 'yi', 'yj', 'zi', 'zj', 'w_ij'}

    Returns:
        Tuple of (B, U) - Fisher information matrix and score vector
    """
    # Convert mini-batch format to main DRGU format
    drgu_data = convert_minibatch_to_drgu_format(batch)

    # Ensure data dtype matches theta dtype for consistency
    target_dtype = theta['delta'].dtype
    target_device = theta['delta'].device
    drgu_data = {k: v.to(dtype=target_dtype, device=target_device) for k, v in drgu_data.items()}

    # Consider add Horvitz-Thompson weights in future

    # Use identity V_inv - standard for DRGU
    V_inv = torch.eye(3, dtype=theta['delta'].dtype, device=theta['delta'].device)
    B, U = compute_B_U(theta, V_inv, drgu_data)

    return B, U


def drgu_compute_B_Sig(
    theta: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute B and Sig matrices for DRGU using the original drgu_torch implementation.

    This function converts minibatch format to drgu_torch format and uses the
    original compute_B_U_Sig function to get proper Sig matrix computation.

    Args:
        theta: Parameter estimates with keys 'delta', 'beta', 'gamma'
        batch: Mini-batch from PairBatchIterableDataset

    Returns:
        B: Fisher information matrix [d_total, d_total]
        Sig: Variance matrix for sandwich estimation [d_total, d_total]
    """
    # Import the original drgu_torch functions
    from ..drgu import _compute_B_u_ij

    drgu_data = convert_minibatch_to_drgu_format(batch)

    # Ensure data types match theta
    target_dtype = theta['delta'].dtype
    target_device = theta['delta'].device
    drgu_data = {
        k: v.to(dtype=target_dtype, device=target_device)
        for k, v in drgu_data.items()
        if isinstance(v, torch.Tensor)
    }

    # Use identity V_inv - standard for DRGU
    V_inv = torch.eye(3, dtype=theta['delta'].dtype, device=theta['delta'].device)
    B, u_ij = _compute_B_u_ij(theta, V_inv, drgu_data)

    # compute outer product of centered u_ij
    u_ij_centered = u_ij - u_ij.mean(dim=0)
    Sig_ij = torch.einsum('ij,ik->ijk', u_ij_centered, u_ij_centered)
    Sig = torch.mean(Sig_ij, dim=0)

    return B, Sig
