import numpy as np
from scipy import stats

def zero_trimmed_u(x, y):
    """Modified Wilcoxon test for zero-inflated data"""
    x, y = np.asarray(x), np.asarray(y)
    n0, n1 = len(x), len(y)

    # Assert that all input values are positive
    assert np.all(x >= 0), "All values in x must be non-negative."
    assert np.all(y >= 0), "All values in y must be non-negative."
    assert n0 > 0 and n1 > 0, "Both input arrays must be non-empty."
    
    # Calculate non-zero proportions
    p_hat0 = np.sum(x > 0) / n0
    p_hat1 = np.sum(y > 0) / n1
    p_hat = max(p_hat0, p_hat1)
    
    # Truncate zeros
    x_nonzero, y_nonzero = x[x > 0], y[y > 0]
    n_plus0, n_plus1 = len(x_nonzero), len(y_nonzero)
    n_prime_0, n_prime_1 = round(n0 * p_hat), round(n1 * p_hat)
    
    # Add zeros to balance proportions
    x_trun = np.concatenate([np.zeros(n_prime_0 - len(x_nonzero)), x_nonzero])
    y_trun = np.concatenate([np.zeros(n_prime_1 - len(y_nonzero)), y_nonzero])
    
    # Compute ranks and statistic
    combined = np.concatenate([y_trun, x_trun])
    # Note: 1) we want descending ranks for y_trun, so we negate combined
    #       2) we use 'ordinal' method (for rank sum this is same as 'average'),
    #        as only one sample has zeros after truncation
    descending_ranks = stats.rankdata(-combined, method='ordinal')
    R1 = np.sum(descending_ranks[:len(y_trun)])
    # negative sign because we have negated combined
    W = - (R1 - len(y_trun) * (len(combined) + 1) / 2)
    
    # Calculate variance
    var_comp1 = (n1**2 * n0**2 / 4) * (p_hat**2) * (
        (p_hat0 * (1 - p_hat0) / n0) + (p_hat1 * (1 - p_hat1) / n1)
    )
    var_comp2 = (n_plus0 * n_plus1 * (n_plus0 + n_plus1)) / 12
    var_W = var_comp1 + var_comp2
    
    # Calculate p-value (2 sided)
    if var_W == 0:
        return W, var_W, 1.0  # If variance is zero, return W and p-value of 1.0
    Z = W / np.sqrt(var_W)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(Z)))
    return W, var_W, p_value
