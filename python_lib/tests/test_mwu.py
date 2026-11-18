import numpy as np
import pytest
from scipy import stats

from robustinfer.mwu import zero_trimmed_u


def test_zero_trimmed_u_simple_arrays():
    # Test with two simple arrays
    x = [1, 2, 3, 0, 0]
    y = [2, 3, 4, 0]
    W, var_W, p_value = zero_trimmed_u(x, y)

    # Check if the output is as expected
    assert isinstance(W, float)
    assert isinstance(var_W, float)
    assert isinstance(p_value, float)

    # Check if the p-value is in the range [0, 1]
    assert 0 <= p_value <= 1


def test_zero_trimmed_u_zero_arrays():
    # Test with arrays containing only zeros
    x_zero = [0, 0, 0]
    y_zero = [0, 0]
    W_zero, var_W_zero, p_value_zero = zero_trimmed_u(x_zero, y_zero)

    # Check if the output is as expected for zero arrays
    assert W_zero == 0
    assert var_W_zero == 0
    assert p_value_zero == 1.0


def test_zero_trimmed_u_mixed_arrays():
    # Test with arrays containing a mix of zeros and positive numbers
    x_mixed = [0, 0, 1, 2, 3]
    y_mixed = [0, 4, 5, 0, 6]
    W_mixed, var_W_mixed, p_value_mixed = zero_trimmed_u(x_mixed, y_mixed)

    # Check if the output is as expected
    assert isinstance(W_mixed, float)
    assert isinstance(var_W_mixed, float)
    assert isinstance(p_value_mixed, float)
    assert 0 <= p_value_mixed <= 1


def test_zero_trimmed_u_large_arrays():
    # Test with large arrays
    x_large = [0] * 100 + [1] * 50
    y_large = [0] * 80 + [2] * 70
    W_large, var_W_large, p_value_large = zero_trimmed_u(x_large, y_large)

    # Check if the output is as expected
    assert isinstance(W_large, float)
    assert isinstance(var_W_large, float)
    assert isinstance(p_value_large, float)
    assert 0 <= p_value_large <= 1


def test_zero_trimmed_u_edge_case_empty_arrays():
    # Test with empty arrays (should raise an assertion error)
    x_empty = []
    y_empty = []
    with pytest.raises(AssertionError) as excinfo:
        zero_trimmed_u(x_empty, y_empty)
    assert str(excinfo.value) == 'Both input arrays must be non-empty.'


def test_zero_trimmed_u_edge_case_negative_values():
    # Test with arrays containing negative values (should raise an assertion error)
    x_negative = [-1, 0, 1]
    y_negative = [0, -2, 3]
    with pytest.raises(AssertionError) as excinfo:
        zero_trimmed_u(x_negative, y_negative)
    assert 'All values in x must be non-negative.' in str(
        excinfo.value
    ) or 'All values in y must be non-negative.' in str(excinfo.value)


def test_zero_trimmed_u_tied_values_across_samples():
    # Positive values that tie across the two samples must not bias W
    d = [0, 0, 0, 1, 1, 1, 2, 2, 3, 5]
    W, var_W, p_value = zero_trimmed_u(d, d)
    assert W == 0
    assert p_value == 1.0

    # Swapping the samples flips the sign of W
    x = [0, 0, 1, 1, 1, 2, 3]
    y = [0, 1, 1, 2, 2, 3, 4]
    W_xy, _, p_xy = zero_trimmed_u(x, y)
    W_yx, _, p_yx = zero_trimmed_u(y, x)
    assert W_xy == pytest.approx(-W_yx)
    assert p_xy == pytest.approx(p_yx)


def test_zero_trimmed_u_matches_mann_whitney_without_zeros():
    # Without zeros, W is the Mann-Whitney U of y minus n0 * n1 / 2
    x = np.array([1, 1, 2, 2, 3, 4, 4])
    y = np.array([1, 2, 2, 3, 3, 4, 5, 5])
    w_stat, _, _ = zero_trimmed_u(x, y)
    u_y = stats.mannwhitneyu(y, x).statistic
    assert w_stat == pytest.approx(u_y - len(x) * len(y) / 2)
