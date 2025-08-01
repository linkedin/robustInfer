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
    try:
        zero_trimmed_u(x_empty, y_empty)
        assert False, "Expected an assertion error for empty arrays"
    except AssertionError as e:
        assert str(e) == "Both input arrays must be non-empty."

def test_zero_trimmed_u_edge_case_negative_values():
    # Test with arrays containing negative values (should raise an assertion error)
    x_negative = [-1, 0, 1]
    y_negative = [0, -2, 3]
    try:
        zero_trimmed_u(x_negative, y_negative)
        assert False, "Expected an assertion error for negative values"
    except AssertionError as e:
        assert "All values in x must be non-negative." in str(e) or "All values in y must be non-negative." in str(e)
