"""
Helper functions for nearest neighbors.
"""

# Authors: Juntang Wang <juntang.wang@gmail.com>
# License: 

import numpy as np


# -----------------------------------------------------------------------------
#          Functions
# -----------------------------------------------------------------------------
def assert_A_requirements(A, k, tolerance=1e-15):
    """
    Assert that the adjacency matrix A satisfies the following requirements:
    1. All weights are positive
    2. Each column has exactly k non-zero elements
    3. The matrix is column-stochastic
    """
    passed = True
    results = []

    # Check that all weights are positive where the adjacency matrix has non-zero entries
    if (A.data >= -tolerance).all():
        results.append("1. All weights are positive: PASS")
    else:
        results.append("1. All weights are positive: FAIL")
        print("Negative weights found at the following positions:")
        negative_positions = np.argwhere(A.data < -tolerance)
        for pos in negative_positions:
            print(f"Position: {pos}, Value: {A.data[pos]}")
        passed = False

    # Check that each column has exactly k non-zero elements
    # Add small epsilon to avoid divide by zero when comparing with tolerance
    non_zero_elements_per_column = (A > tolerance).sum(axis=0).A1
    if np.allclose(non_zero_elements_per_column, k, rtol=1e-10, atol=tolerance):
        results.append(f"2. Each column has exactly {k} non-zero elements: PASS")
    else:
        results.append(f"2. Each column has exactly {k} non-zero elements: FAIL")
        print("Columns with incorrect number of non-zero elements:")
        incorrect_columns = np.where(~np.isclose(non_zero_elements_per_column, k, rtol=1e-10, atol=tolerance))[0]
        for col in incorrect_columns:
            print(f"Column: {col}, Non-zero elements: {non_zero_elements_per_column[col]}")
        passed = False

    # Check that the matrix is column-stochastic
    # Add small epsilon to avoid divide by zero when summing very small numbers
    column_sums = A.sum(axis=0).A1 + np.finfo(float).eps
    if np.allclose(column_sums, 1, rtol=1e-10, atol=tolerance):
        results.append("3. Matrix is column-stochastic: PASS")
    else:
        results.append("3. Matrix is column-stochastic: FAIL")
        print("Columns that are not column-stochastic (sum != 1):")
        non_stochastic_cols = np.where(~np.isclose(column_sums, 1, rtol=1e-10, atol=tolerance))[0]
        for col in non_stochastic_cols:
            print(f"Column: {col}, Sum: {column_sums[col]}")
        passed = False

    # Print results summary
    for result in results:
        print(result)

    if not passed:
        raise AssertionError("One or more graph requirements were not met.")

