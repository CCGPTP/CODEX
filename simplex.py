# Simplex method implementation in pure Python
# For solving linear programming problems in standard form:
# maximize c^T x  subject to A x <= b, x >= 0

from typing import List


def simplex(c: List[float], A: List[List[float]], b: List[float]):
    """Solve max c^T x subject to A x <= b, x >= 0 using simplex.

    Parameters
    ----------
    c : list of coefficients for objective function
    A : matrix of constraint coefficients
    b : right-hand side vector (non-negative)

    Returns
    -------
    tuple (solution, value)
        solution: list with optimal x values
        value: optimal value of objective function
    """
    m = len(A)  # number of constraints
    n = len(c)  # number of variables

    # Build initial tableau with slack variables
    tableau = []
    for i in range(m):
        tableau.append(A[i] + [0.0] * m + [b[i]])
        tableau[i][n + i] = 1.0  # slack variable
    tableau.append([-ci for ci in c] + [0.0] * m + [0.0])  # objective row

    while True:
        # choose entering variable (most negative coefficient in objective row)
        last_row = tableau[-1]
        pivot_col = min(range(n + m), key=lambda j: last_row[j])
        if last_row[pivot_col] >= 0:
            break  # optimal

        # choose leaving variable using minimum ratio test
        ratios = []
        for i in range(m):
            col_val = tableau[i][pivot_col]
            if col_val > 1e-12:
                ratios.append((tableau[i][-1] / col_val, i))
        if not ratios:
            raise ValueError("Problem is unbounded.")
        _, pivot_row = min(ratios)

        # pivot operation
        pivot_val = tableau[pivot_row][pivot_col]
        tableau[pivot_row] = [x / pivot_val for x in tableau[pivot_row]]
        for i in range(len(tableau)):
            if i != pivot_row:
                row = tableau[i]
                factor = row[pivot_col]
                tableau[i] = [r - factor * s for r, s in zip(row, tableau[pivot_row])]

    solution = [0.0] * n
    for i in range(m):
        pivot_cols = [j for j, val in enumerate(tableau[i][:n]) if abs(val - 1.0) < 1e-12]
        if len(pivot_cols) == 1:
            j = pivot_cols[0]
            solution[j] = tableau[i][-1]
    value = tableau[-1][-1]
    return solution, value


def main():
    # Example problem:
    # maximize z = 3x1 + 2x2
    # subject to:
    # 2x1 + x2 <= 18
    # 2x1 + 3x2 <= 42
    # 3x1 + 1x2 <= 24
    c = [3, 2]
    A = [
        [2, 1],
        [2, 3],
        [3, 1],
    ]
    b = [18, 42, 24]

    sol, val = simplex(c, A, b)
    print("Optimal solution:", sol)
    print("Optimal value:", val)


if __name__ == "__main__":
    main()

