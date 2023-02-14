#!/usr/bin/env python

import timeit
from autoDiff_team15_2022 import *

def newton_root(f, df, x0, epsilon, max_iter: int = 10000):
    """Implementation of Newton's Method for root finding using automatic differentiation

    Parameters
    ======
    f : function
        Input function for root finding
    df : function
        Derivative of input function
    x0 : int, float
        Initial guess for root
    epsilon : int, float
        Tolerance for considering root to be found
    max_iter : Maximum iterations of algorithm before giving up on finding root

    Returns
    =======
    iterations : int, None
        Number of algorithm iterations needed to find root
    root : float, None
        Found root of input function

    Example
    =======
    # Input function
    >>> def fi(x):
    >>>    return x**3 + sin(x)

    # Value of function
    >>> def fx(x):
    >>>    return ad.values(x)

    # Derivative of function
    >>> def dfx(x):
    >>>    return ad.grad(x)

    # Find root with forward mode
    >>> forward = Forward_AD(fi)
    >>> ad = forward

    # Parameters
    >>> x_guess = 3
    >>> real_root = 0
    >>> epsilon = 1.e-20
    >>> max_iter = 100

    # Find root
    >>> iterations, root = newton_root(fx, dfx, x_guess, epsilon, max_iter)
    >>> print("Number of iterations to successfully find root:", iterations)
    >>> print("Root found:", root)
    >>> print("Real root:", real_root)
    Number of iterations to successfully find root: 9
    Root found: 0.0
    Real root: 0
    """
    # Initiatize iteration count and current input
    i = 1
    x = x0

    # Attempt to find root by looping until maximum number of iterations reached
    while i <= max_iter:
        # Evaluate function at current input value
        fx = f(x)
        
        # Evaluate derivative of function at current input value
        dfx = df(x)

        if abs(fx) < epsilon:
            # Return iteration count and root once root is found
            return i, x

        # Cannot divide by zero if dfx is zero
        if dfx == 0:
            print("Failed to find the root because derivative is 0.")
            return None, None

        # Update input value
        x = x - (fx/dfx)[0]

        # Increment number of iterations of newton's method
        i = i + 1
    
    # Maximum iterations reached without root
    print("Reached maximum iterations. Failed to find the root")
    return None, None

if __name__ == "__main__":
    # Example function
    def fi(x):
        return x**3 + sin(x)

    # Value of function
    def fx(x):
        return ad.values(x)

    # Derivative of function
    def dfx(x):
        return ad.grad(x)

    # Find root with forward mode
    forward = Forward_AD(fi)
    reverse = Reverse_AD(fi)
    ad = forward

    # Parameters
    x_guess = 3
    real_root = 0
    epsilon = 1.e-20
    max_iter = 100

    # Find root
    iterations, root = newton_root(fx, dfx, x_guess, epsilon, max_iter)
    print("Number of iterations to successfully find root:", iterations)
    print("Root found:", root)
    print("Real root:", real_root)

    # Helper function for benchmarking
    def root_wrap_for_benchmark():
        return newton_root(fx, dfx, 2, 1.e-20, 100)

    # Number of function calls for benchmarking
    loops = 50000

    # Benchmark forward and reverse modes if a root exists
    if root == None:
        print("No roots to benchmark.")
    else:
        # Benchmark forward mode
        forward_benchmark = timeit.timeit(stmt='root_wrap_for_benchmark()',globals=globals(), number=loops)
        print(f"Forward mode time to find root {loops} times: {forward_benchmark} seconds.")
    
        # Benchmark reverse mode
        ad = reverse
        reverse_benchmark = timeit.timeit(stmt='root_wrap_for_benchmark()',globals=globals(), number=loops)
        print(f"Reverse mode time to find root {loops} times: {reverse_benchmark} seconds.")

        # Percent difference in performance between forward and reverse modes
        if reverse_benchmark > forward_benchmark:
            print(f"Forward mode faster than reverse mode by {round(100*((reverse_benchmark - forward_benchmark)/ reverse_benchmark) - 1, 2)}%.")
        else:
            print(f"Reverse mode faster than forward mode by {round(100*((forward_benchmark - reverse_benchmark)/ forward_benchmark) - 1, 2)}%.")    
