import pytest
import numpy as np
from autoDiff_team15_2022.differentiation import gradient, derivative, jacobian, get_values
from autoDiff_team15_2022.elemFunctions import *

class Test_diff:
    """This class uses evaluates the differentation driver module which implements forward AD.

    Parameters
    ==========
    fn: defined function to evaluate derivative, gradient or jacobian
    x: value to compute derivative, gradient or jacobian
 
    Returns
    ==========  
    val: int/float for derivative, vector for gradient, or matrix for jacobian of the first order derivatives of the function.


    assert: assert that the functions returns the correct value

    """

    def test_simple_grad(x): #test simple cases for single, scalar input
        def fn(x):
            return x**2 + 3*x
        val = gradient(fn,[3])
        assert val == [9]
    
    def test_simple_deriv(x): #test simple cases for single, scalar input
        def fn(x):
            return x**2 + 3*x
        val = derivative(fn,3)
        assert val ==  9
    
    def test_multi_grad(x): #test multi-scalar vector 
        def fn(x,y):
            return x**2 + 3*y
        val = gradient (fn, [1,2])
        assert val[0] == 2
        assert val[1] == 3

    def test_multi_elem(x): #test multi-variate scalar inputs using trig fuctions
        def fn(x,y):
            return x * 3 + sin(y)
        val = gradient (fn, [1,0])
        assert val[0] == 3
        assert val[1] == np.cos(0)

    def test_jacobian(x): #test jacobian for multi-dimensional functions
        def fn(x,y):
            return x + 2*y
        def fn2(x,y): 
            return 2*x + cos(y)

        val = jacobian([fn,fn2],[0,0])
        assert val[0][0] == 1
        assert val[0][1] == 2
        assert val[1][0] == 2
        assert val[1][1] == 0

    def test_values(self):
        # check output with single function, single variable
        def fn(x):
            return sin(x)
        val = get_values([fn],[1])
        assert val == np.sin(1)
        # check output with single function, multiple variables
        def fn2(x,y):
            return 2*x + y
        val = get_values([fn2],[1,2])
        assert val == [4]
        # check output with multiple functions, single variable
        val = get_values([fn,fn],[1])
        assert val[0] == np.sin(1)
        assert val[1] == np.sin(1)
        # check output with multiple functions, multiple variables
        def fn3(x,y):
            return 2**x +cos(y)
        val = get_values([fn2,fn3],[2,2])
        assert val[0] == 6
        assert val[1] == 4 + np.cos(2)
    
        with pytest.raises(TypeError):
        #raise type erorors for gradient method
            gradient(None)

        with pytest.raises(TypeError):
        #raise type erorors for derivative method
            derivative(None)

        with pytest.raises(TypeError):
        #raise type erorors for jacobian method
            jacobian(None)

        with pytest.raises(TypeError):
        #raise type erorors for get_values method
            get_values(None)
  