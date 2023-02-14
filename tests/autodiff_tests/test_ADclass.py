import pytest
import numpy as np
from autoDiff_team15_2022.driver import Forward_AD,Reverse_AD
from autoDiff_team15_2022.elemFunctions import *

class Test_forwardAD:
    """This class evaluates forward mode class implementation in driver.py.

    Parameters
    ==========
    fn: defined function to evaluate derivative, gradient or jacobian
    x: value to compute derivative, gradient or jacobian
 
    Returns
    ==========  
    val: int/float for derivative, vector for gradient, or matrix for jacobian of the first order derivatives of the function.


    assert: assert that the functions returns the correct value

    """

    def test_init(self):
    #test init function for forward AD
        def fn(x,y):
            return 2**x + 1/y
        z = Forward_AD(fn)
        assert len(z.fn) == 1
        assert type(z.fn) == list

    def test_simple_vals(self):
    #test values for univariate function input for forward AD
        def fn(x):
            return 2**x
        z = Forward_AD(fn)
        vals = z.values(3)
        vals_list = z.values([3])
        print(vals)
        assert vals == 8
        assert vals_list == 8
        assert z.val == np.array(8)
        assert z.inputs == [3]

    def test_simple_deriv(self):
    #test derivative for univariate function for forward AD 
        def fn(x):
            return 2**x
        z = Forward_AD(fn)
        deriv = z.grad(3)
        deriv_list = z.grad([3])
        assert deriv == np.array(2**3*np.log(2))
        assert deriv_list == np.log(256)
        assert z.inputs == [3]
        assert z.der == np.log(256)
        
    def test_multiple_val(self):
    #test values for multivariate function for forward AD 
        def fn(x,y):
            return 2**x + exp(y)
        z = Forward_AD(fn)
        vals = z.values([3,-2])
        assert vals == 2**3 + np.exp(-2)

    def test_multiple_input_deriv(self):
    #test gradient for multivariate function
        def fn(x,y):
            return 2**x + exp(y)
        z = Forward_AD(fn)
        deriv = z.grad([3,-2])
        assert deriv[0] == 2**3*np.log(2)
        assert deriv[1] == np.exp(-2)

    def test_multiple_func_values(self):
    #test values for mutliple functions input for forward AD 
        def fn(x,y):
            return 2**x + exp(y)
        def fn2(x,y):
            return x**3 + 1/y
        z = Forward_AD([fn,fn2])
        values = z.values([3,-2])
        assert values[0] == 2**3+np.exp(-2)
        assert values[1] == 26.5

    def test_multiple_func_grad(self):
    #test gradient for mutliple functions input for forward AD 
        def fn(x,y):
            return 2**x + exp(y)
        def fn2(x,y):
            return x**3 + 1/y
        z = Forward_AD([fn,fn2])
        deriv = z.grad([3,-2])
        assert deriv[0][0] == 2**3*np.log(2)
        assert deriv[0][1] == np.exp(-2)
        assert deriv[1][0] == 3*(3**2)
        assert deriv[1][1] == -1/((-2)**2)

class Test_reverseAD:
    """This class evaluates forward mode class implementation in driver.py.

    Parameters
    ==========
    fn: defined function to evaluate derivative, gradient or jacobian
    x: value to compute derivative, gradient or jacobian
 
    Returns
    ==========  
    val: int/float for derivative, vector for gradient, or matrix for jacobian of the first order derivatives of the function.


    assert: assert that the functions returns the correct value

    """
    def test_init(self):
        def fn(x,y):
            return 2**x + 1/y
        z = Reverse_AD(fn)
        assert len(z.fn) == 1
        assert type(z.fn) == list

    def test_simple_vals(self):
    #test values for reverse AD object
        def fn(x):
            return 2**x
        z = Reverse_AD(fn)
        vals = z.values(3)
        vals_list = z.values([3])
        print(vals)
        assert vals == 8
        assert vals_list == 8
        assert z.val == np.array(8)
        assert z.inputs[0].value == 3
    
    def test_simple_deriv(self):
    #test derivative for simple univariate function using reverse AD 
        def fn(x):
            return 2**x
        z = Reverse_AD(fn)
        deriv = z.grad(3)
        deriv_list = z.grad([3])
        assert deriv == np.array(2**3*np.log(2))
        assert deriv_list == np.log(256)
        assert z.inputs[0].value == 3
        assert z.der == np.log(256)

    def test_multiple_val(self):
    #test values for mutlivariate function for reverse AD
        def fn(x,y):
            return 2**x + exp(y)
        z = Reverse_AD(fn)
        vals = z.values([3,-2])
        assert vals == 2**3 + np.exp(-2)

    def test_multiple_input_deriv(self):
    #test gradient for mutlivariate function for reverse AD
        def fn(x,y):
            return 2**x + exp(y)
        z = Reverse_AD(fn)
        deriv = z.grad([3,-2])
        assert deriv[0] == 2**3*np.log(2)
        assert deriv[1] == np.exp(-2)

    def test_multiple_func_values(self):
    #test values for mutliple function inputs for reverse AD
        def fn(x,y):
            return 2**x + exp(y)
        def fn2(x,y):
            return x**3 + 1/y
        z = Reverse_AD([fn,fn2])
        values = z.values([3,-2])
        assert values[0] == 2**3+np.exp(-2)
        assert values[1] == 26.5

    def test_multiple_func_grad(self):
    #test gradietn for mutliple function inputs for reverse AD
        def fn(x,y):
            return 2**x + exp(y)
        def fn2(x,y):
            return x**3 + 1/y
        z = Reverse_AD([fn,fn2])
        deriv = z.grad([3,-2])
        assert deriv[0][0] == 2**3*np.log(2)
        assert deriv[0][1] == np.exp(-2)
        assert deriv[1][0] == 3*(3**2)
        assert deriv[1][1] == -1/((-2)**2)