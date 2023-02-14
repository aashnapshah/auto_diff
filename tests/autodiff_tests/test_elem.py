
import pytest
import numpy as np
from autoDiff_team15_2022.differentiation import derivative
from autoDiff_team15_2022.elemFunctions import *


class Test_elem():  
    """This class uses evaluates the elementary functions module.
    This module implements dual Numbers and each operation in the module is tested using pytests. 

    Parameters
    ==========
    fn: elementary operation which is testsed
    x: scalar/integer - value at which the derivative is evaluated at

    Returns
    ==========  
    val: value at which the derivative of the defined function 

    assert: assert that the overloaded elementary operation returns the correct value

    """


    def test_sin(x):
    #test sin function for forward mode implementation
        def fn(x):
            return sin(x)
        val = derivative(fn, 0.5)
        assert val == np.cos(0.5)

        with pytest.raises(TypeError): #raise error if argument is an unsupported type
            sin(None)
        
    def test_sin_node(x):
    #test sin function for Node implemenation
        x = Node(1)
        z = sin(x)
        z.gradient = 1

        assert z.value == np.sin(x.value)
        assert x.grad() == np.cos(x.value)

    def test_sin_support(x): 
        assert sin(0) == np.sin(0)


    def test_cos(x):
    #test cos function for forward mode implementation
        def fn(x):
            return cos(x)
        val = derivative(fn, 0.5)
        assert val == -np.sin(0.5)

        with pytest.raises(TypeError):#raise error if argument is an unsupported type
            cos(None)

    def test_cos_support(x): 
        assert cos(0) == np.cos(0)

    def test_cos_node(x):
    #test cos function for Node implementation (i.e. reverse mode)
        x = Node(1)
        z = cos(x)
        z.gradient = 1

        assert z.value == np.cos(x.value)
        assert x.grad() == -np.sin(x.value)

    def test_tan(x):
    #test tan function for forward mode implemenation
        def fn(x):
            return tan(x)
        val = derivative(fn, 0.5)
        assert val == 1/(np.cos(0.5)**2)
        
        with pytest.raises(TypeError): #raise error if argument is an unsupported type
            tan(None)

    def test_tan_support(x): 
        assert tan(0) == np.tan(0)

    def test_tan_node(x):
    #test tan function for node implementation
        x = Node(1)
        z = tan(x)
        z.gradient = 1

        assert z.value == np.tan(x.value)
        assert x.grad() == 1/(np.cos(x.value)**2)

    def test_sinh(x):
    #test sinh function for forward mode implemtation
        def fn(x):
            return sinh(x)
        val = derivative(fn, 0.5)
        assert val == np.cosh(0.5)

        with pytest.raises(TypeError):#raise error if argument is an unsupported type
            sinh(None)

    def test_sinh_support(x): 
        assert sinh(0) == np.sinh(0)

    def test_sinh_node(x):
    #test sinh function for node implemenation
        x = Node(1)
        z = sinh(x)
        z.gradient = 1

        assert z.value == np.sinh(x.value)
        assert x.grad() == np.cosh(x.value)

    def test_cosh(x):
    #test cosh for for forward mode implementation
        def fn(x):
            return cosh(x)
        val = derivative(fn, 0.5)
        assert val == np.sinh(0.5)

        with pytest.raises(TypeError):#raise error if argument is an unsupported type
            cosh(None)

    def test_cosh_support(x): 
        assert cosh(0) == np.cosh(0)

    def test_cosh_node(x):
    #test cosh for node implementation
        x = Node(1)
        z = cosh(x)
        z.gradient = 1

        assert z.value == np.cosh(x.value)
        assert x.grad() == np.sinh(x.value)
    
    def test_tanh(x):
    #test tanh functin for forward mode implementation
        def fn(x):
            return tanh(x)
        val = derivative(fn, 0)
        assert val ==  1/(np.cosh(0)**2)

        with pytest.raises(TypeError):#raise error if argument is an unsupported type
            tanh(None)

    def test_tanh_support(x): 
        assert tanh(0) == 0

    def test_tanh_node(x):
    #test tanh function for node implementation in reverse mode
        x = Node(1)
        z = tanh(x)
        z.gradient = 1

        assert z.value == np.tanh(x.value)
        assert x.grad() == 1/(np.cosh(x.value)**2)
    
    def test_arcsin(x):
    #test arcsin function for forward mode implementation
        def fn(x):
            return arcsin(x)
        val = derivative(fn, 0.5)
        assert val ==  1/(np.sqrt(1-(0.5**2)))

        with pytest.raises(TypeError): #raise error if argument is an unsupported type
            arcsin(None)

    def test_arcsin_support(x): 
        assert arcsin(0) == 0

    def test_arcsin_node(x):
    #test arcsin function for node implementation in reverse mode
        x = Node(0)
        z = arcsin(x)
        z.gradient = 1

        assert x.grad() == 1/np.sqrt(1-(x.value**2))

    def test_arccos(x):
    #test arccos function for forward mode implementation
        def fn(x):
            return arccos(x)
        val = derivative(fn, 0.5)
        assert val ==  -1/(np.sqrt(1-(0.5**2)))

        with pytest.raises(TypeError):#raise error if argument is an unsupported type
            arccos(None)

    def test_arccos_support(x): 
        assert arccos(0) == np.pi/2

    def test_arccos_node(x):
    #test arccos function for node implementation in reverse mode
        x = Node(0)
        z = arccos(x)
        z.gradient = 1

        assert x.grad() == -1/np.sqrt(1-(x.value**2))

    def test_arctan(x):
    #def arctan function in forward mode implementation
        def fn(x):
            return arctan(x)
        val = derivative(fn, 0.5)
        assert val ==  1/(0.5**2 + 1)

        with pytest.raises(TypeError):#raise error if argument is an unsupported type
            arctan(None)

    def test_arctan_support(x): 
        assert arctan(0) == 0

    def test_arctan_node(x):
    #test arctan function in node implementation for reverse mode
        x = Node(1)
        z = arctan(x)
        z.gradient = 1

        assert x.grad() == 1/(x.value**2 + 1)


    # def test_power(x):
    #     def fn(x):
    #         return x ** 2
    #     val = derivative(fn,4)
    #     assert val ==  8

    #     with pytest.raises(TypeError):
    #         power(None)

    # def test_power_support(x): 
    #     assert 2**2 == 4
    
    def test_log(x,base=10):
    #test log function for forward mode implementation
        def fn(x):
            return log(x)
        val = derivative(fn, 1)
        assert val == 1/(np.log(10))

        with pytest.raises(TypeError): #raise error if argument is an unsupported type
            log(None)

    def test_log_support(x): 
        assert log(0.5,base=10) == pytest.approx(np.log10(0.5))


    def test_log_node(x):
    #test log function for reverse mode implementaiton
        x = Node(1)
        z = log(x)
        z.gradient = 1

        assert x.grad() == 1/(np.log(10))

    def test_exp(x):
    #test exponential function for forward mode implementation
        def fn(x):
            return exp(x)
        val = derivative(fn, 1)
        assert val == np.exp(1)

        with pytest.raises(TypeError): #raise error if argument is an unsupported type
            exp(None)

    def test_exp_node(x):
    #test exponential function for reverse mode implementation
        x = Node(1)
        z = exp(x)
        z.gradient = 1

        assert x.grad() == np.exp(x.value)

    def test_exp_support(x): 
        assert exp(1) == np.exp(1)

    def test_logistic(x):
    #test logistic function for forward mode implementation
        def fn(x):
            return logistic(x)
        val = derivative(fn, 0)
        assert val == (np.exp(-0))/((np.exp(-0)+1)**2)

        with pytest.raises(TypeError): #raise error if argument is an unsupported type
            logistic(None)

    def test_logistic_support(x): 
        assert logistic(0) == 0.5


    def test_logistic_node(x):
    #test logistic function for node implementation in reversenode
        x = Node(0)
        z = logistic(x)
        z.gradient = 1

        assert x.grad() == (np.exp(x.value))/((np.exp(x.value)+1)**2)

    def test_dual_power(x):
    #testing power function for dual numbers
        x = DualNumber(2,3)
        val = x**2
        assert val.real == 4
        assert val.dual == 2*2*3

    def test_dual_sin(x):
    #testing sin function using dual number object
        x = DualNumber(2,3)
        val = sin(x)
        assert val.real == np.sin(2)
        assert val.dual == 3*np.cos(2)
  
    def test_dual_tan(x):
    #testing tan function using dual number object
        x = DualNumber(2,3)
        val = tan(x)
        assert val.real == np.tan(2)
        assert val.dual == 3/(np.cos(2)**2)

    def test_dual_arccos(x):
    #testing tan function using dual number object
        x = DualNumber(0,0.5)
        val = arccos(x)
        assert val.real == np.arccos(0)
        assert val.dual == -0.5/(np.sqrt(1-(0**2)))

    def test_dual_arcsin(x):
    #testing tan function using dual number object
        x = DualNumber(0,0.5)
        val = arcsin(x)
        assert val.real == np.arcsin(0)
        assert val.dual == 0.5/(np.sqrt(1-(0**2)))

    def test_dual_cos(x):
   #testing cos function using dual number object
        x = DualNumber(2,3)
        val = cos(x)
        assert val.real == np.cos(2)
        assert val.dual == -3*np.sin(2)

    def test_dual_exp(x):
   #testing exponential function using dual number object
        x = DualNumber(2,3)
        val = exp(x)
        assert val.real == np.exp(2)
        assert val.dual ==  np.exp(2) * 3


