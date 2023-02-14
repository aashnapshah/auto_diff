

import pytest
import numpy as np
from autoDiff_team15_2022.node import *


class Test_node():

    """This class uses evaluates the node module.
    This module implements the Node object and each operation in the module is tested using pytests. 

    Parameters
    ==========
    x: Node object with an input value (scalar)
    z: elemntary operation using x
    z.gradient: initializing gradient

    Returns
    ==========  
    z.value: value of function z 
    x.grad(): derivative of the function
    x.children[0][0]: value of the derivative in the list of children

    assert: assert that the overloaded elementary operation returns the correct value

    """

    def test_add(self):
    #Test for Node addition operation
        x = Node(2)
        z = 2*x + x
        z.gradient = 1.0

        assert z.value == x.value * 2 + x.value
        assert x.grad() == 3

   
    def test_clear(self):
    #Test for Node clearing operation
        x = Node(1)
        x = x.clear()
        assert x == None
   

    def test_radd(self):
    #Test for Node reverse addition operation
        x = Node(2)
        z = 2 + x
        z.gradient = 1.0

        assert z.value == 2 + x.value
        assert x.grad() == 1
        assert x.children[0][0] == 1


    def test_sub(self):
    #Test for Node subtraction operation
        x = Node(2)
        y = Node(4)
        z =  x - y
        z.gradient = 1.0

        assert z.value == x.value  - y.value
        assert x.grad() == 1
        assert y.grad() == -1


    def test_rsub(self):
    #Test for Node reverse subtraction operation
        x = Node(2)
        z = 3-x
        z.gradient = 1
        x.grad()

        assert z.value == 3 - x.value
        assert x.gradient == -1


    def test_mul(self):
    #Test for Node mutliplication operation
        x = Node(2)
        y = Node(4)
        z =  x**3*y
        z.gradient = 1.0

        assert z.value == (x.value ** 3)*y.value
        assert x.grad() == 48
        assert y.grad() == 8

    def test_rmul(self):
    #Test for Node reverse multiplication operation
        x = Node(2)
        z =  3*x
        z.gradient = 1.0

        assert z.value == 3*x.value
        assert x.grad() == 3

    def test_pow(self):
    #Test for Node power operation using a scalar exponent
        x = Node(3)
        z =  x**2
        z.gradient = 1.0

        assert z.value == x.value**2
        assert x.grad() == 6
        assert x.children[0][0] == 6

        # test for node object as exponent
        x = Node (2)
        y = 2
        z = 2**x
        z.gradient = 1

        assert z.value == 2 ** x.value
        assert x.grad() == y ** x.value * np.log(y)
        assert x.children[0][0] == y ** x.value * np.log(y)
    

    def test_rpow(self):
    #Test for Node reverse power operation
        x = Node(2)
        v = 2
        z =  v**x
        z.gradient = 1.0

        assert z.value == 2**x.value
        assert x.grad() == np.log(16)
        assert z.children == []

    def test_div(self):
    #Test for Node division operation
        x = Node(2) 
        y = Node(4)
        z =  x/y #test for division between two node object
        z.gradient = 1.0

        assert z.value == (x.value/y.value)
        assert x.grad() == 0.25
        assert pytest.approx(y.grad()) == -0.125
        assert x.children[0][0] == 0.25
        assert y.children[0][0] == -0.125

        x = Node(2)
        z =  x/2 #test for division between node object and scalar
        z.gradient = 1.0

        assert z.value == x.value/2
        assert x.grad() == 1/2

    
    def test_rdiv(self):
    #Test for Node reverse division operation
        x = Node(2) 
        z = 2/x
        z.gradient = 1.0

        assert z.value == 2/x.value
        assert x.grad() == -1/2


    def test_neg(self):
    #Test for Node negative operation
        x = Node(-2) 
        z = x
        z.gradient = 1.0
        assert z.value == x.value

        x = Node(2) 
        z = -x
        z.gradient = 1.0
        assert x.children[0][0] == -1
    

    def test_pos(self):
    #Test for Node positive operation
        x = Node(2) 
        z = +x
        z.gradient = 1.0
        assert z.value == x.value
        assert x.grad()== 1

        x = Node(2) 
        z = x
        z.gradient = 1.0

    def test_grad_node(self):
    #test gradient function using two Node objects
        x = Node(7)
        y = Node(2)

        def fn(x,y):
            return 2*x + y 
        z = fn(x,y)
        z.gradient = 1

    #test for x value, df/dx, and children
        assert x.value == 7
        assert x.grad() == 2
        assert x.children[0][0] == 2
    #test for y value, df/dy, and children
        assert y.value == 2
        assert y.grad() == 1
        assert y.children[0][0] == 1
    #test for z value, children, and initialized grad value
        assert z.grad() == 1
        assert z.children == []
        assert z.value == 16


   