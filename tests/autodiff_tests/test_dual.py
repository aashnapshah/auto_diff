import pytest
import numpy as np
from autoDiff_team15_2022.dualNum import DualNumber


class Test_dual:  

    """This class uses evaluates the dual numbers module.

    Parameters
    ==========
    z: Dual Number object which takes float/integer as arguments
 
    Returns
    ==========  
    real: value of the Dual object
    dual: first derivative of the Dual object

    assert: assert that the overloaded methods returns the correct value

    """
    def test_init(self):
        z = DualNumber(1,2)
        assert(z.real == 1)
        assert(z.dual == 2)
        # Testing creation with float values
        z2 = DualNumber(1.0,2.0)
        assert(z2.real ==1)
        assert(z2.dual ==2)
    
    def test_add(self):
        # Testing addition with integers
        z = DualNumber(1,2)
        z2 = z + 1
        assert(z2.real == 2)
        assert(z2.dual == 2)
        # Testing addition with float
        z3 = z + 1.0
        assert(z3.real == 2)
        assert(z3.dual == 2)
        # Testing addition with another dual number
        z4 = DualNumber(4,5)
        z5 = z + z4
        assert(z5.real == 5)
        assert(z5.dual == 7)
        with pytest.raises(TypeError):
            z +'1'
            '1' +z

    def test_radd(self):
        # Testing reverse addition with integers
        z = DualNumber(1,2)
        z2 = 1 + z 
        assert(z2.real == 2)
        assert(z2.dual == 2)
        # Testing reverse addition with float
        z3 = 1.0 + z
        assert(z3.real == 2)
        assert(z3.dual == 2)
    
    def test_sub(self):
        # Testing subtraction with integers
        z = DualNumber(1,2)
        z2 = z - 2
        assert(z2.real == np.negative(1))
        assert(z2.dual == 2)
        # Testing subtraction with float
        z3 = z - 2.0
        assert(z3.real == np.negative(1))
        assert(z3.dual == 2)
        # Testing subtraction with another dual number
        z4 = DualNumber(4,5)
        z5 = z - z4
        assert(z5.real == np.negative(3))
        assert(z5.dual == np.negative(3))
        with pytest.raises(TypeError):
            z - '1'
            '1' - z
            
    def test_rsub(self):
        # Testing reverse subtraction with integers
        z = DualNumber(1,2)
        z2 = 2 - z
        assert(z2.real == 1)
        assert(z2.dual == np.negative(2))
        # Testing reverse subtraction with float
        z3 = 2.0 - z
        assert(z3.real == 1)
        assert(z3.dual == np.negative(2))
    
    def test_mul(self):
        # Testing multiplication with integers
        z = DualNumber(1,2)
        z2 = z * np.negative(2)
        assert(z2.real == np.negative(2))
        assert(z2.dual == np.negative(4))
        # Testing multiplication with float
        z3 = z * 2.0
        assert(z3.real == 2)
        assert(z3.dual == 4)
        # Testing multiplication with another dual number
        z4 = DualNumber(np.negative(1),np.negative(2))
        z5 = z * z4
        assert(z5.real == np.negative(1))
        assert(z5.dual == np.negative(4))
        with pytest.raises(TypeError):
            z *'1'
            '1' *z
    
    def test_rmul(self):
        # Testing reverse multiplication with integers
        z = DualNumber(1,2)
        z2 = -2 * z
        assert(z2.real == np.negative(2))
        assert(z2.dual == np.negative(4))
        # Testing reverse multiplication with float
        z3 = 2.0 * z
        assert(z3.real == 2)
        assert(z3.dual == 4)

    def test_truediv(self):
        # Testing division with integers
        z = DualNumber(1,2)
        z2 = z / 2
        assert(z2.real == .5)
        assert(z2.dual == 1)
        # Testing division with float
        z3 = z / 2.0
        assert(z3.real == .5)
        assert(z3.dual == 1)
        # Testing division with another dual number
        z4 = DualNumber(-1,-2)
        z5 = z / z4
        assert(z5.real == -1)
        assert(z5.dual == 0)
        with pytest.raises(TypeError):
            z /'1'
            '1' /z
    
    def test_rtruediv(self):
        # Testing division with integers
        z = DualNumber(1,2)
        z2 = 2 / z
        assert(z2.real == 2)
        assert(z2.dual == -4)
        # Testing division with float
        z3 = 2.0 / z
        assert(z3.real == 2)
        assert(z3.dual == -4)

    def test_pow(self):
        # Testing raising dual number to integer power
        z = DualNumber(2,2)
        z2 = z**3
        assert(z2.real == 8)
        assert(z2.dual == 24)
        # Testing raising dual number to float power
        z3 = z**3.0
        assert(z3.real == 8)
        assert(z3.dual == 24)
        # Testing raising dual number to another dual number
        z4 = DualNumber(-1,-2)
        z5 = z ** z4
        assert(z5.real == .5)
        assert(z5.dual == 0.5*(-2*np.log(2)-1))
        with pytest.raises(TypeError):
            z **'1'
            '1' **z

    def test_rpow(self):
        # Testing raising integer to dual power
        z = DualNumber(2,1)
        z2 = 3**z
        assert(z2.real == 9)
        assert(z2.dual == 9*np.log(3))
        # Testing raising float to dual power
        z3 = 3.0**z
        assert(z3.real == 9)
        assert(z3.dual == 9*np.log(3))
        
    def test_neg(self):
        z = DualNumber(1,2)
        z2 = -z
        assert(z2.real == -1)
        assert(z2.dual == -2)


