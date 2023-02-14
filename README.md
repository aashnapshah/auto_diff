![Image of coverage workflow status](https://code.harvard.edu/CS107/team15/actions/workflows/coverage.yml/badge.svg?branch=main) ![Image of test workflow status](https://code.harvard.edu/CS107/team15/actions/workflows/test.yml/badge.svg?branch=main)

## Introduction:
This software implements automatic differentiation, a computationally ergonomic method to compute first derivatives of functions. As first derivatives are the heart of Jacobian matrices, which in turn are the heart of many methods in dynamical systems and statistics including linearization techniques like Newton's Method, determination of stability of dynamic systems through observation of eigenvalues, bifurcation analysis, and nonlinear least squares regression. Systems can be modelled through explicit coding of the derivation of a model, but this method is tedious and prone to errors. Alternatively, symbolic derivation through a graph based analysis can be used to find derivatives; however, these methods aren't conducive to interpretative code environments such as Python.  This leaves numerical methods such as finite-differences and automatic differentiation to bridge the gap. Finite differences approximates derivatives from first principles, using some small value of $\epsilon$ to approximate the limit to zero; however, these systems can suffer from accuracy problems, and choosing a suitable value $\epsilon$ becomes non-trivial. The last alternative, automatic differentiation, is what this package implements. Automatic differentiation is exact, efficient, and amenable to OOP systems. This library provides a resource for automatic differentiation.

## How to Use:

### Installing and setting up a virual environment 
Our package, ```autoDiff_team15_2022```, is distributed using setuptools, and users can install it using pip-install. The package is currently on pypi.org.

```
#install autoDiff_team15_2022
python3 -m pip install autoDiff_team15_2022

#import autoDiff_team15_2022 into python environment
from autoDiff_team15_2022 import *
```

Using this package only requires Numpy. The pyproject.toml contains dependencies to automatically install Numpy when the package is installed by the user.

### Utilizing the core classes

After installing ```autoDiff_team15_2022```, the user can utilize two core classes: ```Forward_AD``` and ```Reverse_AD```. These two classes live in the ```driver.py``` module and can be used to implement forward or reverse mode AD. Both classes contain the same key attributes: a ```values``` method to retrieve value(s) for a given function and input(s), and ```grad``` method for obtaining the derivative,gradient, and jacobian for scalar univariate functions,  functions, and multiple vector functions, respectively.


#### Forward Mode AD



To implement forward mode, the user must first import the driver module which contains the ```Forward_AD``` class. 

```  
#import autoDiff 
from autoDiff_team15_2022 import *

```  

Second, the user must then define a function to evaluate. For univariate functions with a scalar input, the derivative can be computed. For multivariate functions with scalar or vector inputs, the gradient or jacobian can be computed, respectively. ```Forward_AD``` and ```Reverse_AD``` classes will automatically perform the jacobian, gradient or derivative, with no additional input needed from the user. The following example demonstrates how to implement the package for the core functions:

1) Scalar, univariate functions
```  
#define function to evaluate
>>> def func(x):
        return 2x + sin(x)

#instantiate an Forward Mode AD ojbect
>>> ad_obj = Forward_AD([func])

#retrieve derivative evaluate at x = 1 
>>> ad_obj.grad([1])
[2.54030]

#retrieve values of f(x) at x = 1
>>> ad_obj.values([1])
[2.84147]

```  

The ```values``` and ```grad``` methods returns a single value in a numpy array with the function value and function gradient at the input values, respectively.

Similarly, the directional derivative or gradient can be computed for multivariate functions with a scalar input:

2) Mulitple functions
```  
#define functions to evaluate for gradient 
>>> def func(x,y):
        return 2*sin(x) + cos(y) + 1

#instantiate an Forward Mode AD ojbect
>>> ad_obj = Forward_AD([func])

#retrieve derivative evaluate at x = 1 and y = 0
>>> ad_obj.grad([1,0])
[1.08060,0]

#retrieve values from the f(x) for x = 1 and y = 0
>>> ad_obj.values([1,0])
[3.6829]
 
``` 
Here, the gradient is computed for the function, $2*sin(x) + cos(y) + 1$ for the values $x=0$ and $y=1$. The gradient function will return an array with $\frac{\partial f}{\partial x}$ as the first value and $\frac{\partial f}{\partial y}$ as the second value. The user can also get the value of the function for a given x and y value using the ```values``` method. 

The user can also compute the jacobian for higher order functions. Using the same procedure as above, the ```grad``` method will return a vector valued matrix of first order partial derivatives.

3) Multiple functions
``` 
#define functions to evaluate for jacobian
>>> def fn(x,y):
        return log(x)

>>> def fn2(x,y):
        return 2*sin(x) + cos(y) + x/4

#instantiate an Forward Mode AD ojbect
>>> ad_obj = Forward_AD([fn1,fn2])

#retrieve derivative evaluate at x = 1 and y = 0
>>> ad_obj.grad([1,0])
[1.33060,0][2.93294,0]

#retrieve values from the f(x) for x = 1 and y = 0
>>> ad_obj.values([1,0])
[2.93294,0]

```

The resulting jacobian matrix is: $\begin{matrix}
\frac{1}{x} & 0 \\
2cos(x) + \frac{1}{4} & -sin(y) 
\end{matrix}$

and is evaluated at the $x =0$ and $y=1$. The ```grad``` method will return a matrix, like above, evaluated at a given x and y value. The user can retrieve the value of the function for a given x and y value. The ```values``` method will return a a list with value fo the first function as the first value and the value of the second function as the second value.


### Reverse Mode AD 


Similar to forward mode, the user must first import the driver module which contains the ```Reverse_AD``` class. Importantly, the ```Reverse_AD``` class utilizes the ```Node``` class which stores the derivative of each node and the relationship between the child and parent node, such that gradient can computed recursively during the reverse pass. The user must then define a function to evaluate. The user then can instantiate a Reverse mode AD object. See the following example:

1) single value, scalar function
```  
#define function to evaluate
>>> def func(x):
        return exp(x) + x ** 2

#instantiate an Forward Mode AD ojbect
ad_obj = Reverse_AD([func])

#retrieve derivative evaluate at x = 1 
>>> ad_obj.grad([1])
[4.71828]

#retrieve values of f(x) at x =1
>>> ad_obj.values([1])
[3.71828]

```  
Similar to forward mode, once the reverse mode object is instantiated the value of the function for a given x value and its derivative can be retrieved using the ```values``` and ```grad``` methods, respectively. Examples for single functions with vector inputs and multiple functions are shown below:

2) Single value function, vector input
```  
#define function to evaluate
>>> def func(x,y):
        return sin(x) + logistic(y)

#instantiate an Reverse Mode AD ojbect
>>> ad_obj = Reverse_AD([func])

#retrieve derivative evaluate at x = 1  and y = 0
>>> ad_obj.grad([1,0])
[0.54030, 0.25]

#retrieve values of f(x) at x =1 and y = 0
>>> ad_obj.values([1,0])
[1.34147]

```  
3) Multiple Functions

``` 
#define functions to evaluate for jacobian
 >>> def fn1(x, y):
        return sin(x)+ logistic(y)
    
 >>> def fn2(x, y):
        return cos(x)+y+30*x + 40*y

 >>> def fn3(x, y):
        return 1 + x**2 + y

#instantiate an Forward Mode AD ojbect
>>> ad_obj = Reverse_AD([fn1,fn2,fn3])

#retrieve derivative evaluate at x = 1 and y = 0
>>> ad_obj.grad([1,0])
[[0.54030, 0.25][29.1585, 41][2, 1]]

#retrieve values from the f(x) for x = 1 and y = 0
>>> ad_obj.values([1,0])
[1.34147,30.54030, 2]

```

Ultimately, the user should manage to implement the forward and reverse mode AD fairly easily using the examples above. The current form does require that the user adhere to implementation guidelines described above, otherwise the user may encounter errors. 

## Software Organization:

**Directory Structure**: The package is organized with a series subpackages which implement basic functionalities to perform autodifferentiation.

``` 
    autoDiff_team15_2022
    ├── README.md
    ├── pyproject.toml
    ├── LICENSE
    ├── examples
       ├── rootfinding.py
       ├── rootfinding.ipynb
    ├── src
        ├── autoDiff_team15_2022
            ├── __init__.py
            ├── dualNum.py
            ├── elemFunctions.py
            ├── differentiation.py
            ├── node.py   
            ├── driver.py
    ├── tests
       ├── autodiff_tests
           ├── test_elem.py
           ├── test_dual.py
           ├── test_differentiation.py
           ├── test_node.py
           ├── test_driver.py
       ├── run_tests.sh
       ├── check_coverage.sh

```
**Source Code Modules**: 
  - ```node.py```
      - Given a function, returns a node in the computational graph. It contains a class object, ```Node```, and stores the children of the node, the value of the function, and the derivative of the function for a given value as attributes. ```node.py``` also overloads basic elementary operations for ```Node``` class.
  - ```elemFunctions.py```:
      - Overloads elementary operation functions and stores their derivatives. For an a DualNumber or Node input, this modules calculates the value and its derviative. However, if the input is another type, we resort to numpy implementations of mathematical operators. 
  - ```dualNum.py```:
      - Implements a DualNumber class and overloads basic numerical and mathematical operations of python. The class implements operators to calculate the the numerical value of a given function and at its derivatives. There is also a derivative function which implements the chain rule for non-elementary operations.  
  - ```differentiation.py```: 
      - contains methods to comput retrieve the value, gradient and jacobian for a given function during forward mode AD.
  - ```driver.py```: 
      - A class which contains methods to istantiate an AD object. It contains methods to implement forward or reverse mode AD.  It contains methods to retrieve the gradient and values by calling methods in the ```differentiation.py``` module for forward mode. 

## Broader Impacts and Inclusivity Statement

#### Broader impacts
Automatic differentiation a is widely used tool, particularly in the field of machine learning and engineering. It is especially useful tool for numerical differentiation of complex differential equations that scientists and non-scientists encounter every day across numerous fields. Continuing in the open access spirit of python, we offer an open access software to perform forward and reverse mode AD. In creating this package, we are hopping to help early career scientists with implementing numerical differentiation, which is precise and requires little computational expense. We hope that this package will serve as a jumping off point for other novice coders.

#### Inclusivity
Successful scientific pursuits hinges on principles of collaboration, honesty, and inclusivity. Research, in itself, is the pursuit of knoweldge and truth, and requires discourse and engagement among researches from diverse fields without discrimination or intolerance. Central to the goal of this project is creating an open-sourced package which is accessible and can be used widely and ubiquitously. As a diverse group of non-computer scientists, we are collectivley committed to ensuring that all users can contribute to this package to improve the quality of the existing code base. We ask that contributors  are respectful to others regardless of identity and to exercise tolerance. Pull requests will be reviewed blindly by at least two core developers. Finally, unethical and/or illegal aplications of this package will not be tolerated. 
