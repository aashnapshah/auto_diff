## Introduction:
This software implements automatic differentiation, a computationally ergonomic method to compute first derivatives of functions. As first derivatives are the heart of Jacobian matrices, which in turn are the heart of many methods in dynamical systems and statistics including linearization techniques like Newton's Method, determination of stability of dynamic systems through observation of eigenvalues, bifurcation analysis, and nonlinear least squares regression. Systems can be modelled through explicit coding of the derivation of a model, but this method is tedious and prone to errors. Alternatively, symbolic derivation through a graph based analysis can be used to find derivatives; however, these methods aren't conducive to interpretative code environments such as Python.  This leaves numerical methods such as finite-differences and automatic differentiation to bridge the gap. Finite differences approximates derivatives from first principles, using some small value of $\epsilon$ to approximate the limit to zero; however, these systems can suffer from accuracy problems, and choosing a suitable value $\epsilon$ becomes non-trivial. The last alternative, automatic differentiation, is what this package implements. Automatic differentiation is exact, efficient, and amenable to OOP systems. This library provides a resource for automatic differentiation.

## Background:
In it's simplest form, automatic differentiation (AD) approximates the gradient for a differentiable function. Its forward mode implementation is based on the notion that a differentiable function consists of a finite set of differentiable elementary functions, which the functional form of the derivative is known. In other words, a given function can be decomposed into multiple elementary steps, which forms a primal trace. The derivative of each elementary step, or node, is computed recursively and ultimately forms the tangent trace. By using the chain rule, the derivatives of the node are combined to approximate the gradient (or jacobian for multivariate functions) for a given value.

###Forward Mode
 The chain rule is integral to AD methods. Utilizing the chain rule allows for the accumulation of the tangent trace and computation of the derivative. In general, the chain rule will be used in AD to compute the gradient of a function  $f = f(y(x))$ where $y ∈ ℝ^{n}$ and $x ∈ ℝ^{m}$:

$\triangledown_{x}f = \sum_{i=1}^{n} \frac{\partial f}{\partial y_{i} }\triangledown y_{i}(x)$

Importantly, forward mode AD computes $\triangledown f * p$, where p is a seed vector. If $f$ is scalar, forward AD computes the gradient, whereas it will compute the Jacobian matrix ($\frac{\partial f_{i}}{\partial x_{j}}$) if $f$ is a vector. In this package, forward AD is implemented using a dual numbers approach. This approach computes the value of the function and its gradient (or Jacobian) for a given input in parallel. More specifically, an input variable can be decomposed into a real and dual part:

$x = a + bε$

where $a, b$ ∈ $ℝ^{m}$, $ε^{2}=0$, and $ε≠0$. Here, a value $x$ is converted to a real part, $a$,  and dual part, $b$. Using these principles, the dual numbers method can be demonstrated using a Taylor series expansion:

$f(x) = f(a) + \frac{f'(a)}{1!}(x-a) + \frac{f''(a)}{2!}(x-a)^{2} + ...$

Because $ε^{2}=0$, this expression simplifies to:

$f(x) = f(a + bε) = f(a) + f'(a)bε$

Therefore, dual numbers computes the primal trace of the real part and the tangent trace of the dual part simultaneously. This expression also generalizes to high-order functions.
 
### Reverse Mode

Unlike forward mode, reverse AD utilizes forward pass and reverse pass in sequence. It is the preferred method when dealing with large inputs and small output, $f: \mathbb{R}^m → \mathbb{R}$ . 

#### Forward Pass

The forward pass determines the primal trace of function, and computes the partial derivatives of each node with respect to its parent node. Importantly, the forward pass does not explicitly use the chain rule, but instead computes $\partial \frac{v_j}{v_i}$, where $j$ and $i$ refers the child and parent node, respectively. The construction fo the relationship beween the parent and child nodes allows for the reconstruction of the chain rule in the reverse pass.

#### Reverse Pass

In reverse pass, the partial derivatives computed in the forward pass are accumulated. Specifically, the sensitivity of each node, represented by $\bar v$ notation, relative to its parent is computed:

$\bar v_{j-m} = \frac {\partial f_{i}}{\partial v_{j-m}}$

Thus, the partial derivatives determined in forward pass are iterated over the children of the node and ultimately accumulated. The adjoint values for all variables are computed as the computational graph is traversed backwards.


$\bar v_{i}$ = $\frac {\partial f}{\partial v_{i}} = \sum_{j} \bar v_{j} \frac {\partial v_{j}}{\partial v_{i}}$


Importantly, the deriviatives are computed with respect to intermediate variables, which is distinct from foward mode. Specifically, forward mode computes $\triangledown_{x}f$, whereas reverse mode computes $\triangledown_{v}f$.
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


**Test Suite**
  - The test suite lives in the root directory. Each module in the package is tested using pytest. Code coverage of >90% is ensured using pytest-cov package.The tests are integrated into the CI workflow, using yml extension filed linked to our Github project. We utilize github badges to monitor code coverage and test performance. 
  - We specifically configured two bash scripts, ```run_tests.sh``` and ```code_coverage.sh```, to implement pytests and code coverage. Specifically, 
  the ```run_tests.sh``` script specifies which modules to implement pytest, and the ```code_coverage.sh``` specifies a passing coverage rate for >90%. Coverage <90% will trigger a failure and will be observed in the coverage badge on the ```ReadMe.md```.
  
<!-- **Distribution of Package**
  - Our package will be distributed using PyPI   -->

## Implementation

**Core classes**

  - Our driver module functions as the interface for users to specify functions to evaluate and which mode of AD to implement. It hosts the the ```Forward_AD``` and ```Reverse_AD``` classes, which can be used to instantiate an autodiff objects for forward and reverse mode, respectively. One instantiated, the methods ```values``` and ```gradient``` can be utilized. 

  - A Dual Numbers approach is used to implement forward mode. The ```dualNum.py``` module contains a class in which a ```DualNumber``` object can be implemented. This class is ulimately used to caculate the derivative of function for a given value in forward mode AD  (see relevant methods below)

  - The ```node.py``` contains a ```Node``` class which contains methods for retrieving a value of function and storing the derivative. It also contains a method to store the relationship between the child and parent node as the computational graph is traversed during reverse mode. The ```Node``` class which calculate the sensitivity of the adjoint. 
  

**Core attributes and methods**
  - The core attributes of our ```driver.py``` module are ```values``` and ```gradient``` to retrieve the value  and compute the gradient for a given function(s) and input(s), respectively. The methods are the same for forward and reverse mode AD. 
  - The ```DualNumber``` class contains two attributes: a value and a derivative. These attributes are overloaded for the operations listed above. These overloaded operators return a ```DualNumber``` and calculate the value and derivative for a given input value. This class overloads basic numerical type of python and implements numerical operators (+, -, *, etc). In its current form, the following methods have been overloaded:
    * ```__mul__```, ```__rmul__```, ```__add__```, ```__radd__```, ```__sub__```, 
* ```__truediv__```, ```__rtruediv__```,```__pow__```, ```__rpow__```, 
*  ```__neg__```, ```__rsub__```
  
  - The ```differentiaion.py``` module integrates the ```dualNum.py``` and ```elemFunctions.py``` modules. The core methods of this module are the ```derivative``` (scalar input), ```gradient``` (for multivariate scalar input), ```jacobian``` (multivariate vector input). Additionally, the ```get_values``` method is employed in the ```driver.py``` module to retrieve values for a function.

  - The ```node.py``` module has three attributes: ```children```,  ```value```, ```gradient```. The former two attributes refer to the children of the parent nodes and the value of a given function. Using this approach, the derivatives of all the children per parent node  are summed and the derviatives are computed recursively. Similar to the Dual numbers module, ```Node``` class also overlaods basic numerical operations (see methods listed above) for reverse mode implementation.

**Elementary Functions**

- The ```elemFunctions.py``` contains the known derivatives of elementary operations using ```numpy```. This modules utilizes the ```DualNumbers``` class  and well as the ```Node``` class to overloads unary functions, which include the following functions:
    * ```sin```, ```cos```, ```tan```, ```arcsin```, ```arccos```
    * ```arctan```, ```sinh```, ```cosh```, ```tanh```, 
    * ```log```, ```logistic```, ```exp```

**Core data structures**
  - The core data strcutures in our package include matrices, vectors, lists, and tree data structures. Gradients and jacobians are stored in lists and matrices, respectively, for both forward and reverse AD. Further, inputs for forward and reverse AD support vector inputs for computation of the jacobian. Finally, for reverse mode, a tree-like data structure is implemented where the value of the node, its derivative, and relationship to the parent node is known and stored. 
  

**External dependencies**
  - This package only requires numpy for its implementation. For testing purposes for possibly contributors of this code-base, pytest and pytest-cov are needed to assess code and coverage.

## Future Work and Applications

 - Future extensions of this project include implementing second order partial derivatives in the current code-base to compute the Hessian matrix: $(H_{f})_{i,j} = \frac{\partial^2 f}{\partial x_{i} \partial x_{j}}$ The hessian matrix can be utilized in many different ways, but conventionally it applies to optimization problems where the hessian can be used to determine saddle points and local extrema of a function. Importantly, determining the hessian is an integral tool for second-order optimization of neural networks. The hessian matrix characterizes the curvature of a function, and can establish the lower bound of an optimal learning rate of a model thereby determining how fast models can be optimized. Other applications include solving second order derivatives in fluids dynamics problems (i.e Navier-Stokes equation) or exploring properties of non-linear dynamical systems. In order to implement this, we will consider applying  forward-on-forward and/or forward-on-reverse methods, where the former refers to a double application of forward AD and the latter refers to differentiation first by reverse mode followed by forward mode. 

- A second application we are interested in exploring is visualizing the computational graph which is implemented in reverse mode. This application would potentially would serve pedagogical purposes for users who are new to automatic differentiation. To implement this, we would consider integrating an open source visualization software, such as Graphviz, into our package in order to construct computation graph visualiations . 
 
## Broader Impacts and Inclusivity Statement

####Broader impacts
Automatic differentiation a is widely used tool, particularly in the field of machine learning and engineering. It is especially useful tool for numerical differentiation of complex differential equations that scientists and non-scientists encounter every day across numerous fields. Continuing in the open access spirit of python, we offer an open access software to perform forward and reverse mode AD. In creating this package, we are hopping to help early career scientists with implementing numerical differentiation, which is precise and requires little computational expense. We hope that this package will serve as a jumping off point for other novice coders.

####Inclusivity
Successful scientific pursuits hinges on principles of collaboration, honesty, and inclusivity. Research, in itself, is the pursuit of knoweldge and truth, and requires discourse and engagement among researches from diverse fields without discrimination or intolerance. Central to the goal of this project is creating an open-sourced package which is accessible and can be used widely and ubiquitously. As a diverse group of non-computer scientists, we are collectivley committed to ensuring that all users can contribute to this package to improve the quality of the existing code base. We ask that contributors  are respectful to others regardless of identity and to exercise tolerance. Pull requests will be reviewed blindly by at least two core developers. Finally, unethical and/or illegal aplications of this package will not be tolerated. 

## Licensing 
We use the MIT License for this package. We want a permissive license allowing the distribution of our package without frills or profits, and in a simple, familiar, and straightforward manner because we want it to be as accessible for use and modification as possible; thus, the MIT License makes the most sense, as it is permissive, brief, and the most popular open source license today.

As our only dependencies is numpy, which has a BSD licenses, we do not have to consider other copyright. Others may freely advertise and monetize software that makes use of our package, but users of our software need to include the MIT License in their use of our code. The copyright holder is Harvard University, as this software was produced in a Harvard University course setting.

It can be found in the LICENSE file in our root directory.
