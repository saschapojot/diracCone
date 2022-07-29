from sympy import *


x,g,beta,absGamma=symbols("x,g,beta,r",cls=Symbol)

eqn=-x**2*absGamma**2+(Rational(1,4)*g*x**3+beta+Rational(3,4)*g*x)**2\
    -Rational(1,2)*g*(Rational(1,4)*g*x**4+beta*x+Rational(3,4)*g*x**2)\
    +(Rational(1,16)*g**2-beta**2)*x**2\
    -(Rational(1,4)*g*(Rational(1,4)*g*x**4+beta*x+Rational(3,4)*g*x**2)+Rational(5,16)*g**2*x**2)*x**2\
    -Rational(5,4)*beta*g*x**3


pprint(-eqn.expand())