from sympy import *

#this script computes equation for E when p=2

g,x,beta=symbols("g,x,beta",cls=Symbol)

eqn=(2*beta+g*x+Rational(1,4)*g*x**3)*(Rational(1,2)*g*x**3-beta)\
    -(Rational(1,2)*g*x*(Rational(1,4)*g*x**3+beta+Rational(3,4)*g*x)+Rational(5,8)*g**2*x**2+Rational(5,4)*g*beta*x)*x**2


pprint(factor(eqn))
