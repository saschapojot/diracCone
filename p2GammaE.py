from sympy import *

g,beta=symbols("g,beta",cls=Symbol)

x=-2*beta/g
E=(Rational(1,4)*g*x**3+beta)/x+Rational(3,4)*g

gamma2=E**2-Rational(1,2)*g*E+Rational(1,16)*g**2-beta**2-(Rational(1,4)*E*g+Rational(5,16)*g**2)*x**2-Rational(5,4)*g*beta*x


pprint(gamma2.expand())
pprint(E.expand())