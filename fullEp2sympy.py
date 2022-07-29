from sympy import *


g,x,beta,gm=symbols("g,x,beta,gamma",cls=Symbol)

E=(Rational(1,4)*g*x**3+beta)/x+Rational(3,4)*g

eqn=-gm**2+E**2-Rational(1,2)*g*E+Rational(1,16)*g**2-beta**2-(Rational(1,4)*g*E+Rational(5,16)*g**2)*x**2\
    -Rational(5,4)*g*beta*x

rst=eqn*x**2

pprint(-rst.expand().simplify())