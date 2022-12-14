from sympy import *

x,beta,g,gm=symbols("x,beta,g,gamma",cls=Symbol)
gm=0
E=beta/x+g/2+g*x**2/2

eqn=-gm**2+E**2-g*E*(Rational(1,4)+Rational(3,4)*x**2)-beta**2\
    -beta*g*(Rational(3,4)*x+Rational(1,4)*x**3)\
    +g**2*(-Rational(1,64)*x**6+Rational(3,64)*x**4-Rational(3,64)*x**2+Rational(1,64))

rst=eqn*x**2

pprint(factor(-rst.expand().simplify()))