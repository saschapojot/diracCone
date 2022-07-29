from sympy import *

E,g,bt,gm=symbols("E,g,beta,gamma",cls=Symbol)

eqn=E**2-g*E-bt**2-bt**2*g/(E-g)+g**2/4*(1-bt**2/(E-g)**2)-gm**2

eqn1=(E-g)**2*eqn

pprint(eqn1.expand().simplify())