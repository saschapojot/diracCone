from sympy import *
#this script computes partial derivative of E0 and D1
from scipy.optimize import root
import  numpy as np
B,g,p,x0=symbols("B,g,p,x0",cls=Symbol)
p=2
B=2
g=2**(1+p)*B/((1-x0)**p-(1+x0)**p)



# p=3
# B=2
# g=2.3*B
# def f(x):
#     return np.abs((1+x)**p-(1-x)**p+2**(1+p)*B/g)**2


# sol0=root(fun=f,x0=0,method="hybr",tol=1e-15)
# x0=sol0.x[0]
E0=B/x0+g/2**(p+1)*((1+x0)**(p+1)-(1-x0)**(p+1))/x0

D1=E0+g*p/2**p*(1+x0)**(p+1)-B-g/2**p*(2*p+1)*(1+x0)**p-g*p/2**p*(1+x0)*(1-x0)**p



# dx0E0=diff(E0,x0)
#
dx0D1=diff(D1,x0)







# td=-4*B*p/dx0D1
#
# td=td.subs(x0,-1+1e-15)
#
# pprint(td.evalf())

###########################Sj and Rj terms

###Sj terms
S1=B*p*((1+x0)**(p-1)+(1-x0)**(p-1))*(-(1-x0)**(p+1)+(1+x0)**(p+1))/(x0*((1-x0)**p-(1+x0)**p)**2)

S2=2*B*p**2*(1-x0)**(p-1)*(1+x0)/((1-x0)**p-(1+x0)**p)

S3=-2*B*p*(2*p+1)*(1+x0)**(p-1)/((1-x0)**p-(1+x0)**p)

S4=2*B*p**2*((1+x0)**(2*p)+(1+x0)**(p+1)*(1-x0)**(p-1))/((1-x0)**p-(1+x0)**p)**2

S5=-2*B*p*(2*p+1)*((1+x0)**(2*p-1)+(1+x0)**p*(1-x0)**(p-1))/((1-x0)**p-(1+x0)**p)**2

S6=-2*B*p**2*((1-x0)**p*(1+x0)**p+(1-x0)**(2*p-1)*(1+x0))/((1-x0)**p-(1+x0)**p)**2

###Rj terms

R1=B*(p+1)*((1+x0)**p+(1-x0)**p)/(x0*((1-x0)**p-(1+x0)**p))

R2=-B*(-(1-x0)**(p+1)+(1+x0)**(p+1))/(x0**2*((1-x0)**p-(1+x0)**p))

R3=-B/x0**2

R4=-2*B*p*(1-x0)**p/((1-x0)**p-(1+x0)**p)

R5=2*B*p*(p+1)*(1+x0)**p/((1-x0)**p-(1+x0)**p)


