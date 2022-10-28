from sympy import *
#this script computes partial derivative of E0 and D1


B,g,p,x0=symbols("B,g,p,x0",cls=Symbol)

E0=B/x0+g/2**(p+1)*((1+x0)**(p+1)-(1-x0)**(p+1))/x0

D1=E0+g*p/2**p*(1+x0)**(p+1)-B-g/2**p*(2*p+1)*(1+x0)**p-g*p/2**p*(1+x0)*(1-x0)**p

td=-2*B*pi*p*(1-x0**2)/D1

# dx0E0=diff(E0,x0)
#
dx0D1=diff(D1,x0)



