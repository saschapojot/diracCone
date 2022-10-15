from sympy import *

#this script computes formal equation of x

p,beta,g,gamma,x=symbols("p,beta,g,gamma,x",cls=Symbol,real=True)

half=Rational(1,2)
quarter=Rational(1,4)

E=beta/x+g*(half+half*x)**(p+1)/x-g*(half-half*x)**(p+1)/x


eqn=-gamma**2+E**2-g*E*((half+half*x)**p+(half-half*x)**p)-beta**2\
    -g*beta*((half+half*x)**p-(half-half*x)**p)+g**2*(quarter-quarter*x**2)**p

rst=(x**2*eqn).expand()

typeInf=beta**2-(beta**2+gamma**2)*x**2+2**(-p)*beta*g*x**2*(1-x)**p-2**(-p)*beta*g*x**2*(1+x)**p\
    -2**(-p)*beta*g*(1-x)**p+2**(-p)*beta*g*(1+x)**p+2**(-2*p-1)*g**2*x**2*(1-x**2)**p\
    -2**(-2*p-1)*g**2*(1-x**2)**p-2**(-2*p-2)*g**2*x**2*(1-x)**(2*p)\
    -2**(-2*p-2)*g**2*x**2*(1+x)**(2*p)+2**(-2*p-2)*g**2*(1-x)**(2*p)\
    +2**(-2*p-2)*g**2*(1+x)**(2*p)



df=diff(rst,x)


typeIndf=-2*(beta**2+gamma**2)*x-2**(-p)*beta*g*p*x**2*(1+x)**(p-1)-2**(-p)*beta*g*p*x**2*(1-x)**(p-1)\
    +2**(-p)*beta*g*p*(1+x)**(p-1)+2**(-p)*beta*g*p*(1-x)**(p-1)+2**(1-p)*beta*g*x*(1-x)**p\
    -2**(1-p)*beta*g*x*(1+x)**p-2**(-2*p+1)*g**2*p*x**3*(1-x**2)**(p-1)\
    -2**(-2*p-1)*g**2*p*x**2*(1-x)**p*(1+x)**(p-1)+2**(-2*p-1)*g**2*p*x**2*(1-x)**(p-1)*(1+x)**p\
    -2**(-2*p-1)*g**2*p*(1-x)**p*(1+x)**(p-1)+2**(-2*p-1)*g**2*p*(1-x)**(p-1)*(1+x)**p\
    +2**(1-2*p)*g**2*x*(1-x**2)**p-2**(-2*p)*g**2*x*(1-x)**p*(1+x)**p\
    -2**(-2*p-1)*g**2*p*x**2*(1+x)**(2*p-1)+2**(-2*p-1)*g**2*p*x**2*(1-x)**(2*p-1)\
    +2**(-2*p-1)*g**2*p*(1+x)**(2*p-1)-2**(-2*p-1)*g**2*p*(1-x)**(2*p-1)\
    -2**(-2*p-1)*g**2*x*(1-x)**(2*p)-2**(-2*p-1)*g**2*x*(1+x)**(2*p)



