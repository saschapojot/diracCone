import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import root
from datetime import datetime
from multiprocessing import Pool
#this script computes spectrum for arbitrary p


B=2

def gamma2(k1,k2):
    return B**2*(np.sin(k1)**2+np.sin(k2)**2)
def gamma(k1Val,k2Val):
    return B*(np.sin(k1Val)-1j*np.sin(k2Val))
def beta(k1,k2):
    # return B*(1/2+1/2*np.cos(k1)+1/2*np.cos(k2))
    M=-1
    return B*(M+np.cos(k1)+np.cos(k2))


def solution(p,k1Val,k2Val,gVal):
    #solve x
    betaVal=beta(k1Val,k2Val)
    gamma2Val=gamma2(k1Val,k2Val)

    def f(x):
        #eqn of x
        return betaVal**2-(betaVal**2+gamma2Val)*x**2+2**(-p)*betaVal*gVal*x**2*(1-x)**p-2**(-p)*betaVal*gVal*x**2*(1+x)**p\
    -2**(-p)*betaVal*gVal*(1-x)**p+2**(-p)*betaVal*gVal*(1+x)**p+2**(-2*p-1)*gVal**2*x**2*(1-x**2)**p\
    -2**(-2*p-1)*gVal**2*(1-x**2)**p-2**(-2*p-2)*gVal**2*x**2*(1-x)**(2*p)\
    -2**(-2*p-2)*gVal**2*x**2*(1+x)**(2*p)+2**(-2*p-2)*gVal**2*(1-x)**(2*p)\
    +2**(-2*p-2)*gVal**2*(1+x)**(2*p)


    def jac(x):
        #derivative of eqn of x
        return -2*(betaVal**2+gamma2Val)*x-2**(-p)*betaVal*gVal*p*x**2*(1+x)**(p-1)-2**(-p)*betaVal*gVal*p*x**2*(1-x)**(p-1)\
    +2**(-p)*betaVal*gVal*p*(1+x)**(p-1)+2**(-p)*betaVal*gVal*p*(1-x)**(p-1)+2**(1-p)*betaVal*gVal*x*(1-x)**p\
    -2**(1-p)*betaVal*gVal*x*(1+x)**p-2**(-2*p+1)*gVal**2*p*x**3*(1-x**2)**(p-1)\
    -2**(-2*p-1)*gVal**2*p*x**2*(1-x)**p*(1+x)**(p-1)+2**(-2*p-1)*gVal**2*p*x**2*(1-x)**(p-1)*(1+x)**p\
    -2**(-2*p-1)*gVal**2*p*(1-x)**p*(1+x)**(p-1)+2**(-2*p-1)*gVal**2*p*(1-x)**(p-1)*(1+x)**p\
    +2**(1-2*p)*gVal**2*x*(1-x**2)**p-2**(-2*p)*gVal**2*x*(1-x)**p*(1+x)**p\
    -2**(-2*p-1)*gVal**2*p*x**2*(1+x)**(2*p-1)+2**(-2*p-1)*gVal**2*p*x**2*(1-x)**(2*p-1)\
    +2**(-2*p-1)*gVal**2*p*(1+x)**(2*p-1)-2**(-2*p-1)*gVal**2*p*(1-x)**(2*p-1)\
    -2**(-2*p-1)*gVal**2*x*(1-x)**(2*p)-2**(-2*p-1)*gVal**2*x*(1+x)**(2*p)

    dx=1e-3
    scanX=np.linspace(-1+dx,1-dx,int(2/dx))
    solutionSet=set()
    for x0 in scanX:
        sol=root(fun=f,x0=x0,jac=jac,method="hybr",tol=1e-10)
        success=sol.success
        funVal=sol.fun[0]
        if success and np.abs(funVal)<1e-10:
            solutionSet.add(round(sol.x[0],8))

    return list(solutionSet)



def x2E(p,betaVal,gVal,x):
    return betaVal/x+gVal*(1/2+1/2*x)**(p+1)/x-gVal*(1/2-1/2*x)**(p+1)/x





gVal=-2.5*B
p=0
k1=0
k2ValsAll=list(np.linspace(-0.1*np.pi,0.1*np.pi,100))
# k2ValsAll.append(0)
pltk2=[]
pltE=[]
# tSolutionStart=datetime.now()

def solutionE(k2):
    xs = solution(p, k1, k2, gVal)
    betaVal = beta(k1, k2)
    if len(xs)>0:
        retE=[x2E(p,betaVal,gVal,x) for x in xs]
        return [k2,retE]
# for k2 in k2ValsAll:
#     xs=solution(p,k1,k2,gVal)
#     betaVal=beta(k1,k2)
#     if len(xs)>0:
#         for x in xs:
#             pltk2.append(k2/np.pi)
#             pltE.append(x2E(p,betaVal,gVal,x))

procNum=24
pool0=Pool(procNum)
tParallelStart=datetime.now()
retk2AndEs=pool0.map(solutionE,k2ValsAll)
tParallelEnd=datetime.now()
print("parallel time: ",tParallelEnd-tParallelStart)
for elem in retk2AndEs:
    k2=elem[0]
    for E in elem[1]:
        pltk2.append(k2/np.pi)
        pltE.append(E)

pltE=np.array(pltE)
# tSolutionEnd=datetime.now()
# print("solution time: ",tSolutionEnd-tSolutionStart)
outDir="./arbitraryp"+str(p)+"/"
Path(outDir).mkdir(parents=True,exist_ok=True)
ftSize=17
plt.figure()
plt.scatter(pltk2,pltE/B,s=1,c="blue",label="numerical solution")
plt.xlabel("$k_{2}/\pi$")
plt.ylabel("$E/B$",fontsize=ftSize)
plt.title("$g/B=$"+str(gVal/B)+", $p=$"+str(p),fontsize=ftSize)
plt.savefig(outDir+"gOverB"+str(gVal/B)+"p"+str(p)+".png")