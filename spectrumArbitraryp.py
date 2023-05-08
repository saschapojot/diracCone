import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import expm
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
from scipy.optimize import root

#this script computes numerical solution for full f(x)=0, k1=0, -0.1<=k2/pi<=0.1
B=2

def beta(k1Val,k2Val):
    """

    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :return: value of beta
    """
    return B*(-1+np.cos(k1Val)+np.cos(k2Val))


def gamma2(k1,k2):
    """

    :param k1: quasimomentum k1
    :param k2: quasimomentum k2
    :return: value of gamma
    """
    return B**2*(np.sin(k1)**2+np.sin(k2)**2)


def solution(p,k1Val,k2Val,gVal):
    """

    :param p: order of nonlinearity
    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :param gVal: g, strength of nonlinearity
    :return:  a list of values of x. If there is no solution, the returned list is empty
    """

    betaVal=beta(k1Val,k2Val)
    gamma2Val=gamma2(k1Val,k2Val)

    def f(x):
        """

        :param x: central quantity
        :return: equation satisfied by x
        """
        #eqn of x
        return np.abs(betaVal**2-(betaVal**2+gamma2Val)*x**2+2**(-p)*betaVal*gVal*x**2*(1-x)**p-2**(-p)*betaVal*gVal*x**2*(1+x)**p\
    -2**(-p)*betaVal*gVal*(1-x)**p+2**(-p)*betaVal*gVal*(1+x)**p+2**(-2*p-1)*gVal**2*x**2*(1-x**2)**p\
    -2**(-2*p-1)*gVal**2*(1-x**2)**p-2**(-2*p-2)*gVal**2*x**2*(1-x)**(2*p)\
    -2**(-2*p-2)*gVal**2*x**2*(1+x)**(2*p)+2**(-2*p-2)*gVal**2*(1-x)**(2*p)\
    +2**(-2*p-2)*gVal**2*(1+x)**(2*p))**2




    dx=1e-2
    # there may be multiple solutions to f(x)=0, therefore we have to start with multiple initial values x0
    scanX=np.linspace(-1+dx,1-dx,int(2/dx)) # an array of starting values x0
    solutionSet=set()# containing solutions to f(x)=0 as a set
    for x0 in scanX:
        sol=root(fun=f,x0=x0,method="hybr",tol=1e-15)
        success=sol.success# if the above numerical procesure is successful
        funVal=sol.fun[0]
        # if success is True
        if success and np.abs(funVal)<1e-10:
            solutionSet.add(round(sol.x[0],8))#round the solution to the first 8 decimals, add this truncated value to the solutionSet. Different initial values of x0 may lead to repeated solutions, this repeatition is eliminated by set.

    return list(solutionSet)#return the values of x as a list



def x2E(p,betaVal,gVal,x):
    """

    :param p: order of nonlinearity
    :param betaVal: beta
    :param gVal: g, strength of nonlinearity
    :param x: central quantity
    :return: eigenenergy
    """
    return betaVal/x+gVal*(1/2+1/2*x)**(p+1)/x-gVal*(1/2-1/2*x)**(p+1)/x






########################################################################################################
#computation starts here


p=2.5

def scanFullNumericalRoots(gk2):
    """

    :param gk2: list [g, k2]. g is strength of nonlinearity. k2 is quasimomentum
    :return: [k2/pi, a list of eigenenergies].
    """
    k1=0
    g,k2=gk2
    xs=solution(p,k1,k2,g)
    if len(xs)==0:
        return [k2/np.pi,[]]
    betaVal=beta(k1,k2)
    Es=[x2E(p,betaVal,g,x) for x in xs]
    return [k2/np.pi,Es]





g=2*B




dk2Full=0.0001#step length of k2/pi
k2Full=np.arange(-0.1,0.1+dk2Full,dk2Full)*np.pi#all values of k2
inFull=[[g,k2] for k2 in k2Full]
tFullStart=datetime.now()#start time
procNum=48#parallel processes number
pool0=Pool(procNum)
ret=pool0.map(scanFullNumericalRoots,inFull)#parallel computations

tFullEnd=datetime.now()#end time
print("full time: ",tFullEnd-tFullStart)

#data retrieval
pltFullk2=[]
pltFullE=[]
for item in ret:
    k2,Es=item
    if len(Es)==0:
        continue
    for elem in Es:
        pltFullk2.append(k2)
        pltFullE.append(elem)


outDir="./spectrump"+str(p)+"/"#output directory

pltFullE=np.array(pltFullE)
Path(outDir).mkdir(exist_ok=True,parents=True)
outData=np.array([pltFullk2,pltFullE]).T

pdOut=pd.DataFrame(data=outData,columns=["k","E"])
pdOut.to_csv(outDir+"g"+str(g/B)+"B.csv",index=False)#data to csv

