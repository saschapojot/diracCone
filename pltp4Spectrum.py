import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# this script computes spectrum for p=4

B=2
p=4
def gamma2(k1,k2):
    return B**2*(np.sin(k1)**2+np.sin(k2)**2)
def gamma(k1Val,k2Val):
    return B*(np.sin(k1Val)-1j*np.sin(k2Val))
def beta(k1,k2):
    # return B*(1/2+1/2*np.cos(k1)+1/2*np.cos(k2))
    M=-1
    return B*(M+np.cos(k1)+np.cos(k2))



def roots(k1,k2,gVal):
    #solve x
    betaVal=beta(k1,k2)
    gammaVal=gamma(k1,k2)
    coef=[gVal**2,0,gVal**2,8*betaVal*gVal,-gVal**2,0,16*betaVal**2+16*np.abs(gammaVal)**2-gVal**2\
          ,-8*betaVal*gVal,-16*betaVal**2]
    rst = np.roots(coef)
    realRoots = []
    for oneRoot in rst:
        if np.abs(np.imag(oneRoot))<1e-8 and np.abs(np.real(oneRoot))<=1:
            realRoots.append(np.real(oneRoot))

    return realRoots



def x2E(x,k1,k2,gVal):
    betaVal = beta(k1, k2)
    E=betaVal/x+1/16*gVal*x**4+5/8*gVal*x**2+5/16*gVal
    return E


def scanRoots(g):
    #solve E
    k1=0
    k2ValsAll=np.linspace(-0.1*np.pi,0.1*np.pi,10000)
    k2Ret=[]
    rootsRet=[]
    for k2 in k2ValsAll:
        rts=roots(k1,k2,g)
        if len(rts)>0:
            for elem in rts:
                k2Ret.append(k2/np.pi)
                rootsRet.append(x2E(elem,k1,k2,g))

    return np.array(k2Ret),np.array(rootsRet)



g=-2.5*B
pltK2,pltRoots=scanRoots(g)
ftSize=16
plt.figure()
plt.scatter(pltK2,pltRoots/B,s=0.05,c="blue",label="numerical solution")
plt.ylabel("$E/B$",fontsize=ftSize)
plt.title("$g/B=$"+str(g/B),fontsize=ftSize)

outDir="./p"+str(p)+"spectrum/"
Path(outDir).mkdir(parents=True,exist_ok=True)
plt.savefig(outDir+"gOverB"+str(g/B)+".png")