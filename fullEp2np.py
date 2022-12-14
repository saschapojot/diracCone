import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

B=2

def gamma2(k1,k2):
    return B ** 2 * (np.sin(k1) ** 2 + np.sin(k2) ** 2)


def beta(k1,k2):
    # return B*(1/2+1/2*np.cos(k1)+1/2*np.cos(k2))
    M=-1
    return B*(M+np.cos(k1)+np.cos(k2))


def roots(k1,k2,g):
    betaVal=beta(k1,k2)
    gamma2Val=gamma2(k1,k2)
    coef=[g**2,4*betaVal*g,4*gamma2Val+4*betaVal**2-g**2,-4*betaVal*g,-4*betaVal**2]
    rst = np.roots(coef)
    realRoots = []
    for oneRoot in rst:
        if np.imag(oneRoot) == 0:
            realRoots.append(np.real(oneRoot))

    EVals=[]
    for x in realRoots:
        EVals.append((1/4*g*x**3+betaVal)/x+3/4*g)
    return EVals


dk=0.0001

def scanRoots(g):
    k1=0
    k2Coef = np.arange(-0.1, 0.1 + dk, dk)
    k2All = k2Coef * np.pi
    k2Ret = []
    rootsRet = []
    for k2 in k2All:
        rts=roots(k1,k2,g)
        if len(rts)>0:
            for elem in rts:
                k2Ret.append(k2/np.pi)
                rootsRet.append(elem)
    return np.array(k2Ret),np.array(rootsRet)


g=5
pltK2,pltRoots=scanRoots(g)

ftSize=16

plt.figure()
plt.scatter(pltK2,pltRoots/B,s=0.05,c="blue",label="numerical solution")
plt.xlabel("$k_{2}/\pi$",fontsize=ftSize)
plt.ylabel("$E/B$",fontsize=ftSize)
plt.title("$g/B=$"+str(g/B),fontsize=ftSize)

#perturbative solution for |g|>2B
def EPlus(k2):
    return B**2/g+1/4*g+(4*B**2+g**2)/(g*np.sqrt(g**2-4*B**2))*B*k2
def EMinus(k2):
    return B ** 2 / g + 1 / 4 * g - (4 * B ** 2 + g ** 2) / (g * np.sqrt(g ** 2 - 4 * B ** 2)) * B * k2





dk2=0.005
end=0.02
smallK2Coef=np.arange(-end,end+dk2,dk2)
smallK2All=smallK2Coef*np.pi
#|g|>2B
ESmallk2Plus=[EPlus(k2)/B for k2 in smallK2All]
ESmallk2Minus=[EMinus(k2)/B for k2 in smallK2All]
plt.scatter(smallK2Coef,ESmallk2Plus,color="red",s=10,label="pertubative solution")
plt.scatter(smallK2Coef,ESmallk2Minus,color="red",s=10)


#|g|<=2B
EflatPlus=(B+g)/B
EflatMinus=(-B+g)/B

# plt.scatter(0,EflatPlus,color="red",s=10,label="pertubative solution")
#
# plt.scatter(0,EflatMinus,color="red",s=10)



lgnd =plt.legend(loc="best",fontsize=ftSize-2)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
outDir="./p2spectrum/"
Path(outDir).mkdir(parents=True, exist_ok=True)
plt.savefig(outDir+"g"+str(g)+".png")