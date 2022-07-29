import numpy as np
import matplotlib.pyplot as plt
B=2

def gamma2(k1,k2):
    return B**2*(np.sin(k1)**2+np.sin(k2)**2)

def beta(k1,k2):
    # return B*(1/2+1/2*np.cos(k1)+1/2*np.cos(k2))
    M=-1
    return B*(M+np.cos(k1)+np.cos(k2))

def roots(k1,k2,g):
    betaVal=beta(k1,k2)
    gamma2Val=gamma2(k1,k2)
    coef=[1,-3*g,-betaVal**2+13/4*g**2-gamma2Val,betaVal**2*g-3/2*g**3+2*g*gamma2Val,-1/4*betaVal**2*g**2+1/4*g**4-g**2*gamma2Val]
    rst=np.roots(coef)
    realRoots=[]
    for oneRoot in rst:
        if np.imag(oneRoot)==0:
            realRoots.append(np.real(oneRoot))


    return realRoots

dk=0.0001
def scanRoots(g):
    k1=0
    k2Coef=np.arange(-0.1,0.1+dk,dk)
    k2All=k2Coef*np.pi
    k2Ret=[]
    rootsRet=[]
    for k2 in k2All:
        rts=roots(k1,k2,g)
        if len(rts)>0:
            for elem in rts:
                k2Ret.append(k2/np.pi)
                rootsRet.append(elem)
    return k2Ret,rootsRet

g=4
pltK2,pltRoots=scanRoots(g)



ftSize=16

plt.figure()
plt.scatter(pltK2,pltRoots,s=0.05,c="blue",label="analytic solution")
plt.xlabel("$k_{2}/\pi$",fontsize=ftSize)
plt.ylabel("$E$",fontsize=ftSize)
plt.title("$g/B=$"+str(g/B),fontsize=ftSize)
# k2less=[]
#
# k2more=[]
#
# rootsless=[]
# rootsmore=[]
#
# bar=4
#
# for j in range(0,len(pltRoots)):
#     if pltRoots[j]>bar:
#         k2more.append(pltK2[j])
#         rootsmore.append(pltRoots[j])
#     else:
#         k2less.append(pltK2[j])
#         rootsless.append(pltRoots[j])

# plt.scatter(k2less,rootsless)
# # plt.plot(k2less,rootsless)
# plt.plot(k2more,rootsmore)

#perturbative solution for |g|>2B
def EPlus(k2):
    return 1/2*g-g*B*k2/np.sqrt(g**2-4*B**2)
def EMinus(k2):
    return 1 / 2 * g + g * B * k2 / np.sqrt(g ** 2 - 4 * B ** 2)

dk2=0.005
end=0.02
smallK2Coef=np.arange(-end,end+dk2,dk2)
smallK2All=smallK2Coef*np.pi

#|g|>2B
# ESmallk2Plus=[EPlus(k2) for k2 in smallK2All]
# ESmallk2Minus=[EMinus(k2) for k2 in smallK2All]
# plt.scatter(smallK2Coef,ESmallk2Plus,color="red",s=10,label="pertubative solution")
# plt.scatter(smallK2Coef,ESmallk2Minus,color="red",s=10)
#|g|<=2B
EflatPlus=B+g
EflatMinus=-B+g
#|g|<=2B

plt.scatter([0,0],[EflatPlus,EflatMinus],color="red",s=10,label="pertubative solution")


lgnd =plt.legend(loc="best",fontsize=ftSize-2)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.savefig("./p1spectrum/g"+str(g)+".png")