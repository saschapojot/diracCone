import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root
from datetime import datetime
from multiprocessing import Pool
#plot phases, theoretical vs numerical in 1 plot

B=2
pVals=[0.5,1,1.5,2,2.5,3]#p values whose phases have already been computed
inDataAll=[]# a list of pandas dataframes
for p in pVals:
    inDataAll.append(pd.read_csv("./arbitraryp"+str(p)+"/truep"+str(p)+".csv"))#read csv
def g2x(g,p):
    """
    solve xi(x0)=0
    :param g: strength of nonlinearity
    if |g|>2B, solve equation xi(x0)=0 to obtain the degenerate root
    if 0<g <=2B, let x0=-1;
    if -2B<=g<0, let x0=1;
    :param p: order of nonlinearity
    :return: x0, central quantity at the origin
    """
    if np.abs(g)>2*B:
        def f(x):
            return np.abs((1 + x) ** p - (1 - x) ** p + 2 ** (1 + p) * B / g) ** 2

        sol = root(fun=f, x0=0, method="hybr", tol=1e-15)
        return sol.x[0]
    elif g>= 0 and g<=2*B:
        return -1
    else:
        return 1


def theoreticalPhases(g,p):
    """

    :param g: strength of nonlinearity, |g|>2B
    :param p: order of nonlinearity
    :return: theoretical value of dynamical, Berry and AB phases in terms of pi
    """
    if np.abs(g)>2*B:
        x0 = g2x(g,p)
        # E0 = B / x0 + g / 2 ** (p + 1) * ((1 + x0) ** (p + 1) - (1 - x0) ** (p + 1)) / x0
        # D1 = E0 + g * p / 2 ** p * (1 + x0) ** (p + 1) - B - g / 2 ** p * (2 * p + 1) * (1 + x0) ** p - g * p / 2 ** p * (
        #             1 + x0) * (1 - x0) ** p
        # tD = -2 * B * p * (1 - x0 ** 2) / D1
        tD = -1*((1+x0)**p-(1-x0)**p)/((1+x0)**(p-1)+(1-x0)**(p-1))
        tB=-1+x0

        AB=tD+tB

        return tD, tB,AB
    else:
        return 0,-2,-2

dg=0.0001
gLeft=np.linspace(-10*B,-(2+dg)*B,1000)#g values <-2B
gMiddle=np.linspace(-(2-dg)*B,(2-dg)*B,100) # -2B < g values < 2B
gRight=np.linspace((2+dg)*B,10*B,1000)# g values > 2B
gAll=np.append(gLeft,gMiddle)
gAll=np.append(gAll,gRight)# all g
gLeftInd=list(range(0,len(gLeft)))#indices of g in gAll, g< -2B
gMiddleInd=list(range(len(gLeft),len(gLeft)+len(gMiddle)))#indices of g in gAll, 2B< g< -2B
gRightInd=list(range(len(gLeft)+len(gMiddle),len(gLeft)+len(gMiddle)+len(gRight)))#indices of g in gAll, g>2B

# pVals=[0.5,1,1.5,2,2.5,3]


tDTensor=np.zeros((len(gAll),len(pVals)))
tBTensor=np.zeros((len(gAll),len(pVals)))
ABTensor=np.zeros((len(gAll),len(pVals)))

def wrapper(nj):
    """

    :param nj: [n,j]
    :return: [n,j,dynamical phase, Berry phase,AB phase]
    """
    # n is index of g
    # j is index of p
    n,j=nj

    g=gAll[n]
    p=pVals[j]
    tD,tB,AB=theoreticalPhases(g,p)
    return [n,j,tD,tB,AB]


inIndices=[[n,j] for n in range(0,len(gAll)) for j in range(0,len(pVals))]

tPhasesStart=datetime.now()
procNum=48
pool0=Pool(procNum)

ret=pool0.map(wrapper,inIndices)#parallel computations

tPhasesEnd=datetime.now()

print("theoretical phases time: ",tPhasesEnd-tPhasesStart)


#data retrieval
for item in ret:
    n,j,tD,tB,AB=item
    tDTensor[n,j]=tD
    tBTensor[n,j]=tB
    ABTensor[n,j]=AB


# pVals=[0.5,1,1.5,2,2.5,3]
colors=["blue","red","limegreen","fuchsia","darkorange","dimgrey"]
x0=-0.1
y0=1.1
fig=plt.figure(figsize=plt.figaspect(3))
plt.rcParams['figure.constrained_layout.use'] = True
#dynamical phase
ax1=fig.add_subplot(3,1,1)
for j in range(0,len(pVals)):
    ax1.plot(gAll / B, tDTensor[:, j], color=colors[j], label="p=" + str(pVals[j]))#plot theoretical values of dynamcal phase
    lenTmp = len(np.array(inDataAll[j].iloc[:, 0]))
    selected = list(range(0, lenTmp, 10))
    ax1.scatter(np.array(inDataAll[j].iloc[selected, 0]) / B, np.array(inDataAll[j].iloc[selected, 1]) / np.pi,
                color=colors[j], s=4)#plot values of dynamcal phase from numerical evolution

ftSize=17
ax1.set_xticks(np.arange(-10,11,2))
ax1.set_xlim((-10,10))
ax1.axhline(y=2, color='black', linestyle='--')
ax1.axhline(y=-2, color='black', linestyle='--')
ax1.axhline(y=1, color='gold', linestyle='--')
ax1.axhline(y=-1, color='gold', linestyle='--')
ax1.legend(loc="best")
ax1.set_xlabel("$g/B$", fontsize=ftSize)
ax1.set_ylabel("$\delta\\theta_{D}/\pi$",fontsize=ftSize)
ax1.set_yticks([-2,-1,0,1,2])
ax1.text(x0, y0, "(a)", transform=ax1.transAxes,
            size=ftSize-2)
####

ax2=fig.add_subplot(3,1,2)

for j in range(0,len(pVals)):
    # plt.plot(gAll/B,tBTensor[:,j],color=colors[j],label="p="+str(pVals[j]))
    ax2.plot(gAll[gLeftInd] / B, tBTensor[gLeftInd, j] , color=colors[j], label="p=" + str(pVals[j]))# plot theoretical values of Berry phase for g< -2B
    ax2.plot(gAll[gMiddleInd]/B,tBTensor[gMiddleInd,j]+2,color=colors[j])# plot theoretical values of Berry phase for  -2B<g< 2B
    ax2.plot(gAll[gRightInd]/B,tBTensor[gRightInd,j]+2,color=colors[j])# plot theoretical values of Berry phase for g>2B


def shiftBerry(phaseVec):
    """

    :param phaseVec: a vector containing Berry phases
    :return: Berry phase shifted to [-pi, pi]
    """
    def tmp(phase):
        if phase>1:
            phase-=2
        elif phase<-1:
            phase+=2
        return phase
    retVec=[tmp(phase) for phase in phaseVec]
    return retVec


for j in range(0,len(pVals)):
    lenTmp = len(np.array(inDataAll[j].iloc[:, 0]))
    selected = list(range(0, lenTmp, 10))
    berrySelected=np.array(inDataAll[j].iloc[selected, 2]) / np.pi-np.array(inDataAll[j].iloc[selected, 1]) / np.pi
    ax2.scatter(np.array(inDataAll[j].iloc[selected, 0]) / B, shiftBerry(berrySelected),
                color=colors[j], s=4)#plot Berry phase from numerical evolution

ax2.set_xticks(np.arange(-10,11,2))
ax2.set_xlim((-10,10))

ax2.legend(loc="best")
ax2.set_xlabel("$g/B$", fontsize=ftSize)
ax2.set_ylabel("$\\theta_{B}/\pi$",fontsize=ftSize)
ax2.text(x0, y0, "(b)", transform=ax2.transAxes,
            size=ftSize-2)











###plt AB phases
ax3=fig.add_subplot(3,1,3)
j1=1
for j in range(0,len(pVals)):
    # plt.plot(gAll/B,ABTensor[:,j],color=colors[j],label="p="+str(pVals[j]))
    #left
    if j==1:
        # continue
        ax3.plot(gAll / B, ABTensor[:, j1] + 2, color=colors[j1], label="p=" + str(pVals[j1]))#plot theoretical values of  AB phase for p=1
        continue
    if j==0:
        shiftLeft=0
    else:
        shiftLeft=2
    ax3.plot(gAll[gLeftInd] / B, ABTensor[gLeftInd, j] + shiftLeft, color=colors[j], label="p=" + str(pVals[j]))#plot theoretical values of  AB phase, g<-2B
    #middle
    shiftMiddle=2
    ax3.plot(gAll[gMiddleInd]/B,ABTensor[gMiddleInd,j]+shiftMiddle, color=colors[j])#plot theoretical values of  AB phase, -2B<g<2B
    #right
    if j==0:
        shiftRight=2
    else:
        shiftRight=0
    ax3.plot(gAll[gRightInd]/B,ABTensor[gRightInd,j]+shiftRight,color=colors[j])#plot theoretical values of  AB phase, g>2B

###p1=1
# j1=1
# ax3.plot(gAll/B,ABTensor[:,j1]+2,color=colors[j1],label="p="+str(pVals[j1]))
for j in range(0,len(pVals)):
    if j==1:
        continue
    lenTmp = len(np.array(inDataAll[j].iloc[:, 0]))
    selected = list(range(0, lenTmp, 10))
    ax3.scatter(np.array(inDataAll[j].iloc[selected, 0]) / B, np.array(inDataAll[j].iloc[selected, 2]) / np.pi,
                color=colors[j], s=4)#plot AB phase from numerical evolution, p!=1

lenTmp = len(np.array(inDataAll[j1].iloc[:, 0]))
selected = list(range(0, lenTmp, 10))
def shiftPhaseForp1(phaseVec):
    """

    :param phaseVec: a vector containing AB phases
    :return: AB shifted to [-pi, pi]
    """
    def tmp(phase):
        if phase<-0.5:
            phase+=2
        return phase
    retVec=[tmp(phase) for phase in phaseVec]
    return retVec

ax3.scatter(np.array(inDataAll[j1].iloc[selected, 0]) / B, shiftPhaseForp1(np.array(inDataAll[j1].iloc[selected, 2]) / np.pi),
                color=colors[j1], s=4)# plot  AB phase from numerical evolutions for p=1
ax3.set_xticks(np.arange(-10,11,2))
ax3.set_xlim((-10,10))

ax3.legend(loc="best")
ax3.set_xlabel("$g/B$", fontsize=ftSize)
ax3.set_ylabel("$\\theta_{AB}/\pi$",fontsize=ftSize)
ax3.set_yticks([-2,-1,0,1,2])
ax3.text(x0, y0, "(c)", transform=ax3.transAxes,
            size=ftSize-2)





plt.savefig("phases3Col.png")
plt.close()