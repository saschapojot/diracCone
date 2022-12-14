import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
#plot phases, theoretical vs numerical in 3 plots for arxiv paper

B=2
pVals=[0.5,1,1.5,2,2.5,3]
inDataAll=[]
for p in pVals:
    inDataAll.append(pd.read_csv("./arbitraryp"+str(p)+"/truep"+str(p)+".csv"))
def g2x(g,p):
    """
    solve xi(x0)=0
    :param g:
    if |g|>2B, solve equation;
    if 0<g <2B, let x0=-1;
    if -2B<g<0, let x0=1;
    :param p:
    :return: x0
    """
    if np.abs(g)>2*B:
        def f(x):
            return np.abs((1 + x) ** p - (1 - x) ** p + 2 ** (1 + p) * B / g) ** 2

        sol = root(fun=f, x0=0, method="hybr", tol=1e-15)
        return sol.x[0]
    elif g>= 0 and g<2*B:
        return -1
    else:
        return 1


def theoreticalPhases(g,p):
    """

    :param g: |g|>2B
    :param p:
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
gLeft=np.linspace(-10*B,-(2+dg)*B,1000)
gMiddle=np.linspace(-(2-dg)*B,(2-dg)*B,100)
gRight=np.linspace((2+dg)*B,10*B,1000)
gAll=np.append(gLeft,gMiddle)
gAll=np.append(gAll,gRight)
gLeftInd=list(range(0,len(gLeft)))
gMiddleInd=list(range(len(gLeft),len(gLeft)+len(gMiddle)))
gRightInd=list(range(len(gLeft)+len(gMiddle),len(gLeft)+len(gMiddle)+len(gRight)))

# pVals=[0.5,1,1.5,2,2.5,3]
tDTensor=np.zeros((len(gAll),len(pVals)))
tBTensor=np.zeros((len(gAll),len(pVals)))
ABTensor=np.zeros((len(gAll),len(pVals)))

def wrapper(nj):
    n,j=nj
    #n is index of g
    #j is index of p
    g=gAll[n]
    p=pVals[j]
    tD,tB,AB=theoreticalPhases(g,p)
    return [n,j,tD,tB,AB]


inIndices=[[n,j] for n in range(0,len(gAll)) for j in range(0,len(pVals))]

tPhasesStart=datetime.now()
procNum=24
pool0=Pool(procNum)

ret=pool0.map(wrapper,inIndices)

tPhasesEnd=datetime.now()

outDir="./arxivFigs/"
Path(outDir).mkdir(exist_ok=True,parents=True)

print("theoretical phases time: ",tPhasesEnd-tPhasesStart)

for item in ret:
    n,j,tD,tB,AB=item
    tDTensor[n,j]=tD
    tBTensor[n,j]=tB
    ABTensor[n,j]=AB


# pVals=[0.5,1,1.5,2,2.5,3]
colors=["blue","red","fuchsia","limegreen","darkorange","dimgrey"]
x0=-0.1
y0=1.1
fig,ax1=plt.subplots()
plt.rcParams['figure.constrained_layout.use'] = True
#dynamical phase
# ax1=fig.add_subplot(1,3,1)
for j in range(0,len(pVals)):
    ax1.plot(gAll / B, tDTensor[:, j], color=colors[j], label="p=" + str(pVals[j]))
    lenTmp = len(np.array(inDataAll[j].iloc[:, 0]))
    selected = list(range(0, lenTmp, 10))
    ax1.scatter(np.array(inDataAll[j].iloc[selected, 0]) / B, np.array(inDataAll[j].iloc[selected, 1]) / np.pi,
                color=colors[j], s=4)

ftSize=17

ax1.set_xlim((-10,10))
ax1.axhline(y=2, color='black', linestyle='--')
ax1.axhline(y=-2, color='black', linestyle='--')
ax1.axhline(y=1, color='gold', linestyle='--')
ax1.axhline(y=-1, color='gold', linestyle='--')
ax1.legend(loc="best")
ax1.set_xlabel("$g/B$", fontsize=ftSize)
ax1.set_ylabel("$\delta\\theta_{D}/\pi$",fontsize=ftSize)
ax1.set_yticks([-2,-1,0,1,2])
ax1.set_xticks(range(-10,14,4))
ax1.tick_params(axis='both', which='major', labelsize=ftSize-3)
ax1.text(x0, y0, "(a)", transform=ax1.transAxes,
            size=ftSize-2)
plt.savefig(outDir+"thetaD.pdf")
plt.close()
####

fig,ax2=plt.subplots()

for j in range(0,len(pVals)):
    # plt.plot(gAll/B,tBTensor[:,j],color=colors[j],label="p="+str(pVals[j]))
    ax2.plot(gAll[gLeftInd] / B, tBTensor[gLeftInd, j] , color=colors[j], label="p=" + str(pVals[j]))
    ax2.plot(gAll[gMiddleInd]/B,tBTensor[gMiddleInd,j]+2,color=colors[j])
    ax2.plot(gAll[gRightInd]/B,tBTensor[gRightInd,j]+2,color=colors[j])


def shiftBerry(phaseVec):
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
                color=colors[j], s=4)

ax2.set_xticks(np.arange(-10,11,2))
ax2.set_xlim((-10,10))
ax2.set_xticks(range(-10,14,4))
ax2.set_yticks([-0.8,-0.4,0,0.4,0.8])
ax2.tick_params(axis='both', which='major', labelsize=ftSize-3)

ax2.legend(loc="best")
ax2.set_xlabel("$g/B$", fontsize=ftSize)
ax2.set_ylabel("$\\theta_{B}/\pi$",fontsize=ftSize)
ax2.text(x0, y0, "(b)", transform=ax2.transAxes,
            size=ftSize-2)

plt.savefig(outDir+"thetaB.pdf")
plt.close()








###plt AB phases
fig,ax3=plt.subplots()

for j in range(0,len(pVals)):
    # plt.plot(gAll/B,ABTensor[:,j],color=colors[j],label="p="+str(pVals[j]))
    #left
    if j==1:
        continue
    if j==0:
        shiftLeft=0
    else:
        shiftLeft=2
    ax3.plot(gAll[gLeftInd] / B, ABTensor[gLeftInd, j] + shiftLeft, color=colors[j], label="p=" + str(pVals[j]))
    #middle
    shiftMiddle=2
    ax3.plot(gAll[gMiddleInd]/B,ABTensor[gMiddleInd,j]+shiftMiddle, color=colors[j])
    #right
    if j==0:
        shiftRight=2
    else:
        shiftRight=0
    ax3.plot(gAll[gRightInd]/B,ABTensor[gRightInd,j]+shiftRight,color=colors[j])

###p1=1
j1=1
ax3.plot(gAll/B,ABTensor[:,j1]+2,color=colors[j1],label="p="+str(pVals[j1]))
for j in range(0,len(pVals)):
    if j==1:
        continue
    lenTmp = len(np.array(inDataAll[j].iloc[:, 0]))
    selected = list(range(0, lenTmp, 10))
    ax3.scatter(np.array(inDataAll[j].iloc[selected, 0]) / B, np.array(inDataAll[j].iloc[selected, 2]) / np.pi,
                color=colors[j], s=4)

lenTmp = len(np.array(inDataAll[j1].iloc[:, 0]))
selected = list(range(0, lenTmp, 10))
def shiftPhaseForp1(phaseVec):
    def tmp(phase):
        if phase<-0.5:
            phase+=2
        return phase
    retVec=[tmp(phase) for phase in phaseVec]
    return retVec

ax3.scatter(np.array(inDataAll[j1].iloc[selected, 0]) / B, shiftPhaseForp1(np.array(inDataAll[j1].iloc[selected, 2]) / np.pi),
                color=colors[j1], s=4)
ax3.set_xticks(np.arange(-10,14,4))
ax3.set_xlim((-10,10))
ax3.tick_params(axis='both', which='major', labelsize=ftSize-3)

ax3.legend(loc="best")
ax3.set_xlabel("$g/B$", fontsize=ftSize)
ax3.set_ylabel("$\\theta_{AB}/\pi$",fontsize=ftSize)
ax3.set_yticks([-2,-1,0,1,2])
ax3.text(x0, y0, "(c)", transform=ax3.transAxes,
            size=ftSize-2)





plt.savefig(outDir+"thetaAB.pdf")
plt.close()