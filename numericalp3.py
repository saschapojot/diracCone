import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import expm
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
#this script computes dynamical, Berry and AB phases for p=1

r=1e-4
tTot=10*1000
B=2
Q=int(1e6)
dt=tTot/Q
p=3

def phiSEN(t):
    #counter clockwise path
    return -1/2*np.pi+np.pi/tTot*t

def phiSWN(t):
    #clockwise path
    return -1/2*np.pi-np.pi/tTot*t


def k1(phi):
    return r*np.cos(phi)


def k2(phi):
    return r*np.sin(phi)


def beta(k1Val,k2Val):
    return B*(-1+np.cos(k1Val)+np.cos(k2Val))


def gamma(k1Val,k2Val):
    return B*(np.sin(k1Val)-1j*np.sin(k2Val))


def solution(gVal,betaVal,gammaVal):
    #solve real values of x, sorted from small to large (with sign)
    inList=[gVal**2,0,5*gVal**2,16*betaVal*gVal,3*gVal**2,32*betaVal*gVal,64*betaVal**2+64*np.abs(gammaVal)**2-9*gVal**2\
            ,-48*betaVal*gVal,-64*betaVal**2]
    roots=np.roots(inList)

    realRoots=[]
    for oneRoot in roots:
        if np.abs(np.imag(oneRoot))<=1e-8 and np.abs(np.real(oneRoot))<=1:
            realRoots.append(np.real(oneRoot))
    rst=sorted(realRoots)
    return rst

def x2E(x,gVal,betaVal):
    """

    :param x:
    :param gVal:
    :param betaVal:
    :return: transform x to E for p=3
    """
    E=betaVal/x+1/2*gVal+1/2*gVal*x**2

    return E

def E2x(gVal,betaVal,gammaVal):
    #choose the value of x based on the value of E
    xs=solution(gVal,betaVal,gammaVal)

    EVals=[x2E(x,gVal,betaVal) for x in xs]
    indsOfE=np.argsort(EVals)#ascending order

    if gVal>0:
        #choose the lowest band
        indx=indsOfE[0]
    else:
        #choose the highest band
        indx=indsOfE[-1]

    x=xs[indx]
    E=EVals[indx]

    return [E,x]


def initVec(gVal,betaVal,gammaVal):
    #computes the 2 components of initial vector, psi1 and psi2
    E,x=E2x(gVal,betaVal,gammaVal)
    psi1=np.sqrt(1/2+1/2*x)
    psi2=(E-betaVal-gVal*np.abs(psi1)**(2*p))/gammaVal*psi1
    return np.array([psi1,psi2])




def oneStepSEN(q,gVal,psiIn):
    """

    :param q:
    :param gVal:
    :param psiIn:
    :return: next step along SEN
    """

    tq=q*dt
    psi1q=psiIn[0]
    psi2q=psiIn[1]
    ###step 1: nonlinear evolution for 1/2 dt
    y1=psi1q*np.exp(-1j*gVal*np.abs(psi1q)**(2*p)*1/2*dt)
    y2=psi2q*np.exp(-1j*gVal*np.abs(psi2q)**(2*p)*1/2*dt)
    yVec=np.array([y1,y2])
    ###step 2: linear evolution for dt
    phiVal=phiSEN(tq+1/2*dt)
    k1Val=k1(phiVal)
    k2Val=k2(phiVal)
    betaVal=beta(k1Val,k2Val)
    gammaVal=gamma(k1Val,k2Val)
    h0=np.array([[betaVal,gammaVal],[np.conj(gammaVal),-betaVal]])

    zVec=expm(-1j*dt*h0).dot(yVec)
    ###step 3: nonlinear evolution for 1/2 dt
    z1,z2=zVec
    psi1Next=z1*np.exp(-1j*gVal*np.abs(z1)**(2*p)*1/2*dt)
    psi2Next=z2*np.exp(-1j*gVal*np.abs(z2)**(2*p)*1/2*dt)

    return np.array([psi1Next,psi2Next])


def oneStepSWN(q, gVal, psiIn):
    """

    :param q:
    :param gVal:
    :param psiIn:
    :return: next step along SWN
    """

    tq = q * dt
    psi1q = psiIn[0]
    psi2q = psiIn[1]
    ###step 1: nonlinear evolution for 1/2 dt
    y1 = psi1q * np.exp(-1j * gVal * np.abs(psi1q) ** (2 * p) * 1 / 2 * dt)
    y2 = psi2q * np.exp(-1j * gVal * np.abs(psi2q) ** (2 * p) * 1 / 2 * dt)
    yVec = np.array([y1, y2])
    ###step 2: linear evolution for dt
    phiVal = phiSWN(tq + 1 / 2 * dt)
    k1Val = k1(phiVal)
    k2Val = k2(phiVal)
    betaVal = beta(k1Val, k2Val)
    gammaVal = gamma(k1Val, k2Val)
    h0 = np.array([[betaVal, gammaVal], [np.conj(gammaVal), -betaVal]])

    zVec = expm(-1j * dt * h0).dot(yVec)
    ###step 3: nonlinear evolution for 1/2 dt
    z1, z2 = zVec
    psi1Next = z1 * np.exp(-1j * gVal * np.abs(z1) ** (2 * p) * 1 / 2 * dt)
    psi2Next = z2 * np.exp(-1j * gVal * np.abs(z2) ** (2 * p) * 1 / 2 * dt)

    return np.array([psi1Next, psi2Next])



def thetaDqSEN(q,gVal,psiIn):
    """

    :param q:
    :param psiIn:
    :param gVal:
    :return: one dynamical phase at time step q along SEN
    """
    tq=dt*q
    phiVal = phiSEN(tq)
    k1Val = k1(phiVal)
    k2Val = k2(phiVal)
    betaVal = beta(k1Val, k2Val)
    gammaVal = gamma(k1Val, k2Val)
    h0 = np.array([[betaVal, gammaVal], [np.conj(gammaVal), -betaVal]])
    psi1q=psiIn[0]
    psi2q=psiIn[1]

    thetaDq=-(np.conj(psiIn).dot(h0).dot(psiIn)*dt\
            +gVal*(np.abs(psi1q)**(2*p+2)+np.abs(psi2q)**(2*p+2))*dt)

    return thetaDq


def thetaDqSWN(q,gVal,psiIn):
    """

    :param q:
    :param gVal:
    :param psiIn:
    :return: one dynamical phase at time step q along SWN
    """
    tq = dt * q
    phiVal = phiSWN(tq)
    k1Val = k1(phiVal)
    k2Val = k2(phiVal)
    betaVal = beta(k1Val, k2Val)
    gammaVal = gamma(k1Val, k2Val)
    h0 = np.array([[betaVal, gammaVal], [np.conj(gammaVal), -betaVal]])
    psi1q = psiIn[0]
    psi2q = psiIn[1]

    thetaDq = -(np.conj(psiIn).dot(h0).dot(psiIn) * dt \
              + gVal * (np.abs(psi1q) ** (2 * p + 2) + np.abs(psi2q) ** (2 * p + 2)) * dt)

    return thetaDq


def thetaDSEN(gVal,psiAll):
    """

    :param gVal:
    :param psiAll: wavefuction at q=0,1,...,Q
    :return: total dynamical phase along SEN
    """
    thetaD=0
    for q in range(0,Q):
        psiq=psiAll[q]
        thetaD+=thetaDqSEN(q,gVal,psiq)
    return thetaD

def thetaDSWN(gVal,psiAll):
    """

    :param gVal:
    :param psiAll: wavefuction at q=0,1,...,Q
    :return: total dynamical phase along SWN
    """
    thetaD=0
    for q in range(0,Q):
        psiq=psiAll[q]
        thetaD+=thetaDqSWN(q,gVal,psiq)
    return thetaD



def dtPsiq(psiAll):
    """

    :param psiAll: wavefunction at q=0,1,...,Q
    :return: time derivative of wavefunction at q=0,1,...,Q-1
    """
    dtPsi=[]
    #time derivative of psi at q=0
    psi1=psiAll[1]
    psi2=psiAll[2]
    psi0=psiAll[0]
    dtPsi.append((4*psi1-psi2-3*psi0)/(2*dt))

    #time derivative of psi at q=1,...,Q-1
    for q in range(1,Q):
        dtPsi.append((psiAll[q+1]-psiAll[q-1])/(2*dt))

    return dtPsi

def thetaB(psiAll):
    """

    :param psiAll: wavefunction at q=0,1,...,Q
    :return: total Berry phase
    """
    timeDifferential=dtPsiq(psiAll)
    thetaBTotal=0
    for q in range(0,Q):
        thetaBTotal+=np.vdot(psiAll[q],timeDifferential[q])
    thetaBTotal*=dt*1j
    return thetaBTotal



def evolutionSEN(gVal):
    """

    :param gVal:
    :return: wavefunctions along path SEN
    """
    phi0=phiSEN(0)
    k10=k1(phi0)
    k20=k2(phi0)
    gamma0=gamma(k10,k20)
    beta0=beta(k10,k20)
    psi0=initVec(gVal,beta0,gamma0)
    # print([np.abs(psi0[0]),np.abs(psi0[1])])
    psiAllSEN=[psi0]
    for q in range(0,Q):
        psiAllSEN.append(oneStepSEN(q,gVal,psiAllSEN[q]))
    return psiAllSEN



def evolutionSWN(gVal):
    """
    :param gVal:
    :return: wavefunctions along path SWN
    """
    phi0 = phiSWN(0)
    k10 = k1(phi0)
    k20 = k2(phi0)
    gamma0 = gamma(k10, k20)
    beta0 = beta(k10, k20)
    psi0 = initVec(gVal, beta0, gamma0)
    psiAllSWN = [psi0]
    for q in range(0, Q):
        psiAllSWN.append(oneStepSWN(q, gVal, psiAllSWN[q]))
    return psiAllSWN


def circularPhase(gVal):
    psiAllSEN=evolutionSEN(gVal)
    psiLastSEN=psiAllSEN[-1]
    # print([np.abs(psiLastSEN[0]),np.abs(psiLastSEN[1])])
    psiAllSWN=evolutionSWN(gVal)
    psiLastSWN=psiAllSWN[-1]
    # print([np.abs(psiLastSWN[0]),np.abs(psiLastSWN[1])])

    prod=np.vdot(psiLastSWN,psiLastSEN)
    AB=np.angle(prod)
    # print("AB/pi="+str(AB/np.pi))
    thetaDSENVal=np.real(thetaDSEN(gVal,psiAllSEN))
    thetaDSWNVal=np.real(thetaDSWN(gVal,psiAllSWN))
    # print("thetaD SEN="+str(thetaDSENVal))
    # print("thetaD SWN="+str(thetaDSWNVal))
    diffthetaD=thetaDSENVal-thetaDSWNVal

    # thetaBSENVal=thetaB(psiAllSEN)
    # thetaBSWNVal=thetaB(psiAllSWN)
    #
    # diffThetaB=thetaBSENVal-thetaBSWNVal

    return [diffthetaD,AB]

# #one value
# tStart=datetime.now()
# gVal=1.2*B
# td,AB=circularPhase(gVal)
#
# tEnd=datetime.now()
# print("one round time: ",tEnd-tStart)

# solve x0
# coefs=[gVal,0,3*gVal,8*B]
# rts=np.roots(coefs)
# x0=0
# for elem in rts:
#     if np.abs(np.imag(elem))<1e-8 and np.abs(np.real(elem))<=1:
#         x0=np.real(elem)
# numericalTd=4*gVal*(B/gVal*x0+2*B**2/gVal**2)/(B*x0**2-gVal*x0-3*B)



# print((td/np.pi))
# print((td/np.pi)-numericalTd)

# multiprocessing

def wrapper(gVal):
    return [gVal,circularPhase(gVal)]


gValsAll=np.linspace(-10,10,500)
procNum=24
pool0=Pool(procNum)

tBatchStart=datetime.now()
ret=pool0.map(wrapper,gValsAll)

tBatchEnd=datetime.now()
print("batch time: ",tBatchEnd-tBatchStart)

outDir="./p"+str(p)+"/"
Path(outDir).mkdir(parents=True,exist_ok=True)

dValsAll=[]
ABValsAll=[]
for elem in ret:
    td=elem[1][0]
    AB=elem[1][1]
    dValsAll.append(td)
    ABValsAll.append(AB)

plt.figure()
plt.scatter(gValsAll,dValsAll,color="black")
plt.savefig(outDir+"p"+str(p)+"dynamicalPhase.png")



pdData=np.array([gValsAll,dValsAll,ABValsAll]).T

outData=pd.DataFrame(data=pdData,columns=["g","td","AB"])

outData.to_csv(outDir+"p"+str(p)+".csv",index=False)