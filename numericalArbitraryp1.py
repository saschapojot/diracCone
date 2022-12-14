import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import expm
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
from scipy.optimize import root

r=1e-4
tTot=10*1000
B=2
Q=int(1e6)
dt=tTot/Q



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


def gamma2(k1,k2):
    return B**2*(np.sin(k1)**2+np.sin(k2)**2)


def solution(p,k1Val,k2Val,gVal):
    #solve x
    betaVal=beta(k1Val,k2Val)
    gamma2Val=gamma2(k1Val,k2Val)

    def f(x):
        #eqn of x
        return np.abs(betaVal**2-(betaVal**2+gamma2Val)*x**2+2**(-p)*betaVal*gVal*x**2*(1-x)**p-2**(-p)*betaVal*gVal*x**2*(1+x)**p\
    -2**(-p)*betaVal*gVal*(1-x)**p+2**(-p)*betaVal*gVal*(1+x)**p+2**(-2*p-1)*gVal**2*x**2*(1-x**2)**p\
    -2**(-2*p-1)*gVal**2*(1-x**2)**p-2**(-2*p-2)*gVal**2*x**2*(1-x)**(2*p)\
    -2**(-2*p-2)*gVal**2*x**2*(1+x)**(2*p)+2**(-2*p-2)*gVal**2*(1-x)**(2*p)\
    +2**(-2*p-2)*gVal**2*(1+x)**(2*p))**2


    # def jac(x):
    #     #derivative of eqn of x
    #     return -2*(betaVal**2+gamma2Val)*x-2**(-p)*betaVal*gVal*p*x**2*(1+x)**(p-1)-2**(-p)*betaVal*gVal*p*x**2*(1-x)**(p-1)\
    # +2**(-p)*betaVal*gVal*p*(1+x)**(p-1)+2**(-p)*betaVal*gVal*p*(1-x)**(p-1)+2**(1-p)*betaVal*gVal*x*(1-x)**p\
    # -2**(1-p)*betaVal*gVal*x*(1+x)**p-2**(-2*p+1)*gVal**2*p*x**3*(1-x**2)**(p-1)\
    # -2**(-2*p-1)*gVal**2*p*x**2*(1-x)**p*(1+x)**(p-1)+2**(-2*p-1)*gVal**2*p*x**2*(1-x)**(p-1)*(1+x)**p\
    # -2**(-2*p-1)*gVal**2*p*(1-x)**p*(1+x)**(p-1)+2**(-2*p-1)*gVal**2*p*(1-x)**(p-1)*(1+x)**p\
    # +2**(1-2*p)*gVal**2*x*(1-x**2)**p-2**(-2*p)*gVal**2*x*(1-x)**p*(1+x)**p\
    # -2**(-2*p-1)*gVal**2*p*x**2*(1+x)**(2*p-1)+2**(-2*p-1)*gVal**2*p*x**2*(1-x)**(2*p-1)\
    # +2**(-2*p-1)*gVal**2*p*(1+x)**(2*p-1)-2**(-2*p-1)*gVal**2*p*(1-x)**(2*p-1)\
    # -2**(-2*p-1)*gVal**2*x*(1-x)**(2*p)-2**(-2*p-1)*gVal**2*x*(1+x)**(2*p)

    dx=1e-2
    scanX=np.linspace(-1+dx,1-dx,int(2/dx))
    solutionSet=set()
    for x0 in scanX:
        sol=root(fun=f,x0=x0,method="hybr",tol=1e-10)
        success=sol.success
        funVal=sol.fun[0]
        # if success and np.abs(funVal)<1e-10:
        if success:
            solutionSet.add(round(sol.x[0],8))

    return list(solutionSet)

def x2E(p,betaVal,gVal,x):
    return betaVal/x+gVal*(1/2+1/2*x)**(p+1)/x-gVal*(1/2-1/2*x)**(p+1)/x


def E2x(p,k1Val,k2Val,gVal):
    """

    :param p:
    :param k1Val:
    :param k2Val:
    :param gVal:
    :return: choose the value of x based on the value of E
    """
    xs=solution(p,k1Val,k2Val,gVal)
    if len(xs)==0:
        signal=False
        return [signal,1j,1j]
    # print(xs)
    betaVal=beta(k1Val,k2Val)
    EVals=[x2E(p,betaVal,gVal,x) for x in xs]
    # print(np.array(EVals)/B)
    indsOfE = np.argsort(EVals)  # ascending order
    if gVal>0:
        #choose the lowest band
        indx=indsOfE[0]
    else:
        #choose the highest band
        indx=indsOfE[-1]

    x=xs[indx]
    E=EVals[indx]
    # print("choose x="+str(x)+", choose E="+str(E))
    # Ecomp=x2E(p,betaVal,gVal,x)
    # print("computation E="+str(Ecomp))
    signal=True

    return [signal,E,x]


def initVec(p,k1Val,k2Val,gVal):
    # computes the 2 components of initial vector, psi1 and psi2
    signal,E,x=E2x(p,k1Val,k2Val,gVal)
    if signal==False:
        return np.array([10,10])
    psi1=np.sqrt(1/2+1/2*x)
    betaVal=beta(k1Val,k2Val)
    gammaVal=gamma(k1Val,k2Val)
    # psi2 = (E - betaVal - gVal * np.abs(psi1) ** (2 * p)) / gammaVal * psi1
    angleTmp=np.angle((E - betaVal - gVal * np.abs(psi1) ** (2 * p)) / gammaVal * psi1)
    psi2=np.sqrt(1/2-1/2*x)*np.exp(1j*angleTmp)
    return np.array([psi1, psi2])


def oneStepSEN(q,gVal,psiIn,p):
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



def oneStepSWN(q, gVal, psiIn,p):
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

def thetaDqSEN(q,gVal,psiIn,p):
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


def thetaDqSWN(q,gVal,psiIn,p):
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


def thetaDSEN(gVal,psiAll,p):
    """

    :param gVal:
    :param psiAll: wavefuction at q=0,1,...,Q
    :return: total dynamical phase along SEN
    """
    thetaD=0
    for q in range(0,Q):
        psiq=psiAll[q]
        thetaD+=thetaDqSEN(q,gVal,psiq,p)
    return thetaD


def thetaDSWN(gVal,psiAll,p):
    """

    :param gVal:
    :param psiAll: wavefuction at q=0,1,...,Q
    :return: total dynamical phase along SWN
    """
    thetaD=0
    for q in range(0,Q):
        psiq=psiAll[q]
        thetaD+=thetaDqSWN(q,gVal,psiq,p)
    return thetaD


def evolutionSEN(gVal,p):
    """

    :param gVal:
    :return: wavefunctions along path SEN
    """
    phi0=phiSEN(0)
    k10=k1(phi0)
    k20=k2(phi0)
    psi0=initVec(p,k10,k20,gVal)
    psiAllSEN = [psi0]
    if np.linalg.norm(psi0,ord=2)>1+1e-3:
        return [False,psiAllSEN]
    for q in range(0, Q):
        psiAllSEN.append(oneStepSEN(q,gVal,psiAllSEN[q],p))

    return [True,psiAllSEN]




def evolutionSWN(gVal,p):
    """
    :param gVal:
    :return: wavefunctions along path SWN
    """
    phi0 = phiSWN(0)
    k10 = k1(phi0)
    k20 = k2(phi0)
    psi0 = initVec(p,k10,k20,gVal)
    psiAllSWN = [psi0]
    if np.linalg.norm(psi0,ord=2)>1+1e-3:
        return [False,psiAllSWN]
    for q in range(0, Q):
        psiAllSWN.append(oneStepSWN(q, gVal, psiAllSWN[q],p))
    return [True,psiAllSWN]



def circularPhase(gVal,p):
    """

    :param gVal:
    :param p:
    :return:
    """
    signalSEN,psiAllSEN = evolutionSEN(gVal,p)
    psiLastSEN = psiAllSEN[-1]
    signalSWN,psiAllSWN = evolutionSWN(gVal,p)
    psiLastSWN = psiAllSWN[-1]
    if signalSEN==False or signalSWN==False:
        return [False,1j,1j]

    prod = np.vdot(psiLastSWN, psiLastSEN)
    AB = np.angle(prod)
    thetaDSENVal = np.real(thetaDSEN(gVal,psiAllSEN,p))
    thetaDSWNVal = np.real(thetaDSWN(gVal, psiAllSWN,p))
    diffthetaD = thetaDSENVal - thetaDSWNVal

    # thetaBSENVal=thetaB(psiAllSEN)
    # thetaBSWNVal=thetaB(psiAllSWN)
    #
    # diffThetaB=thetaBSENVal-thetaBSWNVal

    return [True,diffthetaD, AB]


# #one value
# tStart=datetime.now()
# gVal=2.5*B
# p=1.1
# td,AB=circularPhase(gVal,p)
#
# tEnd=datetime.now()
# print("one round time: ",tEnd-tStart)
#
# print(AB/np.pi)
# print((td/np.pi))
# print((td/np.pi)-2*B/gVal)

p=1
def wrapper(gValp):
    gVal,p=gValp
    return [gVal,circularPhase(gVal,p)]


gValsAll=np.linspace(-10*B,10*B,500)
# gValsAll=[0.1*B]
gValsAllAndp=[[gVal,p] for gVal in gValsAll]

procNum=24
pool0=Pool(procNum)
tBatchStart=datetime.now()
ret=pool0.map(wrapper,gValsAllAndp)

tBatchEnd=datetime.now()

print("batch time: ",tBatchEnd-tBatchStart)
outDir="./arbitraryp"+str(p)+"/"
Path(outDir).mkdir(parents=True,exist_ok=True)
dTrueValsAll=[]
ABTrueValsAll=[]
gTrueValsAll=[]
gFalseValsAll=[]
for elem in ret:
    gTmp=elem[0]
    signal=elem[1][0]
    td=elem[1][1]
    AB=elem[1][2]
    if signal==True:
        gTrueValsAll.append(gTmp)
        ABTrueValsAll.append(AB)
        dTrueValsAll.append(td)
    else:
        gFalseValsAll.append(gTmp)

plt.figure()
plt.scatter(gTrueValsAll,dTrueValsAll,color="black")
plt.savefig(outDir+"p"+str(p)+"dynamicalPhase.png")
plt.close()
pdDataTrue=np.array([gTrueValsAll,dTrueValsAll,ABTrueValsAll]).T

outDataTrue=pd.DataFrame(data=pdDataTrue,columns=["g","td","AB"])

outDataTrue.to_csv(outDir+"truep"+str(p)+".csv",index=False)

pdDataFalse=np.array([gFalseValsAll]).T
outDataFalse=pd.DataFrame(data=pdDataFalse,columns=["g"])
outDataFalse.to_csv(outDir+"falsep"+str(p)+".csv",index=False)