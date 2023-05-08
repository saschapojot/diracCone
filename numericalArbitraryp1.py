import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import expm
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
from scipy.optimize import root

r=1e-4#radius of evolution path
tTot=1e4#total time along one path, must be >>1
B=2
Q=int(1e6)#time step number
dt=tTot/Q#time step length



def phiSEN(t):
    """

    :param t: time
    :return: phi value on counter clockwise path SEN
    """

    return -1/2*np.pi+np.pi/tTot*t

def phiSWN(t):
    """

    :param t: time
    :return: phi value on clockwise path SWN
    """
    return -1/2*np.pi-np.pi/tTot*t


def k1(phi):
    """

    :param phi: to parameterize quasimomenta
    :return: quasimomentum k1
    """
    return r*np.cos(phi)


def k2(phi):
    """

    :param phi: to parameterize quasimomenta
    :return: quasimomentum k2
    """
    return r*np.sin(phi)


def beta(k1Val,k2Val):
    """

    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :return: value of beta
    """
    return B*(-1+np.cos(k1Val)+np.cos(k2Val))


def gamma(k1Val,k2Val):
    """

    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :return: value of gamma
    """
    return B*(np.sin(k1Val)-1j*np.sin(k2Val))


def gamma2(k1,k2):
    """

    :param k1: quasimomentum k1
    :param k2: quasimomentum k2
    :return: value of |gamma|^2
    """
    return B**2*(np.sin(k1)**2+np.sin(k2)**2)


def solution(p,k1Val,k2Val,gVal):
    """

    :param p: order of nonlinearity
    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :param gVal: g, strength of nonlinearity
    :return: a list of values of x. If there is no solution, the returned list is empty
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
        sol=root(fun=f,x0=x0,method="hybr",tol=1e-10)
        success=sol.success# if the above numerical procesure is successful
        #funVal=sol.fun[0]
        # if success is True
        if success:
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


def E2x(p,k1Val,k2Val,gVal):
    """

    :param p: order of nonlinearity
    :param k1Val:quasimomentum k1
    :param k2Val: quasimomentum k2
    :param gVal: g, strength of nonlinearity
    :return: [signal,E,x]. E is eigenenergy, x is the central quantity, signal is a boolean value indicating whether such an x can be computed numerically.
    """
    xs=solution(p,k1Val,k2Val,gVal)#a list containing the values of x
    if len(xs)==0:#if the list is empty
        signal=False#signal indicates whether a solution exists
        return [signal,1j,1j]#function E2x(p,k1Val,k2Val,gVal) is returned if list is empty
    # print(xs)
    #if the list is not empty, the execution of E2x(p,k1Val,k2Val,gVal) continues
    betaVal=beta(k1Val,k2Val)
    EVals=[x2E(p,betaVal,gVal,x) for x in xs]#a list of eigenenergies computed from x
    # print(np.array(EVals)/B)
    indsOfE = np.argsort(EVals)  # sort the values of eigenenergies in ascending order, indsOfE is an array containing the indices of the sorted elements in the original list EVals
    if gVal>0:
        #choose the lowest band, from which the Dirac cone emerges when g>0
        indx=indsOfE[0]
    else:
        #choose the highest band, from which the Dirac cone emerges when g<0
        indx=indsOfE[-1]

    # the values of x and E
    x=xs[indx]
    E=EVals[indx]
    # print("choose x="+str(x)+", choose E="+str(E))
    # Ecomp=x2E(p,betaVal,gVal,x)
    # print("computation E="+str(Ecomp))
    signal=True# to indicate that we have successfully obtained the values of x and E

    return [signal,E,x]


def initVec(p,k1Val,k2Val,gVal):
    """

    :param p: order of nonlinearity
    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :param gVal: g, strength of nonlinearity
    :return: initial wavefunction
    """

    signal,E,x=E2x(p,k1Val,k2Val,gVal)
    if signal==False:
        return np.array([10,10])#if the computation of E and x is not successful, return a vector whose norm >1
    psi1=np.sqrt(1/2+1/2*x)
    betaVal=beta(k1Val,k2Val)
    gammaVal=gamma(k1Val,k2Val)
    # psi2 = (E - betaVal - gVal * np.abs(psi1) ** (2 * p)) / gammaVal * psi1
    angleTmp=np.angle((E - betaVal - gVal * np.abs(psi1) ** (2 * p)) / gammaVal * psi1)
    psi2=np.sqrt(1/2-1/2*x)*np.exp(1j*angleTmp)
    return np.array([psi1, psi2])#when the computation is successful, return the wavefunction, whose norm =1


def oneStepSEN(q,gVal,psiIn,p):
    """

    :param q: the q-th step in time evolution, along SEN
    :param gVal: g, strength of nonlinearity
    :param psiIn: the (known) wavefunction at step q, along SEN
    :return: wavefunction at step q+1, along SEN. The algorithm is operator splitting
    """

    tq=q*dt
    psi1q=psiIn[0]#first component of wavefunction
    psi2q=psiIn[1]#second component of wavefunction
    ###step 1: nonlinear evolution for 1/2 dt
    y1=psi1q*np.exp(-1j*gVal*np.abs(psi1q)**(2*p)*1/2*dt)
    y2=psi2q*np.exp(-1j*gVal*np.abs(psi2q)**(2*p)*1/2*dt)
    yVec=np.array([y1,y2])
    ###step 2: linear evolution for dt
    phiVal=phiSEN(tq+1/2*dt)#value of phi at time=tq+1/2*dt along path SEN
    k1Val=k1(phiVal)
    k2Val=k2(phiVal)
    betaVal=beta(k1Val,k2Val)
    gammaVal=gamma(k1Val,k2Val)
    h0=np.array([[betaVal,gammaVal],[np.conj(gammaVal),-betaVal]])#linear part of the Hamiltonian

    zVec=expm(-1j*dt*h0).dot(yVec)
    ###step 3: nonlinear evolution for 1/2 dt
    z1,z2=zVec
    psi1Next=z1*np.exp(-1j*gVal*np.abs(z1)**(2*p)*1/2*dt)
    psi2Next=z2*np.exp(-1j*gVal*np.abs(z2)**(2*p)*1/2*dt)

    return np.array([psi1Next,psi2Next])



def oneStepSWN(q, gVal, psiIn,p):
    """

    :param q: the q-th step in time evolution, along SWN
    :param gVal: g, strength of nonlinearity
    :param psiIn: the (known) wavefunction at step q, along SWN
    :return: wavefunction at step q+1, along SWN. The algorithm is operator splitting
    """

    tq = q * dt
    psi1q = psiIn[0]#first component of wavefunction
    psi2q = psiIn[1]#second component of wavefunction
    ###step 1: nonlinear evolution for 1/2 dt
    y1 = psi1q * np.exp(-1j * gVal * np.abs(psi1q) ** (2 * p) * 1 / 2 * dt)
    y2 = psi2q * np.exp(-1j * gVal * np.abs(psi2q) ** (2 * p) * 1 / 2 * dt)
    yVec = np.array([y1, y2])
    ###step 2: linear evolution for dt
    phiVal = phiSWN(tq + 1 / 2 * dt)#value of phi at time=tq+1/2*dt along path SWN
    k1Val = k1(phiVal)
    k2Val = k2(phiVal)
    betaVal = beta(k1Val, k2Val)
    gammaVal = gamma(k1Val, k2Val)
    h0 = np.array([[betaVal, gammaVal], [np.conj(gammaVal), -betaVal]])#linear part of the Hamiltonian

    zVec = expm(-1j * dt * h0).dot(yVec)
    ###step 3: nonlinear evolution for 1/2 dt
    z1, z2 = zVec
    psi1Next = z1 * np.exp(-1j * gVal * np.abs(z1) ** (2 * p) * 1 / 2 * dt)
    psi2Next = z2 * np.exp(-1j * gVal * np.abs(z2) ** (2 * p) * 1 / 2 * dt)

    return np.array([psi1Next, psi2Next])

def thetaDqSEN(q,gVal,psiIn,p):
    """

    :param q: the q-th step in time evolution, along SEN
    :param psiIn: the  wavefunction at step q, along SEN
    :param gVal: g, strength of nonlinearity
    :return: dynamical phase from step q to step q+1 along SEN
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

    :param q: the q-th step in time evolution, along SWN
    :param gVal: g, strength of nonlinearity
    :param psiIn: the  wavefunction at step q, along SWN
    :return: dynamical phase from step q to step q+1 along SEN along SWN
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

    :param gVal: g, strength of nonlinearity
    :param psiAll: a list of wavefunctions at q=1,2,...,Q
    :param p: order of nonlinearity
    :return: dynamical phase at the end of SEN
    """
    thetaD=0
    for q in range(0,Q):
        psiq=psiAll[q]
        thetaD+=thetaDqSEN(q,gVal,psiq,p)
    return thetaD


def thetaDSWN(gVal,psiAll,p):
    """

    :param gVal: g, strength of nonlinearity
    :param psiAll: a list of wavefunctions at q=1,2,...,Q
    :param p: order of nonlinearity
    :return: dynamical phase at the end of SWN
    """
    thetaD=0
    for q in range(0,Q):
        psiq=psiAll[q]
        thetaD+=thetaDqSWN(q,gVal,psiq,p)
    return thetaD


def evolutionSEN(gVal,p):
    """

    :param gVal: g, strength of nonlinearity
    :param p: order of nonlinearity
    :return: [signal, a list of wavefunctions along SEN at q=0,1,...,Q]. signal indicates whether the initial wavefunction is successfully computed
    """
    phi0=phiSEN(0)
    k10=k1(phi0)
    k20=k2(phi0)
    psi0=initVec(p,k10,k20,gVal)
    psiAllSEN = [psi0]
    if np.linalg.norm(psi0,ord=2)>1+1e-3:#when the computation of initial wavefunction fails, psi0=[10,10]
        return [False,psiAllSEN]
    for q in range(0, Q):
        psiAllSEN.append(oneStepSEN(q,gVal,psiAllSEN[q],p))

    return [True,psiAllSEN]#computation is successful




def evolutionSWN(gVal,p):
    """

    :param gVal: g, strength of nonlinearity
    :param p: order of nonlinearity
    :return: [signal, a list of wavefunctions along SWN at q=0,1,...,Q]. signal indicates whether the initial wavefunction is successfully computed
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
    return [True,psiAllSWN]#computation is successful



def circularPhase(gVal,p):
    """

    :param gVal: g, strength of nonlinearity
    :param p: order of nonlinearity
    :return: [signal, dynamical phase around the origin, AB phase around the origin]
    """
    signalSEN,psiAllSEN = evolutionSEN(gVal,p)#computation along SEN
    psiLastSEN = psiAllSEN[-1]
    signalSWN,psiAllSWN = evolutionSWN(gVal,p)#computation along SEN
    psiLastSWN = psiAllSWN[-1]
    if signalSEN==False or signalSWN==False:# if one of the above computations fails, the returned signal is False
        return [False,1j,1j]

    #both computations are successful, the execution continues here
    #AB phase
    prod = np.vdot(psiLastSWN, psiLastSEN)
    AB = np.angle(prod)
    #dynamical phase around the origin
    thetaDSENVal = np.real(thetaDSEN(gVal,psiAllSEN,p))
    thetaDSWNVal = np.real(thetaDSWN(gVal, psiAllSWN,p))
    diffthetaD = thetaDSENVal - thetaDSWNVal


    return [True,diffthetaD, AB]


########################################################################################################
#computation starts here

p=1

#a wrapper function for parallel computation
def wrapper(gValp):
    """

    :param gValp: list [g,p]. g is strength of nonlinearity. p is order of nonlinearity
    :return: [g, a list of the dynamical phase and AB phase around the origin]
    """
    gVal,p=gValp
    return [gVal,circularPhase(gVal,p)]#the dynamical phase and AB phase around the origin


gValsAll=np.linspace(-10*B,10*B,500)# all of the g values
# gValsAll=[0.1*B]
gValsAllAndp=[[gVal,p] for gVal in gValsAll]

procNum=48#parallel processes number
pool0=Pool(procNum)
tBatchStart=datetime.now()#start time
ret=pool0.map(wrapper,gValsAllAndp)#parallel computations

tBatchEnd=datetime.now()#end time

print("batch time: ",tBatchEnd-tBatchStart)
outDir="./arbitraryp"+str(p)+"/"#output directory
Path(outDir).mkdir(parents=True,exist_ok=True)


#data retrieval
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
####plot dynamical phase
plt.figure()
plt.scatter(gTrueValsAll,dTrueValsAll,color="black")
plt.savefig(outDir+"p"+str(p)+"dynamicalPhase.png")
plt.close()

#dynamical phase and AB phase to csv
pdDataTrue=np.array([gTrueValsAll,dTrueValsAll,ABTrueValsAll]).T

outDataTrue=pd.DataFrame(data=pdDataTrue,columns=["g","td","AB"])

outDataTrue.to_csv(outDir+"truep"+str(p)+".csv",index=False)#data to csv (successful computations)

pdDataFalse=np.array([gFalseValsAll]).T
outDataFalse=pd.DataFrame(data=pdDataFalse,columns=["g"])
outDataFalse.to_csv(outDir+"falsep"+str(p)+".csv",index=False)#data to csv (failed computations)