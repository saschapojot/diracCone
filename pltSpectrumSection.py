import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.optimize import root
#this script plots spectrum for full numerical solution and perturbative solution
#along section k1=0

B=2

def beta(k1Val,k2Val):
    return B*(-1+np.cos(k1Val)+np.cos(k2Val))



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
        sol=root(fun=f,x0=x0,method="hybr",tol=1e-15)
        success=sol.success
        funVal=sol.fun[0]
        # if success and np.abs(funVal)<1e-10:
        if success and np.abs(funVal)<1e-10:
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


def xn(p,k2Val,g,s):
    """

    :param p:
    :param k2Val:
    :param g: |g|>2B
    :param s:
    :return:
    """

    k1Val=0
    signal,E,x0=E2x(p,0,0,g)
    if signal==False:
        return []
    F=g*p*((1+x0)**(p-1)+(1-x0)**(p-1))
    xnVal=2**(p+1)*np.abs(x0)*s/(F*np.sqrt(1-x0**2))*np.sqrt(B**2*k1Val**2+B**2*k2Val**2)
    return [x0,xnVal]


def EPerturbative(p,k2Val,g,s):
    """

    :param p:
    :param k2Val:
    :param g: |g|>2B
    :param s:
    :return:
    """
    ret=xn(p,k2Val,g,s)
    if len(ret)==0:
        return [False,1j]
    x0,xnVal=ret
    E0=B/x0+g/2**(p+1)*((1+x0)**(p+1)-(1-x0)**(p+1))/x0

    EVal=E0+(g*(p+1)/2**(p+1)*((1+x0)**p+(1-x0)**p)/x0
             -g/2**(p+1)*((1+x0)**(p+1)-(1-x0)**(p+1))/x0**2
             -B/x0**2)*xnVal
    return [True,EVal]

p=2

dk2Small=0.005
end=0.02
smallK2All=np.arange(-end,end+dk2Small,dk2Small)*np.pi

##################################################################
#|g|>2B
#perturbative solutions
# g=-2.5*B
# ESmallK2Plus=[]
# k2Plus=[]
# ESmallK2Minus=[]
# k2Minus=[]
# tPerturbativeStart=datetime.now()
# for k2 in smallK2All:
#     signalPlus,EPlus=EPerturbative(p,k2,g,1)
#     if signalPlus:
#         ESmallK2Plus.append(EPlus)
#         k2Plus.append(k2/np.pi)
#     signalMinus,EMinus=EPerturbative(p,k2,g,-1)
#     if signalMinus:
#         ESmallK2Minus.append(EMinus)
#         k2Minus.append(k2/np.pi)
#
#
# tPerturbativeEnd=datetime.now()
# print("perturbative time: ",tPerturbativeEnd-tPerturbativeStart)
#
# ESmallK2Plus=np.array(ESmallK2Plus)
# ESmallK2Minus=np.array(ESmallK2Minus)
#########################################################
########################################################
##|g|<=2B
g=2*B
if g>0 and g<=2*B:
    x0=-1
if g<0 and g>=-2*B:
    x0=1

k1=0
k2=0
betaVal=beta(k1,k2)
ERedPoint=x2E(p,betaVal,g,x0)


#######################################################
inDir="./spectrump"+str(p)+"/"
inData=pd.read_csv(inDir+"g"+str(g/B)+"B.csv")
pltFullk2=inData.loc[:,"k"]
pltFullE=np.array(inData.loc[:,"E"])
ftSize=16
tickSize=14
fig=plt.figure()
ax=fig.add_subplot()
ax.scatter(pltFullk2,pltFullE/B,s=7,c="blue",label="numerical solution")
ax.set_xlabel("$k_{2}/\pi$",fontsize=ftSize,labelpad=2)
ax.set_ylabel("$E/B$",fontsize=ftSize)
ax.yaxis.set_label_position("right")
ax.set_title("$p=$"+str(p)+", $g/B=$"+str(g/B),fontsize=ftSize)
ax.tick_params(axis='both', which='major', labelsize=tickSize)
x1=-0.1
y1=1.01
ax.text(x1,y1,"(d)",transform=ax.transAxes,
            size=ftSize-4)#numbering
##########################################
##############################################################
# #perturbative solution for |g|>2B
# ax.scatter(k2Plus,ESmallK2Plus/B,color="red",s=10,label="pertubative solution")
# ax.scatter(k2Minus,ESmallK2Minus/B,color="red",s=10)
################################
#perturbative solution for |g|<=2B
ax.scatter(0,ERedPoint/B,color="red",s=10,label="pertubative solution")
##############################################
lgnd =plt.legend(loc="best",fontsize=ftSize-2)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.savefig(inDir+"g"+str(g/B)+"B.pdf")
