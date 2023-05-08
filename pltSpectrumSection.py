import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.optimize import root
#this script plots spectrum for full numerical solution and perturbative solution
#along section k1=0

B=2

def beta(k1Val,k2Val):
    """

    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :return:  value of beta
    """
    return B*(-1+np.cos(k1Val)+np.cos(k2Val))



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
        # eqn of x
        return np.abs(betaVal**2-(betaVal**2+gamma2Val)*x**2+2**(-p)*betaVal*gVal*x**2*(1-x)**p-2**(-p)*betaVal*gVal*x**2*(1+x)**p\
    -2**(-p)*betaVal*gVal*(1-x)**p+2**(-p)*betaVal*gVal*(1+x)**p+2**(-2*p-1)*gVal**2*x**2*(1-x**2)**p\
    -2**(-2*p-1)*gVal**2*(1-x**2)**p-2**(-2*p-2)*gVal**2*x**2*(1-x)**(2*p)\
    -2**(-2*p-2)*gVal**2*x**2*(1+x)**(2*p)+2**(-2*p-2)*gVal**2*(1-x)**(2*p)\
    +2**(-2*p-2)*gVal**2*(1+x)**(2*p))**2




    dx=1e-2
    # there may be multiple solutions to f(x)=0, therefore we have to start with multiple initial values x0
    scanX=np.linspace(-1+dx,1-dx,int(2/dx))# an array of starting values x0
    solutionSet=set()# containing solutions to f(x)=0 as a set
    for x0 in scanX:
        sol=root(fun=f,x0=x0,method="hybr",tol=1e-10)
        success=sol.success# if the above numerical procesure is successful

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
    :param k1Val: quasimomentum k1
    :param k2Val: quasimomentum k2
    :param gVal: g, strength of nonlinearity
    :return: [signal,E,x]. E is eigenenergy, x is the central quantity, signal is a boolean value indicating whether such an x can be computed numerically.
    """
    xs=solution(p,k1Val,k2Val,gVal)#a list containing the values of x
    if len(xs)==0:#if the list is empty
        signal=False#signal indicates whether a solution exists
        return [signal,1j,1j]#function E2x(p,k1Val,k2Val,gVal) is returned if list is empty
    # print(xs)
    # if the list is not empty, the execution of E2x(p,k1Val,k2Val,gVal) continues
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
    signal=True

    return [signal,E,x]# to indicate that we have successfully obtained the values of x and E


def xn(p,k2Val,g,s):
    """

    :param p: order of nonlinearity
    :param k2Val: quasimomentum k2
    :param g: strength of nonlinearity, |g|>2B
    :param s: sign variable in the perturbative expansions
    :return: [x0, perturbation to x]
    """

    k1Val=0
    signal,E,x0=E2x(p,0,0,g)
    if signal==False:
        return []# return empty list if computation is not successful
    F=g*p*((1+x0)**(p-1)+(1-x0)**(p-1))
    xnVal=2**(p+1)*np.abs(x0)*s/(F*np.sqrt(1-x0**2))*np.sqrt(B**2*k1Val**2+B**2*k2Val**2)
    return [x0,xnVal]#computation is successful


def EPerturbative(p,k2Val,g,s):
    """

    :param p: order of nonlinearity
    :param k2Val: quasimomentum k2
    :param g: strength of nonlinearity, |g|>2B
    :param s: sign variable in the perturbative expansions
    :return: [signal, eigenenergy up to 1st order]
    """
    ret=xn(p,k2Val,g,s)
    if len(ret)==0:
        return [False,1j]#computation is not successful
    x0,xnVal=ret
    E0=B/x0+g/2**(p+1)*((1+x0)**(p+1)-(1-x0)**(p+1))/x0#0th order value of eigenenergy

    EVal=E0+(g*(p+1)/2**(p+1)*((1+x0)**p+(1-x0)**p)/x0
             -g/2**(p+1)*((1+x0)**(p+1)-(1-x0)**(p+1))/x0**2
             -B/x0**2)*xnVal#0th order+1st order
    return [True,EVal]#computation is successful


########################################################################################################
#computation starts here

p=2.5

dk2Small=0.005#step length of k2/pi
end=0.02
smallK2All=np.arange(-end,end+dk2Small,dk2Small)*np.pi#all values of k2

##################################################################
#open this part if |g|>2B, otherwise this part is commented
#perturbative solutions
# g=2.5*B
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
##open this part if  |g|<=2B, otherwise this part is commented
g=2*B
if g>0 and g<=2*B:
    x0=-1
if g<0 and g>=-2*B:
    x0=1

k1=0
k2=0
betaVal=beta(k1,k2)
ERedPoint=x2E(p,betaVal,g,x0)# plot the point at the origin


#######################################################
inDir="./spectrump"+str(p)+"/"
inData=pd.read_csv(inDir+"g"+str(g/B)+"B.csv")#input csv file containing full numerical solutions from pltSpectrumSection.py
pltFullk2=inData.loc[:,"k"]
pltFullE=np.array(inData.loc[:,"E"])
#plot full numerical solutions using blue dots
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
# plot perturbative results of eigenenergy using red dots, use this part if |g|>2B, otherwise this part is commented

# ax.scatter(k2Plus,ESmallK2Plus/B,color="red",s=25)
# ax.scatter(k2Minus,ESmallK2Minus/B,color="red",s=25)
# print(k2Plus)
# print(ESmallK2Plus)
# print(k2Minus)
# print(ESmallK2Minus)
################################
# plot perturbative results of eigenenergy using red dots, use this part if |g|<=2B, otherwise this part is commented
ax.scatter(0,ERedPoint/B,color="red",s=25)
##############################################

plt.savefig(inDir+"g"+str(g/B)+"B.pdf")
