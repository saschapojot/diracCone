import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root
#this script plots dynamical and AB phases for arbitraty p
#also plots the theoretical values





B=2
p=1.1

def g2x(g):
    """
    solve function xi(x0)=0
    :param g:>2B or <-2B
    :return: x0
    """
    def f(x):
        return np.abs((1+x)**p-(1-x)**p+2**(1+p)*B/g)**2

    sol=root(fun=f,x0=-0.5,method="hybr",tol=1e-15)
    return sol.x[0]

def thetaDTheoretical(g):
    """

    :param g: |g|>2B
    :return: theoretical value of dynamical phase for g>2B or g<-2B, in terms of pi
    """
    x0=g2x(g)
    E0=B/x0+g/2**(p+1)*((1+x0)**(p+1)-(1-x0)**(p+1))/x0
    D1=E0+g*p/2**p*(1+x0)**(p+1)-B-g/2**p*(2*p+1)*(1+x0)**p-g*p/2**p*(1+x0)*(1-x0)**p
    tD=-2*B*p*(1-x0**2)/D1
    return tD


inDir="./arbitraryp"+str(p)+"/"
inData=pd.read_csv(inDir+"truep"+str(p)+".csv")

gValsAll=np.array(inData.iloc[:,0])
thetaDValsAll=np.array(inData.iloc[:,1])
ABValsAll=np.array(inData.iloc[:,2])

BerryValsAll=ABValsAll-thetaDValsAll


ftSize=17
xTicks=np.arange(-10,12,2)

gRightVals=np.linspace(2.01*B,10*B,100)
tDVals=[thetaDTheoretical(g) for g in gRightVals]
fig1, ax1 = plt.subplots()
ax1.set_title("$p=$"+str(p))
ax1.scatter(gValsAll/B,thetaDValsAll/np.pi,color="black",s=2)
ax1.plot(gRightVals/B,tDVals,color="red")
ax1.set_xlabel("$g/B$",fontsize=ftSize)
ax1.set_xticks(xTicks)

ax1.yaxis.set_label_position("right")
ax1.set_ylabel("dynamical phase$/\pi$",fontsize=ftSize)
ax1.set_ylim((min(thetaDValsAll/np.pi)*1.1,max(thetaDValsAll/np.pi)*1.1))
ax1.set_yticks(np.linspace(min(thetaDValsAll/np.pi),max(thetaDValsAll/np.pi),5))
plt.savefig(inDir+"p"+str(p)+"dynamicalPhase.png")
plt.close()


fig2,ax2=plt.subplots()
ax2.set_title("$p=$"+str(p))
ax2.scatter(gValsAll/B,ABValsAll/np.pi,color="black",s=2)
ax2.set_xlabel("$g/B$",fontsize=ftSize)
ax2.set_xticks(xTicks)

ax2.yaxis.set_label_position("right")
ax2.set_ylabel("AB phase$/\pi$",fontsize=ftSize)
ax2.set_ylim((min(ABValsAll/np.pi)*1.1,max(ABValsAll/np.pi)*1.1))
ax2.set_yticks(np.linspace(min(ABValsAll/np.pi),max(ABValsAll/np.pi),5))
plt.savefig(inDir+"p"+str(p)+"ABPhase.png")
plt.close()

fig3,ax3=plt.subplots()
ax3.set_title("$p=$"+str(p))
ax3.scatter(gValsAll/B,BerryValsAll/np.pi,color="black",s=2)
ax3.set_xlabel("$g/B$",fontsize=ftSize)
ax3.set_xticks(xTicks)

ax3.yaxis.set_label_position("right")
ax3.set_ylabel("Berry phase$/\pi$",fontsize=ftSize)
ax3.set_ylim((min(BerryValsAll/np.pi)*1.1,max(BerryValsAll/np.pi)*1.1))
ax3.set_yticks(np.linspace(min(BerryValsAll/np.pi),max(BerryValsAll/np.pi),5))
plt.savefig(inDir+"p"+str(p)+"BerryPhase.png")
plt.close()