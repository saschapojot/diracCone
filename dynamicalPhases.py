import numpy as np
import matplotlib.pyplot as plt

B=1
#this script computes dynamical phases in unit of pi
def thetaDp1(g):
    if np.abs(g)<=2*B:
        return 0
    else:
        return 2*B/g

def thetaDp2(g):
    if np.abs(g)<=2*B:
        return 0
    else:
        return 4*B/g
def thetaDp3(g):
    if np.abs(g)<=2*B:
        return 0
    else:
        lmd=(108*B/g+1/2*(46656*B**2/g**2+2916)**(1/2))**(1/3)
        x0=-lmd/3+3/lmd
        thd=4*g*(B/g*x0+2*B**2/g**2)/(B*x0**2-g*x0-3*B)
        return thd

n=10
gVals=np.arange(-n,n+0.1,0.05)

thDp1=[thetaDp1(g) for g in gVals]
thDp2=[thetaDp2(g) for g in gVals]
thDp3=[thetaDp3(g) for g in gVals]
xVals=np.arange(-10,12,2)
ftSize=16
plt.figure()
plt.plot(gVals,thDp1,label="$p=1$",color="red",linewidth=1.5)
plt.plot(gVals,thDp2,label="$p=2$",color="green",linewidth=1)
plt.plot(gVals,thDp3,label="$p=3$",color="blue",linewidth=0.4)
plt.xticks(xVals)
plt.ylim((-2,2))
plt.xlim((-n,n))
plt.xlabel("$g/B$",fontsize=ftSize)
plt.ylabel("Dynamical phase$/\pi$",fontsize=ftSize)
plt.legend(loc="best")
plt.title("Dynamical phase for $p=1,2,3$",fontsize=ftSize)
plt.hlines(y=0, xmin=-n, xmax=n, linewidth=0.05, color='k',linestyles="dotted")

# plt.scatter(gVals,thDp1,label="$p=1$",color="red")
# plt.scatter(gVals,thDp2,label="$p=2$",color="green")
# plt.scatter(gVals,thDp3,label="$p=3$",color="blue")
plt.savefig("dynamicalPhase.eps")
plt.close()