import numpy as np
import matplotlib.pyplot as plt

B=1

def thetaBp1(g):
    if np.abs(g)<=2*B:
        return -2
    else:
        return -1-2*B/g
def thetaBp2(g):
    if np.abs(g)<=2*B:
        return -2
    else:
        return -1-2*B/g


def thetaBp3(g):
    if np.abs(g)<=2*B:
        return -2
    else:
        lmd = (108 * B / g + 1 / 2 * (46656 * B ** 2 / g ** 2 + 2916) ** (1 / 2)) ** (1 / 3)
        x0 = -lmd / 3 + 3 / lmd
        return -1+x0


n=10
gVals=np.arange(-n,n+0.1,0.05)

thBp1=[thetaBp1(g) for g in gVals]
thBp2=[thetaBp2(g) for g in gVals]
thBp3=[thetaBp3(g) for g in gVals]
xVals=np.arange(-10,12,2)
ftSize=16
plt.figure()
plt.plot(gVals,thBp1,label="$p=1$",color="red",linewidth=1.5)
plt.plot(gVals,thBp2,label="$p=2$",color="green",linewidth=0.9)
plt.plot(gVals,thBp3,label="$p=3$",color="blue",linewidth=0.4)
plt.xticks(xVals)
plt.hlines(y=-1, xmin=-n, xmax=n, linewidth=0.05, color='k',linestyles="dotted")
plt.xlim((-n,n))
plt.ylim((-2,0))
plt.xlabel("$g/B$",fontsize=ftSize)
plt.ylabel("Berry phase$/\pi$",fontsize=ftSize)
plt.legend(loc="best")
plt.title("Berry phase for $p=1,2,3$",fontsize=ftSize)
plt.savefig("berryPhase.eps")
plt.close()