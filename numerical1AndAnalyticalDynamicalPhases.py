import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

B=2
#this script computes dynamical phases in unit of pi analytically
#and reads numerical solutions of dynamical phases computed using
#np.root() function
#the following 3 functions are dynamical phase/pi from analytical results
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

n=10*B
gVals=np.arange(-n,n+0.1,0.05)

thDp1=[thetaDp1(g) for g in gVals]
thDp2=[thetaDp2(g) for g in gVals]
thDp3=[thetaDp3(g) for g in gVals]

########reading from csv

p1=1
inFile1="./p"+str(p1)+"/"+"p"+str(p1)+".csv"
inData1=pd.read_csv(inFile1)
g1ValsAll=np.array(inData1.iloc[:,0])
thetaDp1ValsAll=np.array(inData1.iloc[:,1])

p2=2
inFile2="./p"+str(p2)+"/"+"p"+str(p2)+".csv"
inData2=pd.read_csv(inFile2)
g2ValsAll=np.array(inData2.iloc[:,0])
thetaDp2ValsAll=np.array(inData2.iloc[:,1])

p3=3
inFile3="./p"+str(p3)+"/"+"p"+str(p3)+".csv"
inData3=pd.read_csv(inFile3)
g3ValsAll=np.array(inData3.iloc[:,0])
thetaDp3ValsAll=np.array(inData3.iloc[:,1])




ftSize=17
fig1,ax1=plt.subplots()
ax1.set_title("Dynamical phase for $p=1,2,3$",fontsize=ftSize)
ax1.plot(gVals/B,thDp1,label="$p=1$",color="red",linewidth=1.5)
ax1.scatter(g1ValsAll/B,thetaDp1ValsAll/np.pi,color="red",s=5)
ax1.plot(gVals/B,thDp2,label="$p=2$",color="green",linewidth=1.5)
ax1.scatter(g2ValsAll/B,thetaDp2ValsAll/np.pi,color="green",s=5)
ax1.plot(gVals/B,thDp3,label="$p=3$",color="blue",linewidth=1.5)
ax1.scatter(g3ValsAll/B,thetaDp3ValsAll/np.pi,color="blue",s=5)
xTicks=np.arange(-10,12,2)
ax1.set_xticks(xTicks)
ax1.set_ylim((-2,2))
ax1.set_xlim((-10,10))
ax1.set_xlabel("$g/B$",fontsize=ftSize)
ax1.set_ylabel("Dynamical phase$/\pi$",fontsize=ftSize)
ax1.legend(loc="best")
ax1.hlines(y=0, xmin=-10, xmax=10, linewidth=0.5, color='k',linestyles="dotted")
plt.savefig("dynamicalPhaseNum1.png")
plt.close()