import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


B=2
#this script computes AB phases in unit of pi analytically
#and reads numerical solutions of dynamical phases computed using
#scipy.optimize.root() function
#the following 3 functions are AB phase/pi from analytical results
#in the analytic results, a jump of 2pi is set to 0
def thetaABp1(g):
    if np.abs(g)<=2*B:
        return -2+2
    else:
        return -1+2

def thetaABp2(g):
    if g<=-2*B:
        return -1+2*B/g+2
    elif g>2*B:
        return -1+2*B/g
    else:
        return 0

def thetaABp3(g):
    if np.abs(g)<=2*B:
        return -2+2
    elif g<-2*B:
        lmd = (108 * B / g + 1 / 2 * (46656 * B ** 2 / g ** 2 + 2916) ** (1 / 2)) ** (1 / 3)
        x0 = -lmd / 3 + 3 / lmd
        return -1-(g*x0**2+2*B*x0)/(B*x0**2-g*x0-3*B)+2
    else:
        lmd = (108 * B / g + 1 / 2 * (46656 * B ** 2 / g ** 2 + 2916) ** (1 / 2)) ** (1 / 3)
        x0 = -lmd / 3 + 3 / lmd
        return -1 - (g * x0 ** 2 + 2 * B * x0) / (B * x0 ** 2 - g * x0 - 3 * B)

########reading from csv

p1=1
inFile1="./arbitraryp"+str(p1)+"/true"+"p"+str(p1)+".csv"
inData1=pd.read_csv(inFile1)
sep1=5
g1ValsAll=np.array(inData1.iloc[:,0])
g1ValsAll=[g1ValsAll[j] for j in range(0,len(g1ValsAll),sep1)]
g1ValsAll=np.array(g1ValsAll)
thetaABp1ValsAll=np.array(inData1.iloc[:,2])
thetaABp1ValsAll=[thetaABp1ValsAll[j] for j in range(0,len(thetaABp1ValsAll),sep1)]
thetaABp1ValsAll=np.array(thetaABp1ValsAll)

p2=2
sep2=5
inFile2="./arbitraryp"+str(p2)+"/true"+"p"+str(p2)+".csv"
inData2=pd.read_csv(inFile2)
g2ValsAll=np.array(inData2.iloc[:,0])
g2ValsAll=[g2ValsAll[j] for j in range(0,len(g2ValsAll),sep2)]
g2ValsAll=np.array(g2ValsAll)
thetaABp2ValsAll=np.array(inData2.iloc[:,2])
thetaABp2ValsAll=[thetaABp2ValsAll[j] for j in range(0,len(thetaABp2ValsAll),sep2)]
thetaABp2ValsAll=np.array(thetaABp2ValsAll)

p3=3
sep3=5
inFile3="./p"+str(p3)+"/"+"p"+str(p3)+".csv"
inData3=pd.read_csv(inFile3)
g3ValsAll=np.array(inData3.iloc[:,0])
g3ValsAll=[g3ValsAll[j] for j in range(0,len(g3ValsAll),sep3)]
g3ValsAll=np.array(g3ValsAll)
thetaABp3ValsAll=np.array(inData3.iloc[:,2])
thetaABp3ValsAll=[thetaABp3ValsAll[j] for j in range(0,len(thetaABp3ValsAll),sep3)]
thetaABp3ValsAll=np.array(thetaABp3ValsAll)


n=10
gVals=np.arange(-10*B,10*B+0.1,0.005)

thABp1=[thetaABp1(g) for g in gVals]
thABp2=[thetaABp2(g) for g in gVals]
thABp3=[thetaABp3(g) for g in gVals]
xTicks=np.arange(-10,12,2)
ftSize=16
plt.figure()
plt.plot(gVals/B,thABp1,label="$p=1$",color="red",linewidth=1.5)
plt.scatter(g1ValsAll/B,(thetaABp1ValsAll/np.pi)%2,color="red",s=5)
plt.plot(gVals/B,thABp2,label="$p=2$",color="green",linewidth=1.5)
plt.scatter(g2ValsAll/B,thetaABp2ValsAll/np.pi,color="green",s=5)
plt.plot(gVals/B,thABp3,label="$p=3$",color="blue",linewidth=1.5)
plt.scatter(g3ValsAll/B,thetaABp3ValsAll/np.pi,color="blue",s=5)
plt.xticks(xTicks)
# plt.hlines(y=-1, xmin=-n, xmax=n, linewidth=0.05, color='k',linestyles="dotted")
plt.xlim((-n,n))
plt.ylim((-1,1.1))
plt.xlabel("$g/B$",fontsize=ftSize)
plt.ylabel("AB phase$/\pi$",fontsize=ftSize)
plt.legend(loc="best")
plt.title("AB phase for $p=1,2,3$",fontsize=ftSize)
plt.savefig("ABPhaseNum2.png")
plt.close()
