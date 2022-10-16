import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#this script plots dynamical and AB phases for arbitraty p

B=2
p=1
inDir="./arbitraryp"+str(p)+"/"
inData=pd.read_csv(inDir+"truep"+str(p)+".csv")

gValsAll=np.array(inData.iloc[:,0])
thetaDValsAll=np.array(inData.iloc[:,1])
ABValsAll=np.array(inData.iloc[:,2])

ftSize=17
xTicks=np.arange(-10,12,2)

fig1, ax1 = plt.subplots()
ax1.set_title("$p=$"+str(p))
ax1.scatter(gValsAll/B,thetaDValsAll/np.pi,color="black",s=2)
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