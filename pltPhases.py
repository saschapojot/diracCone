import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#this script plots phases without modulo 2pi

p=4
B=2

inDir="./p"+str(p)+"/"

inFile=inDir+"p"+str(p)+".csv"

inData=pd.read_csv(inFile)
gValsAll=np.array(inData.iloc[:,0])

thetaDVals=np.array(inData.iloc[:,1])
ABVals=np.array(inData.iloc[:,2])


ftSize=17
s=10
plt.figure()
plt.scatter(gValsAll/B,thetaDVals/np.pi,color="black",s=2)
plt.xlabel("$g/B$",fontsize=ftSize)
plt.ylabel("dynamical phase$/\pi$",fontsize=ftSize)
plt.savefig(inDir+"p"+str(p)+"dynamicalPhase.png")
plt.close()

plt.figure()
plt.scatter(gValsAll/B,ABVals/np.pi,color="black",s=2)
plt.xlabel("$g/B$",fontsize=ftSize)
plt.ylabel("AB phase/$/\pi$",fontsize=ftSize)
plt.savefig(inDir+"p"+str(p)+"ABPhase.png")
plt.close()