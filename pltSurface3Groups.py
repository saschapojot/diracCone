import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



#plot energy surfaces for p=1, 1.5, 2

B=2
g=2.5*B
p1=1
inData1=pd.read_csv("./surfacep"+str(p1)+"/"+"g"+str(g/B)+"B.csv")
p2=1.5
inData2=pd.read_csv("./surfacep"+str(p2)+"/"+"g"+str(g/B)+"B.csv")
p3=2
inData3=pd.read_csv("./surfacep"+str(p3)+"/"+"g"+str(g/B)+"B.csv")
fig=plt.figure(figsize=plt.figaspect(1/3))
size=0.1
############################ p1
plt1k1=inData1.loc[:,"k1"]
plt1k2=inData1.loc[:,"k2"]
plt1E=np.array(inData1.loc[:,"E"])
ftSize=16
tickSize=14

ax1=fig.add_subplot(1,3,1,projection="3d")
ax1.scatter(plt1k1,plt1k2,plt1E/B,s=size,c='blue')

ax1.set_xlabel("$k_{1}/\pi$",fontsize=ftSize,labelpad=10)
ax1.set_ylabel("$k_{2}/\pi$",fontsize=ftSize,labelpad=10)
ax1.set_zlabel("$E/B$",fontsize=ftSize)
ax1.set_title("$p=$"+str(p1)+", $g/B=$"+str(g/B),fontsize=ftSize)
ax1.set_xticks([-0.1,0,0.1])
ax1.set_yticks([-0.1,0,0.1])
ax1.tick_params(axis='both', which='major', labelsize=tickSize)
x1=0.1
y1=-1.01
z1=46.1
ax1.text(x1,y1,z1,"(a)",transform=ax1.transAxes,size=ftSize-4)#numbering
ax1.azim = 60
ax1.dist = 10
ax1.elev = 7

#########################p2

plt2k1=inData2.loc[:,"k1"]
plt2k2=inData2.loc[:,"k2"]
plt2E=np.array(inData2.loc[:,"E"])
ftSize=16


ax2=fig.add_subplot(1,3,2,projection="3d")
ax2.scatter(plt2k1,plt2k2,plt2E/B,s=size,c='blue')

ax2.set_xlabel("$k_{1}/\pi$",fontsize=ftSize,labelpad=10)
ax2.set_ylabel("$k_{2}/\pi$",fontsize=ftSize,labelpad=10)
ax2.set_zlabel("$E/B$",fontsize=ftSize)
ax2.set_title("$p=$"+str(p2)+", $g/B=$"+str(g/B),fontsize=ftSize)
ax2.set_xticks([-0.1,0,0.1])
ax2.set_yticks([-0.1,0,0.1])
ax2.tick_params(axis='both', which='major', labelsize=tickSize)
x1=0.1
y1=-1.01
z1=50.1
ax2.text(x1,y1,z1,"(b)",transform=ax2.transAxes,size=ftSize-4)#numbering
ax2.azim = 60
ax2.dist = 10
ax2.elev = 7


###############p3

plt3k1=inData3.loc[:,"k1"]
plt3k2=inData3.loc[:,"k2"]
plt3E=np.array(inData3.loc[:,"E"])
ftSize=16


ax3=fig.add_subplot(1,3,3,projection="3d")
ax3.scatter(plt3k1,plt3k2,plt3E/B,s=size,c='blue')

ax3.set_xlabel("$k_{1}/\pi$",fontsize=ftSize,labelpad=10)
ax3.set_ylabel("$k_{2}/\pi$",fontsize=ftSize,labelpad=10)
ax3.set_zlabel("$E/B$",fontsize=ftSize)
ax3.set_title("$p=$"+str(p3)+", $g/B=$"+str(g/B),fontsize=ftSize)
ax3.set_xticks([-0.1,0,0.1])
ax3.set_yticks([-0.1,0,0.1])
ax3.tick_params(axis='both', which='major', labelsize=tickSize)
x1=0.1
y1=-1.01
z1=53.1
ax3.text(x1,y1,z1,"(c)",transform=ax3.transAxes,size=ftSize-4)#numbering
ax3.azim = 60
ax3.dist = 10
ax3.elev = 7













plt.savefig("3surfacesRow.pdf")