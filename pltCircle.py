import numpy as np
import matplotlib.pyplot as plt

angle = np.linspace(0, 2 * np.pi, 150)

r = 0.4

x = r * np.cos(angle)
y = r * np.sin(angle)

figure, ax = plt.subplots(1)
ftSize=16
ax.plot(x, y,color="blue",linewidth=2)
plt.plot([0,r],[0,0],color="black",linewidth=0.5)
alpha=np.pi/10
plt.plot([0,r*np.cos(alpha)],[0,r*np.sin(alpha)],color="black",linewidth=0.5)
s=r/1.5
plt.text(s*np.cos(alpha/3),s*np.sin(alpha/3),"$\lambda$")

ax.set_aspect(1)

#O
Dp=plt.scatter(0,0,color="black",s=15,label="Origin")

dis=0.2


#W
plt.scatter(-r,0,s=16,color="red")
plt.text(-r-dis,0,"W",fontsize=ftSize)

#E
plt.scatter(r,0,s=16,color="red")
plt.text(r+dis,0,"E",fontsize=ftSize)


#N
plt.scatter(0,r,s=16,color="red")
plt.text(0,r+dis,"N",fontsize=ftSize)

#S
plt.scatter(0,-r,s=16,color="red")
plt.text(0,-r-dis,"S",fontsize=ftSize)

dTheta=0.01*np.pi
#arrow 1
theta1=-np.pi/4
x1=r*np.cos(theta1)
y1=r*np.sin(theta1)
dx1=-r*np.sin(theta1)*dTheta
dy1=r*np.cos(theta1)*dTheta
arr1=plt.arrow(x1,y1,dx1,dy1, shape='full', lw=0, length_includes_head=True, head_width=.05,color="green")

#arrow 2
theta2=np.pi/4
x2=r*np.cos(theta2)
y2=r*np.sin(theta2)
dx2=-r*np.sin(theta2)*dTheta
dy2=r*np.cos(theta2)*dTheta
plt.arrow(x2,y2,dx2,dy2, shape='full', lw=0, length_includes_head=True, head_width=.05,color="green")

#arrow 3
theta3=3/4*np.pi
x3=r*np.cos(theta3)
y3=r*np.sin(theta3)
dx3=-r*np.sin(theta3)*dTheta*(-1)
dy3=r*np.cos(theta3)*dTheta*(-1)
arr3=plt.arrow(x3,y3,dx3,dy3, shape='full', lw=0, length_includes_head=True, head_width=.05,color="magenta")

#arrow 4
theta4=-3/4*np.pi
x4=r*np.cos(theta4)
y4=r*np.sin(theta4)
dx4=-r*np.sin(theta4)*dTheta*(-1)
dy4=r*np.cos(theta4)*dTheta*(-1)
plt.arrow(x4,y4,dx4,dy4,shape='full', lw=0, length_includes_head=True, head_width=.05,color="magenta")

plt.xlim((-1,1))
plt.ylim((-1,1))
plt.xticks([0])
plt.yticks([0])

plt.xlabel("$k_{1}$",fontsize=ftSize)
plt.ylabel("$k_{2}$",fontsize=ftSize)
# plt.legend(loc="best")
plt.legend([arr3,arr1,Dp],["Path SWN","Path SEN","Origin"])
plt.title("Dynamical path",fontsize=ftSize)
plt.savefig("conePath.pdf")