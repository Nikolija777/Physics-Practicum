#damped oscillations
def prigusene(y,t,omega0,alpha):
    y1,y2 = y
#     dy1 = y2
#     dy2 = -np.power(omega0,2)*y1 -2*alpha*y2
#     return[dy1,dy2]
    return[y2,-np.power(omega0,2)*y1 -2*alpha*y2]


# initial conditions
omega0 = 1.
alpha = [omega0, omega0/10, 15*omega0, 0.]
inits = [10, -13]
t = np.linspace(0,60,1000)


# position vs time
for index,value in enumerate(alpha):
    sol = integrate.odeint(prigusene,inits,t,args=(omega0,value))
    plt.figure(1,figsize=(20,10))
    plt.subplot(2,2,index+1)
    plt.plot(t,sol[:,0],label="alpha = "+str(value)+" 1/s")
    plt.xlabel("vreme [s]")
    plt.ylabel("x(t) [m]")
    plt.grid(b=True, which='both', color='grey', linestyle='--')
    plt.legend()
    plt.show()




omega0 = 1.
alpha = [omega0, omega0/10, 15*omega0, 0.]
inits = [10, 20]
t = np.linspace(0,60,1000)
print(np.shape(t))
for index,value in enumerate(alpha):
    sol = integrate.odeint(prigusene,inits,t,args=(omega0,value))
    plt.figure(1,figsize=(20,10))
    plt.subplot(2,2,index+1)
    plt.plot(sol[:,0],sol[:,1],label="alpha = "+str(value)+" 1/s")
    plt.xlabel("x(t) [m]")
    plt.ylabel("v(t) [m/s]")
    plt.grid(b=True, which='both', color='grey', linestyle='--')
    plt.legend()
    plt.show()


# kinetic vs potential energy
omega0 = 1.
m=1
alpha = omega0/10
inits = [10, 20]
t = np.linspace(0,60,1000)

sol = integrate.odeint(prigusene,inits,t,args=(omega0,alpha))
k=np.power(omega0,2)*m

Ek=0.5*m*np.power(sol[:,1],2)

Ep=0.5*k*np.power(sol[:,0],2)
plt.figure(3,figsize=(8,6),dpi=150)
plt.plot(t,Ek,color='blue',linewidth=1.5,linestyle='-')
plt.plot(t,Ep,color='red',linewidth=1.5,linestyle='--')
plt.xlabel("vreme[s]")
plt.ylabel("Ek i Ep [J]")
plt.legend(["Ek","Ep"])



# energy
omega0 = 1.
m=1
alpha = omega0/10
inits = [10, 20]
t = np.linspace(0,60,1000)
sol = integrate.odeint(prigusene,inits,t,args=(omega0,alpha))
k=np.power(omega0,2)*m

Ek=0.5*m*np.power(sol[:,1],2)
Ep=0.5*k*np.power(sol[:,0],2)
plt.figure(3,figsize=(8,6),dpi=150)
plt.plot(t,Ek+Ep,color='blue',linewidth=1.5,linestyle='-')
plt.xlabel("vreme[s]")
plt.ylabel("E[J]")
plt.legend(["E"])



# forced osscilations

def prinudne(y,t,omega0,alpha,F0,m,omega1):
    y1,y2 = y
    return[y2,-np.power(omega0,2)*y1 -2*alpha*y2+F0*np.sin(omega1*t)/m]


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# initial conditions
omega0=10.
alpha=omega0/20
F0=20.0
omega1=2.
m=1
puslov=[10,0]
t=np.linspace(0,60,6000)



# position vs time
sol=integrate.odeint(prinudne,puslov,t,args=(omega0,alpha,F0,m,omega1))
plt.figure(6,figsize=(8,6),dpi=80)
plt.plot(t,sol[:,0])
plt.xlabel('vreme [s]')
plt.ylabel('x(t) [m]')
plt.show()


# velocity vs position

sol=integrate.odeint(prinudne,puslov,t,args=(omega0,alpha,F0,m,omega1))
plt.figure(6,figsize=(8,6),dpi=80)
plt.plot(sol[:,0],sol[:,1])
plt.xlabel('x [m]')
plt.ylabel('v(t) [m/s]')
plt.show()




sol=integrate.odeint(prinudne,puslov,t,args=(omega0,alpha,F0,m,omega1))
plt.figure(6,figsize=(8,6),dpi=80)
plt.plot(sol[:,0],sol[:,1])
plt.xlabel('x [m]')
plt.ylabel('v(t) [m/s]')
plt.xlim([-0.5, 0.5])
plt.ylim([-4, 4])
plt.show()



# amplitude
omega1=np.arange(0,2*omega0,omega0/50)
A=(F0/m)/(np.power((omega0**2-omega1**2)**2+(2*alpha*omega1)**2,0.5))
A.shape
plt.plot(omega1,A)
plt.xlabel('učestanost prinudne sile')
plt.ylabel('amplituda prinudnih oscilacija')

# analytical solution
omega0 = 10.
alpha = omega0/20
f0 = 20.
m = 1.
inits = [10, 0]
t = np.linspace(0, 300, 2000)


for index, value in enumerate(omega1):
    sol = integrate.odeint(prinudne,puslov,t,args=(omega0,alpha, f0, m, value))
    A[index] = max(sol[200:300, 0])
    

plt.plot(omega, A, "r--")
plt.xlabel("ucestanost prinudne sile")
plt.ylabel("aplituda prinudnih oscilacija")
plt.show()


