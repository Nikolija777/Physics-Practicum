# # Scientific Computing
# ## 10.3.2022.
# ### __NUMPY__   

# In[199]:


import numpy as np


# Metode za formiranje nizova i prikaz najvažnijih atributa ndarray objekta.

# In[200]:


a=np.array([[1,2,3],[3,4,5]])


# In[201]:


a


# ndarray.ndim → rang tj. broj osa (dimenzija)

# In[202]:


a.ndim


# ndarray.shape → dimenzija niza, torka (n, m) 

# In[203]:


a.shape


# ndarray.size → ukupan broj elemenata u nizu

# In[204]:


a.size


# ndarray.dtype → tip elemenata (NumPy tipovi num.int32...)

# In[205]:


a.dtype


# Specificiranje tipa prilikom kreiranja niza

# In[206]:


c=np.array([[1,2],[2,3]], dtype=complex)


# In[207]:


c


# In[208]:


np.zeros((2,3))


# In[209]:


np.empty((2,3))


# Kreiranje ekvidistantne sekvence brojeva - funcije arange i linspace.

# In[210]:


np.arange(5,35,6)


# In[211]:


from numpy import pi
x=np.linspace(0, 2*pi, 100)
f=np.sin(x)
x[-1]


# Aritmeričke operacije - izvršavaju se nad elementima.

# In[212]:


a=np.array([10,20,30,40])
b=np.arange(4)
b


# In[213]:


c=a-b


# In[214]:


c


# In[215]:


b**2


# In[216]:


a<39


# In[217]:


sum(a<39)


# Unarne operacije -   implementirane su kao
# metode ndarray klase:

# In[218]:


a = np.random.random((2,3))
a


# In[219]:


a.max()


# In[220]:


a.min()


# In[221]:


a.sum()


# In[ ]:





# Univerzalne funkcije - izvršavaju se nad elementima.

# In[222]:


B = np.arange(3)
B


# In[223]:


np.exp(B)


# In[224]:


C = np.array([2., -1., 4.])
np.add(B, C)


# In[225]:


B+C


# 
# 
# ### __Grafički prikaz: matplotlib.pyplot__

# In[226]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])#prosledimo dimenzije x i y ose (prva dva broja x-osa, druga dva y)
plt.ylabel('y osa')
plt.show()


# In[227]:


import matplotlib.pyplot as plt
t=np.arange(0,7,0.5)
plt.plot(t,t,'y*',t,t**2,'bs',t,t**3,'g^')
plt.xlabel('x-osa')
plt.ylabel('y-osa')
plt.show()


# In[228]:


def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

t1=np.arange(0,7,0.1)
t2=np.arange(0,8,0.05)
plt.figure(1,figsize=(8,6),dpi=80)
plt.subplot(1,2,1)
plt.plot(t1,f(t1),'bo',t2,f(t2),'r-')
plt.subplot(1,2,2)
plt.plot(t2,np.cos(2*np.pi*t2),'r-')
plt.show()


# # Diferencijalne jednačine
# ### __Primer 1: Kolonija bakterija__
# 
# 

# Jedna kolonija od 1000 bakterija razmnožava se brzinom od r = 0.8 jedinki na 1 sat. Koliko će biti bakterija u koloniji nakon 10 sati?
# 

# #### Korak 1:                  
# 
# <br>
# <ul>
#     <li>Proces se može modelovati diferencijalnom jednačinom oblika: dN/dt = r ∙ N, gde je N broj jedinki u populaciji.</li>
#     <li>Početni uslov je definisan sa: N(0) = 1000.</li>
#     <li>Za datu jednačinu postoji analitičko rešenje:  N(t) = N(0) ∙ exp(r∙t)</li>
# </ul>

# #### Korak 2:     
# 
# <br>
# <ul>
#     <li> formiranje funkcijske datoteke na osnovu funkcije zapisane u obliku:
# dN/dt = r ∙N,</li>
# 
# </ul>

# In[229]:


def dNdt(N, t):
    return 0.8*N


# #### Korak 3:                  
# 
# <br>
# <ul>
#     <li>Izbor integratora.</li>
# </ul>
# 
# #### Korak 4:                  
# 
# <br>
# <ul>
#     <li>Definisanje vremenskog intervala i pokretanje solvera za rešavanje diferencijalne jednačine.</li>
#     <li>Glavni program</li>
# </ul>

# In[230]:


from scipy import integrate

tpoc, tkraj = 0, 10
vreme=np.linspace(tpoc,tkraj,100) #poslednji broj oznacava broj tacaka iz datog intervala
N0=1000
NodT=integrate.odeint(dNdt,N0,vreme) #resavanje dif jna prvog reda
#analitcko res
Nanalit=N0*np.exp(0.8*vreme)


# In[231]:


plt.figure(figsize=(8,6),dpi=100)
plt.plot(vreme, NodT, label='numericki',color='blue',linewidth=2, linestyle='--')
plt.plot(vreme, Nanalit, label='analiticki',color='red',linewidth=1, linestyle='-')
plt.xlabel('vreme [h]')
plt.ylabel('broj bakterija')
plt.legend()


# # Primer 2: Koncentracija leka
# 

# 
# Koncentracija leka u organizmu (u slučaju uprošćenog modela) menja se po diferencijalnoj jednačini:
# 
# <br>
# \begin{equation}
#     dc/dt  = - c(t)/τ\\
# \end{equation}
# gde je $τ$ vremenska konstanta koja modeluje rastvorljivost leka.<br>
# 
# Analitičko rešenje može se zapisati u obliku:
# <br>
# \begin{equation}
#     c(t) = c_0exp(-t/τ)\\
# \end{equation}
# <br>

# In[232]:


def dcdt(c, t, tau):
    return -c/tau
tpoc, tkraj= 0, 10
vreme=np.linspace(tpoc,tkraj,100)
c0=100
tau=2
#numericko
Cnum=integrate.odeint(dcdt,c0,vreme, args=(tau,))
#analiticko
Can= c0*np.exp(-vreme/tau)
#prikaz rez
plt.figure(figsize=(8,6), dpi=80)
plt.plot(vreme,Cnum,color='blue',linewidth=1.5,linestyle='-')
plt.plot(vreme, Can, color='pink', linewidth=3, linestyle='--')
plt.xlabel('vreme [h]')
plt.ylabel('koncentracija leka')
plt.grid()
plt.show()


# Zanimljivost: šta se dešava kada se uzimaju redovne doze, recimo 10 doza na svaka 2 sata. Kako izgleda promena koncentracije leka u periodu od 24 sata? Da li dolazi do nagomilavanja koncentracije leka u organizmu? Primeniti analitičko rešenje:
# 
# <br>
# \begin{equation}
#     c(t) = c_0exp(-t/τ)+h(t-τ)c_0exp([-(t-τ)]/τ)++h(t-2τ)c_0exp([-(t-2τ)]/τ)\\
# \end{equation}
# <br>
# 
# gde je $h(t-τ)$ Hevisajdova step funkcija.
# 

# In[233]:


def heaviside(x):
    return 0.5*(np.sign(x)+1)

import numpy as np
tpoc, tkraj = 0, 24
vreme=np.linspace(tpoc,tkraj,1000)
c0,tau=100,2
conc=c0*np.exp(-vreme/tau)
for i in range(10):
    conc+=heaviside(vreme-(i+1)*tau)*c0*np.exp(-(vreme-(i+1)*tau)/tau)
plt.figure(figsize=(8,6), dpi=80)
plt.plot(vreme,conc,color='blue',linewidth=1.5,linestyle='-')
plt.xlabel('vreme [h]')
plt.ylabel('koncentracija leka')
plt.grid()
plt.show()


# ### __Zadatak 1__: 
# Posmatra se telo mase $m$ vezano za idealnu oprugu krutosti $k$, koje se kreće po glatkoj horizontalnoj podlozi duž $x$ ose - linearni harmonijski oscilator. Poznate su vrednosti sopstvene kružne učestanosti $\omega_0$, početne pozicije u kojoj se nalazi telo $x_0$ i početne brzine $v_0$.

# a) Formirati Python definiciju __LHO__ koja modeluje zadati problem i omogućava rešavanje diferencijalne jednačine primenom integrate.odeint metode. Funkcija poziva parametar $\omega_0$.

# In[243]:


def LHO (y,t,omega0):
    y1,y2=y
    return [y2,(-omega0**2)*y1]


# b) Napisati komande koje uvoze modul numpy, modul matplotlib.pyplot i funkciju za integraciju iz scipy modula.

# In[244]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# c) Definisati listu četiri različite vrednosti koje uzima sopstvena kružna učestanost:  $ω_0 = 0,5$ rad/s, $ω_0 = 1$ rad/s, $ω_0 = 2$ rad/s i $ω_0 = 4$ rad/s. <br>
# Definisati početne uslove: telo u početnom trenutku miruje na rastojanju $x_0 = 5$ m.<br>
# Definisati vremensku osu: $0$ do $20$ sekundi u $1000$ ekvidistantnih tačaka.

# In[245]:


omega0=[0.5,1,2,4]
pUslov=[5,0]
t=np.linspace(0,20,1000)


# d) Napisati kod koji omogućava da se u okviru istog Figure prozora korišćenjem naredbe subplot, iscrtaju vremenski odzivi $x(t)$ za sve četiri vrednosti kružne učestanosti. Označiti ose grafika (“vreme [s]” i “x(t) [m]”). Na graficima prikazati legendu koja se odnosi na odgovarajuću vrednost kružne učestanosti $ω_0$.<br>
# 

# In[250]:


for index, value in enumerate(omega0):
    sol=integrate.odeint(LHO,pUslov,t,args=(value,))
    plt.figure(1,figsize=(8,6), dpi=150)
    plt.subplot(2,2,index+1)
    plt.plot(t, sol[:,0], label='omega0='+str(value))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.xlabel('vreme [s]')
    plt.ylabel('x(t) [m]')
    plt.legend()
   


# e) Napisati kod koji omogućava da se u okviru istog Figure prozora korišćenjem naredbe subplot, iscrtaju fazni dijagrami za sve četiri vrednosti kružne učestanosti. Označiti ose grafika (“x [m]” i “v(t) [m/s]”). Na graficima prikazati legendu koja se odnosi na odgovarajuću vrednost kružne učestanosti $ω_0$.<br>.

# In[251]:


for index, value in enumerate(omega0):
    sol=integrate.odeint(LHO,pUslov,t,args=(value,))
    plt.figure(1,figsize=(8,6), dpi=150)
    plt.subplot(2,2,index+1)
    plt.plot(sol[:,0], sol[:,1], label='omega0='+str(value))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.xlabel('vreme [s]')
    plt.ylabel('v [m/s]')
    plt.legend()


# ### __Zadatak 2__: 
# 
# 
# Za LHO mase $m = 1$ kg, koji osciluje na kružnoj frekvenciji $ω_0 = 0,5$ rad/s, prikazati zavisnost:
# <br>
# <ul>
#     <li>kinetičke i potencijalne energije od vremena;</li>
#     <li> kinetičke i potencijalne energije od koordinate;</li>
#     <li> ukupne energije LHOa od vremena.</li>
# </ul>
# U početnom trenutku telo miruje na rastojanju $x_0 = 5$ m.<br>

# ### __Zadatak 2__: 
# 
# 
# Za LHO mase $m = 1$ kg, koji osciluje na kružnoj frekvenciji $ω_0 = 0,5$ rad/s, prikazati zavisnost:
# <br>
# <ul>
#     <li>kinetičke i potencijalne energije od vremena;</li>
#     <li> kinetičke i potencijalne energije od koordinate;</li>
#     <li> ukupne energije LHOa od vremena.</li>
# </ul>
# U početnom trenutku telo miruje na rastojanju $x_0 = 5$ m.<br>

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




