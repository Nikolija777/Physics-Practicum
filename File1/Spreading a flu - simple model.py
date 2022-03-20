#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import matplotlib.pyplot as plt

beta = 0.0008


gamma = 0.0086
dt=0.1
T= int(30*24/dt)   

t=np.linspace(0,dt*T,T+1)
S=np.zeros(T+1)
I=np.zeros(T+1)
R=np.zeros(T+1)
# Initial conditions
S[0]=80
I[0]=1
R[0]=0

for i in range(T):
    S[i+1]=S[i]-beta*I[i]*S[i]*dt
    R[i+1]=R[i]+gamma*I[i]*dt
    I[i+1]=I[i]+beta*I[i]*S[i]*dt-gamma*I[i]*dt

#plotting results
fig = plt.figure()
plt.plot(t,S,color='blue',linewidth=1.5,linestyle='--')
plt.plot(t,I,color='red',linewidth=1.5,linestyle='--')
plt.plot(t,R,color='green',linewidth=1.5,linestyle='--')
plt.xlabel('time [h]')
plt.ylabel('S,I,R')
plt.legend(["S","I","R"])
plt.show()
fig.savefig('graph.png', bbox_inches='tight')


# In[ ]:




