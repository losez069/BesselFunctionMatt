import math
import matplotlib.pyplot as plt
import cmath
import numpy as np
import scipy.special as sp
alpha=4.1*10**(-6)
wavelength = 1310*10**(-9)
m=1
n1 = 1.47093878 
n2 = 1.46426121
c=3*10**8
w = 2*math.pi*c/wavelength
Beta = np.linspace(w*n2/c,w*n1/c,100)
Phi0 = math.pi/6
v=m*math.pi/Phi0
Beta_List = []
TM_List = np.array([])
TE_List = np.array([])
for i in range(1,len(Beta)+1):
    Beta_value = Beta[i-1]
    Beta_t=cmath.sqrt( (w**2/c**2) * n1**2 - Beta_value**2 )
    q_t = cmath.sqrt(Beta_value**2-w**2/c**2*n2**2)
    if q_t != 0 and Beta_t !=0:  
        Beta_List.append(Beta_value)        
        x=Beta_t*alpha
        y=q_t*alpha
        TM = np.array([[1, -1,-1], [1/Beta_t**2, 1/q_t**2,1/q_t**2],[n1**2*sp.jvp(v,x)/(Beta_t*sp.jv(v,x)),n2**2*sp.kvp(v,y)/(q_t*sp.kv(v,y)),n2**2*sp.ivp(v,y)/(q_t*sp.iv(v,y))]])
        TM_determinant = np.linalg.det(TM)
        TM_List=np.append(TM_List,TM_determinant)
        TE = np.array([[1, -1,-1], [1/Beta_t**2, 1/q_t**2,1/q_t**2],[sp.jvp(v,x)/(Beta_t*sp.jv(v,x)),sp.kvp(v,x)/(q_t*sp.kv(v,y)),sp.ivp(v,x)/(q_t*sp.iv(v,y))]])
        TE_determinant = np.linalg.det(TE)
        TE_List=np.append(TE_List,TE_determinant)
        
print(TE_List)

print(TM_List)
fig = plt.figure(1)
plt.plot(Beta_List,TM_List, 'g')
plt.title('TM mode Beta versus Determinant')
plt.xlabel('Beta')
plt.ylabel('Determinant')

fig = plt.figure(2)
plt.plot(Beta_List,TE_List, 'r')
plt.title('TE mode Beta versus Determinant')
plt.xlabel('Beta')
plt.ylabel('Determinant')

plot2 = plt.figure(3)
x=TM_List.real
y=TM_List.imag
plt.plot(x,y,'o', color='black')
plt.title('TM mode Determinant')
plt.xlabel('Real')
plt.ylabel('Imaginary')

plot2 = plt.figure(4)
x=TE_List.real
y=TE_List.imag
plt.plot(x,y,'o', color='black')
plt.plot(x,y,'o', color='black')
plt.title('TE mode Determinant')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()