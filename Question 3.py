# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 18:36:01 2020

@author: Jonathan Beaulieu-Emond
"""

#Importation des modules nécessaires
import numpy as np
from numba import jit
from scipy.constants import *
from edos import an,am,ah,bn,bm,bh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numba import jit
from scipy.constants import pi
from scipy.optimize import curve_fit
#Fonction calculant le cote droit des EDO

@jit #Préfixe nécessaire pour numba(compilateur )
def g(t0,u,Ia) : # Calcule le prochain pas de temps
    global Cm,condition # Permet de définir ces variables en dehors de la fonction
    if t0>=1 and condition : # Condition permettant de couper le courant après une seconde
        Ia=0
    #Liste des paramètres
    Vk,Vna,Vl= -12,115,10.6
    Gk,Gna,Gl=36,120,0.3
    #Liste des variables
    V,n,m,h=u[0],u[1],u[2],u[3]
    
    #Calcul des équations différentielles
    gV=Ia-Gk*n**4*(V-Vk)-Gna*m**3*h*(V-Vna)-Gl*(V-Vl) # RHS Eq (1.68) 
    #print(Ia)
    gV=gV/Cm
    
    gN=an(V)*(1-n)-bn(V)*(n)
    gM=am(V)*(1-m)-bm(V)*(m)
    gH=ah(V)*(1-h)-bh(V)*(h)
    eval1=np.array([gV,gN,gM,gH]) # vecteur RHS (pentes) 
    return eval1


 # FONCTION CALCULANT UN SEUL PAS DE RUNGE-KUTTA D’ORDRE 4 
@jit
def rk(h,t0,uu,Ia): 
    g1=g(t0,uu,Ia) # Eq (1.15)
    g2=g(t0+h/2.,uu+h*g1/2.,Ia) # Eq (1.16) 
    g3=g(t0+h/2.,uu+h*g2/2.,Ia) # Eq (1.17)
    g4=g(t0+h,uu+h*g3,Ia) # Eq (1.18)
    unew=uu+h/6.*(g1+2.*g2+2.*g3+g4) # Eq (1.19)
    return unew

# OSCILLATIONS NONLINEAIRES: 4 EDOS NONLINEAIRES COUPLEES 
@jit
def main(A) :
    lambd_f=[]
    amplitude=[]
    Ia=A
    
    nMax=10000 # nombre maximal de pas de temps 
    eps =1.e-8 # tolerance
    tfin=100. # duree d’integration
    t=np.zeros(nMax) # tableau temps
    u=np.zeros([nMax,4]) # tableau solution
    u[0,:]=np.array([0,an(0)/(an(0)+bn(0)),am(0)/(am(0)+bm(0)),ah(0)/(ah(0)+bh(0))]) # condition initiale 26 
    
    nn=0 # compteur iterations temporelles 27 
    h=0.1 # pas initial 28 
    while (t[nn] < tfin) and (nn < nMax-1): # boucle temporelle 29 
        u1 =rk(h, t[nn],u[nn,:],Ia) # pas pleine longueur 30 
        u2a=rk(h/2.,t[nn],u[nn,:],Ia) # premier demi-pas 31 
        u2 =rk(h/2.,t[nn],u2a[:],Ia) # second demi-pas 32 
        delta=max(abs((u2[0]-u1[0])/u2[0]),abs(u2[1]-u1[1]),abs(u2[2]-u1[2]),abs(u2[3]-u1[3])) # Eq (1.42) 33 
        if delta > eps: # on rejette 34 
            h/=1.5 # reduction du pas 35 
        else: # on accepte le pas 36 
            nn=nn+1 # compteur des pas de temps 37 
            t[nn]=t[nn-1]+h # le nouveau pas de temps 38 
            u[nn,:]=u2[:] # la solution a ce pas 39 
            if delta <= eps/2.: h*=1.5 # on augmente le pas 40 
            #print("{0}, t {1}, V {2}, M {3} , N {4} , H{5} .".format(nn,t[nn],u[nn,0],u[nn,1],u[nn,2],u[nn,3]))
            # fin boucle temporelle
            # END
    c=np.where(t!=0)
    #ax1.plot(t[c],u[:,0][c],label='$I_{a}=$'+str(Ia),color='red')
    print(nn)
    x=t[c]
    y=u[:,0][c]
    point=find_peaks(y)[0] # point maximum
    
    
    m=x[point][1::]
    
    lambd=[]
    amplitude=(np.mean(y[point][1::])) # amplitude moyenne
    for j in range(0,len(m)-1):
        lambd.append(abs(m[j]-m[j+1])) # calcule la longueur d'onde moyenne
        
    lambd_f=np.mean(lambd)
    
    
    return lambd_f,amplitude



Iaa=np.arange(6.9,20,0.04)
#Iaa=[16]
Cm=1
condition=False # condition sur le courant I=0 apres une seconde
fig=plt.figure(dpi=1400)
ax1 = fig.subplots()
ax2 = ax1.twinx()
lambd_f=[]
amplitude=[]
#Boucle pour rouler le code avec différent courant
for i in Iaa :
    
    a,b=main(i)
    lambd_f.append(a)
    amplitude.append(b)
    
"""
plt.xlabel('temps en seonde')
ax1.set_ylabel('Voltage en $\mu$A ')
ax2.set_ylabel('probabilité n,l,m sans unité')
plt.title('Voltage en fonction du temps ')
ax1.axvspan(0,1, alpha=0.5, color='grey')
ax1.set_ylim(ymin=-20)
ax1.set_xlim(xmin=0)
ax2.set_ylim(ymin=0)
ax2.set_xlim(xmin=0)
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
#plt.legend()
plt.show()
"""
lambd_f=np.array(lambd_f)
lambd_f=2*pi/lambd_f
def f(x,a,b) :
    return a*x+b

plt.show()
plt.figure(dpi=1000)
plt.plot(Iaa,lambd_f,label='data')
#popt,pcov=curve_fit(f,Iaa,lambd_f)
results1,error1=np.polyfit(Iaa,lambd_f,3,cov=True)
plt.plot(Iaa,results1[0]*Iaa**3+results1[1]*Iaa**2+results1[2]*Iaa+results1[3],label='fit')
#plt.plot(Iaa,popt[0]*Iaa+popt[1])
plt.title('Fréqence en fonction du courant')
plt.xlabel('courant $\mu$A')
plt.ylabel('fréquence en Hertz')
plt.legend()
plt.show()
plt.figure(dpi=1000)
plt.plot(Iaa,amplitude,label='data')
#popt,pcov=curve_fit(f,Iaa,amplitude)
results2,error2=np.polyfit(Iaa,amplitude,3,cov=True)
plt.plot(Iaa,results2[0]*Iaa**3+results2[1]*Iaa**2+results2[2]*Iaa+results2[3],label='fit')
#plt.plot(Iaa,popt[0]*Iaa+popt[1])
plt.title('amplitude en fonction du courant')
plt.xlabel('courant $\mu A/cm^{2}$')
plt.ylabel('amplitude en mV')
plt.legend()
plt.show()
