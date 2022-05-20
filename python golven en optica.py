#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:22:52 2022

@author: Jeroen
"""

"""
script voor golf en optica.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import scipy.odr as odr


#%%
#we gebruiken een csv omdat een xlsx niet moeilijker met numpy importeert.
filename = 'metingen eigen frequenties koper 0.2 - Sheet1.csv'
koper_2 = np.genfromtxt(filename, delimiter=',')

ef1k2 = koper_2[:,0]
ef2k2 = koper_2[:,1]
ef3k2 = koper_2[:,2]
t = np.linspace(1,8, 8)
f_err = 1 # onzekerheid is 1 Hz

plt.figure()
plt.errorbar(t, ef1k2, xerr=0.0, yerr=f_err, fmt='r.', label= 'meting 1')
plt.errorbar(t, ef2k2, xerr=0.0, yerr=f_err, fmt='b.', label= 'meting 2')
plt.errorbar(t, ef3k2, xerr=0.0, yerr=f_err, fmt='g.', label= 'meting 3')

# opmaak
plt.title('Koperen draad 0.2 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
plt.show()

#%%
#we gebruiken een csv omdat een xlsx niet moeilijker met numpy importeert.
filename = 'metingen eigen frequenties koper 0.3 - Sheet1.csv'
koper_3 = np.genfromtxt(filename, delimiter=',')

ef1k3 = koper_3[:,0]
ef2k3 = koper_3[:,1]
ef3k3 = koper_3[:,2]
t = np.linspace(1,8, 8)
f_err = 1 # onzekerheid is 1 Hz

plt.figure()
plt.errorbar(t, ef1k3, xerr=0.0, yerr=f_err, fmt='r.', label= 'meting 1')
plt.errorbar(t, ef2k3, xerr=0.0, yerr=f_err, fmt='b.', label= 'meting 2')
plt.errorbar(t, ef3k3, xerr=0.0, yerr=f_err, fmt='g.', label= 'meting 3')

# opmaak
plt.title('Koperen draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
plt.show()

#%%
#we gebruiken een csv omdat een xlsx niet moeilijker met numpy importeert.
filename = 'metingen eigen frequenties koper 0.4 - Sheet1.csv'
koper_4 = np.genfromtxt(filename, delimiter=',')

ef1k4 = koper_4[:,0]
ef2k4 = koper_4[:,1]
ef3k4 = koper_4[:,2]
t = np.linspace(1,8, 8)
f_err = 1 # onzekerheid is 1 Hz

plt.figure()
plt.errorbar(t, ef1k4, xerr=0.0, yerr=f_err, fmt='r.', label= 'meting 1')
plt.errorbar(t, ef2k4, xerr=0.0, yerr=f_err, fmt='b.', label= 'meting 2')
plt.errorbar(t, ef3k4, xerr=0.0, yerr=f_err, fmt='g.', label= 'meting 3')

# opmaak
plt.title('Koperen draad 0.4 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
plt.show()

#%%
#we gebruiken een csv omdat een xlsx moeilijk met numpy importeert.
filename = 'metingen eigen frequenties nikkel 0.3 - Sheet1.csv'
nikkel_3 = np.genfromtxt(filename, delimiter=',')

ef1n3 = nikkel_3[:,0]
ef2n3 = nikkel_3[:,1]
ef3n3 = nikkel_3[:,2]
t = np.linspace(1,8, 8)

f_err = 1 # onzekerheid is 1 Hz

plt.figure()
plt.errorbar(t, ef1n3, xerr=0.0, yerr=f_err, fmt='r.', label= 'meting 1')
plt.errorbar(t, ef2n3, xerr=0.0, yerr=f_err, fmt='b.', label= 'meting 2')
plt.errorbar(t, ef3n3, xerr=0.0, yerr=f_err, fmt='g.', label= 'meting 3')

# opmaak
plt.title('Nikkel draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
plt.show()

#%% gemiddeldes
# zo vinden we gemiddeldes per eigenfreq per draad
efk2 = (ef1k2+ef2k2+ef3k2)/3
efk3 = (ef1k3+ef2k3+ef3k3)/3
efk4 = (ef1k4+ef2k4+ef3k4)/3
efn3 = (ef1n3+ef2n3+ef3n3)/3

#%%  onzekerheiden van meetwaardes

#hierberekenen we alle standaardeviaties per meetpunt en doen het in een array
bruh = np.array([0,1,2,3,4,5,6,7])
o1 = []
for i in bruh:
    d = i
    a = ef1k2[d]
    b = ef2k2[d]
    c = ef3k2[d]
    stdi = np.std([a,b,c])
    o1.append(stdi)
    
# hier berekenen we de standaareviaties van het gemiddelde
SDOMk2 = np.array(o1)/((3)**(1/2))

o2 = []
for i in bruh:
    d = i
    a = ef1k3[d]
    b = ef2k3[d]
    c = ef3k3[d]
    stdi = np.std([a,b,c])
    o2.append(stdi)
    
SDOMk3 = np.array(o2)/((3)**(1/2))
o3 = []
for i in bruh:
    d = i
    a = ef1k4[d]
    b = ef2k4[d]
    c = ef3k4[d]
    stdi = np.std([a,b,c])
    o3.append(stdi)
    
SDOMk4 = np.array(o3)/((3)**(1/2))
o4 = []
for i in bruh:
    d = i
    a = ef1n3[d]
    b = ef2n3[d]
    c = ef3n3[d]
    stdi = np.std([a,b,c])
    o4.append(stdi)
    
SDOMn3 = np.array(o4)/((3)**(1/2))



#%% ideale fit voor kopere draad van 0.2 mm (k2)
#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efk2


#alle onzekerheden
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
#doordat we de meting 3 keer hebben uitgevoerd wordt de onzekerheid kleiner.
sig_y = SDOMk2
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.', label='gemiddelde meetwaarden')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x,freq):
    ef = x*freq
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])


odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De gereduceerde chi-kwadraat van k2 is:', chi2red)

plt.title('fit ideale snaar van koperen draad 0.2 mm ')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f([par_best[0]],xplot),'r-')

#%% ideale fit voor kopere draad van 0.3 mm (k3)

#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efk3

#alle onzekerheden
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
sig_y = np.array([1,1,1,1,1,1,1,1])/3
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.',label='gemiddelde meetwaarden')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x,freq):
    ef = x*freq
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])


odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.

print('De gereduceerde chi-kwadraat van k3 is:', chi2red)
plt.title('fit ideale snaar van koperen draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f([par_best[0]],xplot),'r-')
#%% ideale fit voor kopere draad van 0.4 mm (k4)

#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efk4

#alle onzekerheden
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
sig_y = np.array([1,1,1,1,1,1,1,1])/3
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.', label='gemiddelde meetwaarden')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x,freq):
    ef = x*freq
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])


odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De gereduceerde chi-kwadraat van k4 is:', chi2red)
plt.title('fit ideale snaar  van koperen draad 0.4 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f([par_best[0]],xplot),'r-')
#%% ideale fit voor nikkel draad van 0.3 mm (n3)

#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efn3

#alle onzekerheden
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
sig_y = np.array([1,1,1,1,1,1,1,1])/3
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.',label='gemiddelde meetwaarden')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x,freq):
    ef = x*freq
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])


odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De gereduceerde chi-kwadraat van n3 is:', chi2red)
plt.title('fit ideale snaar van nikkel draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f([par_best[0]],xplot),'r-')

#%% lower en upper waardes voor in de young modulus
m = 0.160 # massa aan draad in kg +- 0.0002 kg
L = 1.453 # lengte draad in meter +- 0.005
g = 9.81  # zwaartekrachtsversnelling in m/s^2 zonder onzekerheid
Pk = 8960 # dichtheid koper in kg/m3
Pn = 8902 # dichtheid nikkel in kg/m3
Yk_mi = 121*10**9  # minimale youngs modulus van koper in Gpa
Yk_ma = 133*10**9  # maximale youngs modulus van koper in Gpa
Yn_mi = 190*10**9 # minimale youngs modulus van nikkel in Gpa
Yn_ma = 220*10**9 # minimale youngs modulus van nikkel in Gpa
sig_r = 0.000005 # onzekerheid van de straal van draad

f_upperk2 = (1/(2*(L-0.005)))*((((m+0.0002)*g))/(np.pi*Pk*(0.0001-sig_r)**2))**(1/2)
f_upperk3 = (1/(2*(L-0.005)))*((((m+0.0002)*g))/(np.pi*Pk*(0.00015-sig_r)**2))**(1/2)
f_upperk4 = (1/(2*(L-0.005)))*((((m+0.0002)*g))/(np.pi*Pk*(0.0002-sig_r)**2))**(1/2)
f_uppern3 = (1/(2*(L-0.005)))*((((m+0.0002)*g))/(np.pi*Pk*(0.00015-sig_r)**2))**(1/2)

f_lowerk2 = (1/(2*(L+0.005)))*((((m-0.0002)*g))/(np.pi*Pk*(0.0001+sig_r)**2))**(1/2)
f_lowerk3 = (1/(2*(L+0.005)))*((((m-0.0002)*g))/(np.pi*Pk*(0.00015+sig_r)**2))**(1/2)
f_lowerk4 = (1/(2*(L+0.005)))*((((m-0.0002)*g))/(np.pi*Pk*(0.0002+sig_r))**2)**(1/2)
f_uppern3 = (1/(2*(L+0.005)))*((((m-0.0002)*g))/(np.pi*Pk*(0.00015+sig_r)**2))**(1/2)

B_upperk2 = (np.pi**3)*((0.0001-sig_r)**(4))*Yk_ma/(4*(m-0.0002)*9.81*(L-0.005)**2)
B_upperk3 = (np.pi**3)*((0.00015-sig_r)**(4))*Yk_ma/(4*(m-0.0002)*9.81*(L-0.005)**2)
B_upperk4 = (np.pi**3)*((0.0002-sig_r)**(4))*Yk_ma/(4*(m-0.0002)*9.81*(L-0.005)**2)
B_uppern3 = (np.pi**3)*((0.00015-sig_r)**(4))*Yn_ma/(4*(m-0.0002)*9.81*(L-0.005)**2)

B_lowerk2 = (np.pi**3)*((0.0001+sig_r)**(4))*Yk_ma/(4*(m+0.0002)*9.81*(L+0.005)**2)
B_lowerk3 = (np.pi**3)*((0.00015+sig_r)**(4))*Yk_ma/(4*(m+0.0002)*9.81*(L+0.005)**2)
B_lowerk4 = (np.pi**3)*((0.0002+sig_r)**(4))*Yk_ma/(4*(m+0.0002)*9.81*(L+0.005)**2)
B_lowern3 = (np.pi**3)*((0.00015+sig_r)**(4))*Yn_ma/(4*(m+0.0002)*9.81*(L+0.005)**2)

#%% fit voor kopere draad van 0.2mm (k2) met youngs modulus met upper and lower bounds

#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efk2

#alle onzekerheden
sig_x = np.array([1,1,1,1,1,1,1,1])*10**(-20)
sig_y = SDOMk2
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x, f_upperk2):
    ef = x*f_upperk2*(1+(x**(2))*B_upperk2)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De upper gereduceerde chi-kwadraat van k2 is:', chi2red)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'r-',label= 'fit met upperbounds')

# =============================================================================
# zelfde als hier boven maar lower bounds
# =============================================================================
	
# de functie/relatie
def f(x, f_lowerk2):
    ef = x*f_lowerk2*(1+(x**(2))*B_lowerk2)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De lower gereduceerde chi-kwadraat van k2 is:', chi2red)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'b--', label= 'fit met lowerbounds')
plt.title('fit niet ideale snaar van nikkel draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')

#%% fit voor kopere draad van 0.3mm (k3) met youngs modulus met upper and lower bounds

#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efk3

#alle onzekerheden
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
sig_y = np.array([1,1,1,1,1,1,1,1])/3
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.',label='gemiddelde meetwaarden')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x, f_upperk3):
    ef = x*f_upperk3*(1+(x**(2))*B_upperk3)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De upper gereduceerde chi-kwadraat van k3 is:', chi2red)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'r-', label= 'fit met upperbounds')

# =============================================================================
# zelfde als hier boven maar lower bounds
# =============================================================================
	
# de functie/relatie
def f(x, f_lowerk3):
    ef = x*f_lowerk3*(1+(x**(2))*B_lowerk3)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De lower gereduceerde chi-kwadraat van k3 is:', chi2red)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'b--', label= 'fit met lowerbounds')
plt.title('fit niet ideale snaar van nikkel draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')

#%% fit voor kopere draad van 0.4mm (k4) met youngs modulus met upper and lower bounds

#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efk4

#alle onzekerheden
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
sig_y = np.array([1,1,1,1,1,1,1,1])/3
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.',label='gemiddelde meetwaarden')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x, f_upperk4):
    ef = x*f_upperk4*(1+(x**(2))*B_upperk4)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De upper gereduceerde chi-kwadraat van k4 is:', chi2red)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'r-', label= 'fit met upperbounds')

# =============================================================================
# zelfde als hier boven maar lower bounds
# =============================================================================
	
# de functie/relatie
def f(x, f_lowerk4):
    ef = x*f_lowerk4*(1+(x**(2))*B_lowerk4)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De lower gereduceerde chi-kwadraat van k4 is:', chi2red)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'b--', label= 'fit met lowerbounds')
plt.title('fit niet ideale snaar van nikkel draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
#%% fit voor nikkel draad van 0.3mm (23) met youngs modulus met upper and lower bounds

#nummer van eigenfreq van betreffende draad
x = np.array([1,2,3,4,5,6,7,8]) 
#gevonden eigenfrequenties
y = efn3

#alle onzekerheden
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
sig_y = np.array([1,1,1,1,1,1,1,1])/3
	
# Dit is een niet opgemaakte sub plot voor de stippen
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.',label='gemiddelde meetwaarden')
#ax.show()
freq_start=0
	
# de functie/relatie
def f(x, f_uppern3):
    ef = x*f_uppern3*(1+(x**(2))*B_uppern3)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De upper gereduceerde chi-kwadraat van n3 is:', chi2red)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'r-',label= 'fit met upperbounds')

# =============================================================================
# zelfde als hier boven maar lower bounds
# =============================================================================
	
# de functie/relatie
def f(x, f_lowern3):
    ef = x*f_lowern3*(1+(x**(2))*B_lowern3)**(1/2)
    return ef
	
#model-object dat we gebruiken in odr
odr_model = odr.Model(f)
	
#real data opbject
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
# start waarde voor parameters
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
odr_res   = odr_obj.run()
	
# de beste schatters voor de parameters
par_best = odr_res.beta

# de onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta

# covariantiematrix
par_cov_ext = odr_res.cov_beta 

# De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var

# Een compacte weergave van de resultaten als output
# odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset met opmaak
# En printen we de gereduceerde chi kwadraat uit.
print('De lower gereduceerde chi-kwadraat van n3 is:', chi2red)

xplot=np.linspace(1,8, 8)
ax.plot(xplot, f(xplot,[par_best[0]]),'b--', label= 'fit met lowerbounds')
plt.title('fit niet ideale snaar van nikkel draad 0.3 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')