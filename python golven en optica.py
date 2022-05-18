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


folder = "/Users/apple/Documents/GitHub/golf-en-optica-practicum-1/"

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
#we gebruiken een csv omdat een xlsx niet moeilijker met numpy importeert.
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

#%%
efk2 = (ef1k2+ef2k2+ef3k2)/3
efk3 = (ef1k3+ef2k3+ef3k3)/3
efk4 = (ef1k4+ef2k4+ef3k4)/3
efn3 = (ef1n3+ef2n3+ef3n3)/3


#%%
## (0) Importeer het ODR pakket (ODR = Orthogonal Distance Regression)
import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt
	
# Fictieve dataset x- en y-waarden, beide met onzekerheid
x = np.array([1,2,3,4,5,6,7,8])
sig_x = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
y = efk2
sig_y = np.array([1,1,1,1,1,1,1,1])
	
# We maken aan deze data een fit met de functie y = A + B x
# Het is goed om deze dataset te plotten, en op basis hiervan een schatting 
# te maken voor startwaarden voor de parameters A en B. Voor deze 
# specifieke dataset kunnen we als schatting voor A de waarde gemeten bij 
# de asafsnede nemen (0.5 dus), en voor B, de helling, ongeveer 1.5
# Met deze opzet kun je overigens eerst een plot maken met data (activeer 
# dan ax.show), en op een later moment de beste rechte lijn erbij plotten
# NB Dit is een niet opgemaakte plot
f,ax = plt.subplots(1)
ax.errorbar(x,y,xerr=sig_x,yerr=sig_y,fmt='k.')
#ax.show()
freq_start=0
	
## (1) Definieer een Python-functie die het model bevat, in dit geval een 
## rechte lijn
## B is een vector met parameters, in dit geval twee (A = B[0], B = B[1])
## x is de array met x-waarden
def f(x,freq):
    ef = x*freq
    return ef
	
## (2) Definieer het model-object om te gebruiken in odr
odr_model = odr.Model(f)
	
## (3) Definieer een RealData object
## Een RealData-object vraagt om de onzekerheden in beide richtingen. 
## !! De onzekerheid in de x-richting mag ook nul zijn (dan mag je 
## sx=0 weglaten), maar dan moet bij onderdeel (4)/(4a) wel gekozen 
## worden voor een kleinste-kwadratenfit !!
odr_data  = odr.RealData(x,y,sx=sig_x,sy=sig_y)
	
## (4) Maak een ODR object met data, model en startwaarden
## Je geeft startwaarden voor parameters mee bij keyword beta0
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[freq_start])
## (4a) Stel in op kleinste kwadraten (optioneel)
## Als de onzekerheden in de x-richting gelijk zijn aan nul, dan faalt de 
## default-aanpak van ODR. Je moet dan als methode een kleinste-kwadraten- 
## fit instellen. Dit gaat met het volgende commando (nu uit):
#odr_obj.set_job(fit_type=2)
	
## (5) Voer de fit uit
## Dit gebeurt expliciet door functie .run() aan te roepen
odr_res   = odr_obj.run()
	
## (6) Haal resultaten uit het resultaten-object:
# (6a) De beste schatters voor de parameters
par_best = odr_res.beta
# (6b) De (EXTERNE!) onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta
# (6c) De (INTERNE!) covariantiematrix
par_cov_ext = odr_res.cov_beta 
# (6d) De chi-kwadraat en de gereduceerde chi-kwadraat van deze fit
chi2 = odr_res.sum_square
chi2red = odr_res.res_var
# (6e) Een compacte weergave van de belangrijkste resultaten als output
odr_res.pprint()

# Hier plotten we ter controle de fit met de dataset (niet opgemaakt)
xplot=np.linspace(1,8, 8)
ax.plot(xplot, f([par_best[0]],xplot),'r-')



