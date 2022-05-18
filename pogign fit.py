# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:20:44 2022

@author: saban
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import scipy.odr as odr


#%% Grafiek voor kopere draad 0.2 mm
folder = 'C:/Users/saban/OneDrive/Documents/GitHub/golf-en-optica-practicum-1/metingen eigen frequenties koper 0.2 - Sheet1.csv'

#we gebruiken een csv omdat een xlsx niet moeilijker met numpy importeert.
filename = 'proefmetingen - Sheet1.csv'
proefdata = np.genfromtxt(filename, delimiter=',', skip_header=1)

ef1 = proefdata[:,0]
ef2 = proefdata[:,1]
ef = (ef1+ef2)/2
n= np.linspace(1,8, 8)
f_err = 1 # onzekerheid is 1 Hz

plt.figure()
plt.errorbar(n, ef1, xerr=0.0, yerr=f_err, fmt='r.', label= 'meting 1')
plt.errorbar(n, ef2, xerr=0.0, yerr=f_err, fmt='b.', label= 'meting 2')

# opmaak
plt.title('gevonden frequenties voor staande golven van kopere draad 0.2 mm')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
plt.show()
#%% data fit


n = n.reshape(-1,1)
f,ax = plt.subplots(1)
ax.errorbar(n,ef,xerr=0,yerr= f_err,fmt='k.')

#ax.show()
A_start=0.5
B_start=1.5

def f(ef, n):
    f = ef/n
    return f

## (2) Definieer het model-object om te gebruiken in odr
odr_model = odr.Model(f)

## (3) Definieer een RealData object
## Een RealData-object vraagt om de onzekerheden in beide richtingen. 
## !! De onzekerheid in de x-richting mag ook nul zijn (dan mag je sx=0 weglaten), 
## maar dan moet bij onderdeel (4)/(4a) wel gekozen worden voor een
## kleinste-kwadratenaanpassing !!
odr_data  = odr.RealData(n,ef,sx= 0,sy=f_err)

## (4) Maak een ODR object met data, model en startwaarden
## Je geeft startwaarden voor parameters mee bij keyword beta0
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[A_start,B_start])
## (4a) Stel in op kleinste kwadraten (optioneel)
## Als de onzekerheden in de x-richting gelijk zijn aan nul, dan faalt de 
## default-aanpak van ODR. Je moet dan als methode een kleinste-kwadraten- 
## aanpassing instellen. Dit gaat met het volgende commando (nu uit):
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
par_cov = odr_res.cov_beta 
print(" De (INTERNE!) covariantiematrix  = \n", par_cov)

# (6d) De chi-kwadraat en de gereduceerde chi-kwadraat van deze aanpassing
chi2 = odr_res.sum_square
print("\n Chi-squared         = ", chi2)
chi2red = odr_res.res_var
print(" Reduced chi-squared = ", chi2red, "\n")

# (6e) Een compacte weergave van de belangrijkste resultaten als output
odr_res.pprint()

# Hier plotten we ter controle de aanpassing met de dataset (niet opgemaakt)
xplot=np.arange(-1,13,1)
ax.plot(xplot,par_best[0] + par_best[1]*xplot,'r-')