"""
script voor golf en optica.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import pandas as pd


folder = 'C:/Users/saban/OneDrive/Documents/GitHub/golf-en-optica-practicum-1/'

#we gebruiken een csv omdat een xlsx niet moeilijker met numpy importeert.
filename = 'proefmetingen - Sheet1.csv'
proefdata = np.genfromtxt(filename, delimiter=',', skip_header=1)

ef1 = proefdata[:,0]
ef2 = proefdata[:,1]
t = np.linspace(1,8, 8)
f_err = 1 # onzekerheid is 1 Hz

plt.figure()
plt.errorbar(t, ef1, xerr=0.0, yerr=f_err, fmt='r.', label= 'meting 1')
plt.errorbar(t, ef2, xerr=0.0, yerr=f_err, fmt='b.', label= 'meting 2')

# opmaak
plt.title('gevonden frequenties voor staande golven')
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('frequentie in Hz')
plt.xlabel('n-de eigenfrequentie')
plt.show()
