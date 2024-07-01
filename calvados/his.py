# import matplotlib_inline.backend_inline
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
from openmm import app, unit
import openmm
import mdtraj as md
from main import BlockAnalysis
from localcider.sequenceParameters import SequenceParameters
from numpy import linalg
from scipy.optimize import curve_fit
import scipy.stats as scs
import pandas as pd
import itertools
import numpy as np
import os
import shutil
import subprocess

print('Installing libraries...')

# matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')

NAME = "ab40"

SEQUENCE = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV"
if " " in SEQUENCE:
    SEQUENCE = ''.join(SEQUENCE.split())
    print('Blank character(s) found in the provided sequence. Sequence has been corrected, but check for integrity:')
    print(SEQUENCE)
    print('\n')

Temperature = 278
Ionic_strength = 0.1397


charged_N_terminal_amine = True
charged_C_terminal_carboxyl = True
charged_histidine = False
Nc = 1 if charged_N_terminal_amine == True else 0
Cc = 1 if charged_C_terminal_carboxyl == True else 0

pH = 4
pKa = 6.4

if charged_histidine == True:
    print('Define pH and pKa to set the charge of Histidines according to the Henderson-Hasselbalch equation.')
    pH = float(pH)
    pKa = float(pKa)
    Hc = 1/(1+10**(pH-pKa))
if charged_histidine == False:
    Hc = 0

Hc6 = 0.5
Hc13 = 0.5
Hc14 = 0.5

residues = pd.read_csv('residues.csv')
residues = residues.set_index('one')

# Simulation_time = "AUTO"
Simulation_time = 1

N_res = len(SEQUENCE)
N_save = 700 if N_res < 150 else int(np.ceil(3e-4*N_res**2)*1000)


def genParamsDH(df, seq, temp, ionic, Nc, Cc, Hc):
    kT = 8.3145*temp*1e-3
    fasta = seq.copy()
    r = df.copy()

    # Set the charge on HIS based on the pH of the protein solution
    # r.loc['H', 'q'] = Hc
    if Nc == 1:
        r.loc['X'] = r.loc[fasta[0]]
        r.loc['X', 'q'] = r.loc[seq[0], 'q'] + 1.
        fasta[0] = 'X'
    if Cc == 1:
        r.loc['Z'] = r.loc[fasta[-1]]
        r.loc['Z', 'q'] = r.loc[seq[-1], 'q'] - 1.
        fasta[-1] = 'Z'
    # Calculate the prefactor for the Yukawa potential

    def fepsw(T): return 5321/T+233.76-0.9297 * \
        T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT

    # yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    yukawa_eps = []
    for i in range(len(fasta)):
        a = fasta[i]
        if i == 5:
            print(fasta[i])
            yukawa_eps.append(Hc6 * np.sqrt(lB*kT))
        elif i == 12:
            print(fasta[i])
            yukawa_eps.append(Hc13 * np.sqrt(lB*kT))
        elif i == 13:
            print(fasta[i])
            yukawa_eps.append(Hc14 * np.sqrt(lB*kT))
        else:
            yukawa_eps.append(r.loc[a].q*np.sqrt(lB*kT))

    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    return yukawa_eps, yukawa_kappa


yukawa_eps, yukawa_kappa = genParamsDH(
    residues, list(SEQUENCE), Temperature, Ionic_strength, Nc, Cc, Hc)
