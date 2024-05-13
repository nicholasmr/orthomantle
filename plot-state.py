#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2023-2024

"""
Model state plot
"""

import copy, os, sys, code # code.interact(local=locals())
import numpy as np
import pickle

from dolfin import *
from state import *
from plottools import *

def load_and_plot(case, i, annotate=False, fname=None):
    (meshargs, timevec,metrics, stokes,rheology,thermal) = load_state(case, i, path=EXPPATH)
    rheology.fabric.update_Eij()
    (mesh, boundary_parts, ds, norm) = meshargs # unpack
    t = timevec[-1]
    print('time %.3e'%(t))
    plot_diagnostics(stokes.vc, thermal.temp, rheology, mesh, t, case,i, path=EXPPATH, fname=fname, annotate=annotate)

### Frames for animation    
if 0:
    dn = 50
    for nt in np.arange(dn, 2200+ 1*dn, 1*dn): load_and_plot(3, nt, annotate=True)
    load_and_plot(3, 5, annotate=True)

### Still frames for paper
if 1:
    nt = 5000 
#    load_and_plot(1, nt)
#    load_and_plot(2, nt)
    load_and_plot(3, nt, annotate=True, fname='case-3-steadystate')

