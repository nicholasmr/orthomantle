#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2022-2024

"""
Simulates thermal convection in a rectangular cell.
Problem setup is described in Tosi et al. (2015).
This code is adapted from that kindly provided by Petra Maierová <maipe@seznam.cz>.

Uses:
    - Non-dimensional formulation
    - Taylor-Hood elements
    - Implicit Euler time-stepping with Courant criterion
    - Temperature and strain-rate dependent effective viscosity

Tested with python version 3.10.6 and dolphin version 2019.2.0.dev0
"""

import copy, sys, time, code # code.interact(local=locals())
import numpy as np
from dolfin import Constant, Expression 

from specfabpy.fenics.olivine import OlivineFabric
from thermal import * 
from stokes import * 
from rheology import *
from mesh import * 
from state import *

#----------------
# Debug flags 
#----------------

DEBUG = 0 # (low resolution etc.)

ENABLE_FABRIC_EVOLUTION = True 

#----------------
# Experiment (case) selection
#----------------

# Model case to run (see definitions below)
if len(sys.argv)==2:
    cases = [int(sys.argv[1]),]    
else:
    cases = [11,12]

#----------------
# Constants and parameters by P. Maierová
#----------------

### Model geometry

#XDIV, ZDIV = [30]*2 if not DEBUG else [18]*2  # grid resolution
XDIV, ZDIV = [30]*2 if not DEBUG else [18]*2  # grid resolution

### Boundary conditions

TS = 0 # surface temperature
TB = 1 # bottom boundary temperature (melting point)

### Rheological parameters

RA      = 1e2 # Rayleigh number
YSTRESS = 1.0 # Yield stress
ETA_T   = 1e5
ETA0    = 1e-3

### Stopping criteria

TMAX = 3e-1
IMAX = 12*1000

### Numerics

C_CFL = 0.5          # Courant number
dt = Constant(1e-10) # initial value of time step

#----------------
# For each case...
#----------------

for case in cases:

    print('**** Running experiment %i ****'%(case))

    #----------------
    # Additional parameters
    #----------------

    L = 8 # spectral resolution
    nu_realspace = 5e-1 # regularization magnitude (for real-space stabilization)
    nu_S2_mult = 1.05 # multiply magnitude of orientation-space regularization by this factor
    Enb = 100 # n--b slip system enhancement

    if case == 1 or case == 11:
        rheo = 'Isotropic-Plastic'
        n_bulk = np.inf # n -> inf is the plastic limit
        A0 = 1
        
    if case == 2 or case == 12:
        rheo = 'Isotropic-Viscoplastic'
        n_bulk = 3.5
        A0 = 100 # calibrate fluidity to match plastic (n=inf) behavior approximately  

    if case == 3 or case == 13:
        rheo = 'Orthotropic-Viscoplastic'
        n_bulk = 3.5
        A0 = 100 # use calibration from "Isotropic-Viscoplastic" experiment

    kwargs_mesh     = dict(XDIV=XDIV, ZDIV=ZDIV)
    kwargs_fabric   = dict(L=L, nu_realspace=nu_realspace, nu_multiplier=nu_S2_mult, Cij=OlivineFabric.cij_Abramson1997, rho=OlivineFabric.rho_olivine)    
    kwargs_rheology = dict(rheology=rheo, n=n_bulk, A0=A0, Enb=Enb, YSTRESS=YSTRESS, ETA_T=ETA_T, ETA0=ETA0)
    kwargs_stokes   = dict(RA=RA, friction_coef=2.0/100 if case > 10 else 0) # lid friction law: sigxz = fric_coef * u
    kwargs_thermal  = dict(TS=TS, TB=TB)

    #----------------
    # Initialize components
    #----------------

    meshargs = unit_mesh(**kwargs_mesh) 
    mesh, boundary_parts, ds, norm = meshargs # unpack
    fabric   = OlivineFabric(meshargs, **kwargs_fabric) # initial state is isotropic, so no need to initialize CPO state vector field
    rheology = Rheology(meshargs, **kwargs_rheology)
    stokes   = Stokes(meshargs, **kwargs_stokes)
    thermal  = Thermal(meshargs, **kwargs_thermal)

    rheology.set_fabric(fabric)
    stokes.set_rheology(rheology)
    
    ### Initial fields

    u0 = Constant((0,0,0)) # u=(p,v)=(p,vx,vy)
    stokes.set_state(u0, interp=True)
    
    temp0 = Expression("(1 - x[1]) + 0.01*cos(pi*x[0]/width)*sin(pi*x[1]/height)", width=1, height=1, degree=2) 
    thermal.set_state(temp0, interp=True)

    #----------------
    # Time evolution
    #----------------

    ### Time keeping

    t = 0.0
    i = 0

    ### Integrated diagnostic measures

    N = IMAX+1
    timevec = np.zeros(N)
    timevec[:] = np.nan
    vs, ts = np.zeros(N), np.zeros(N)

    ### Simulate!

    while (t < TMAX) and (i < IMAX):

        i += 1
        t += float(dt) # advance time
        timevec[i] = t
        info("*** Step = %i :: dt=%.3e, t=%.3e" % (i, dt, t))

        info("*** Solving thermal problem")
        thermal.solve(stokes.vc, dt)
        
        info("*** Solving momentum balance")
        stokes.solve(thermal.temp)
        
        if ENABLE_FABRIC_EVOLUTION:
            info("*** Solving fabric evolution")
            stokes.rheology.fabric.evolve(stokes.v, dt)
            stokes.rheology.update_Eij()
        else:
            info("*** Skipping fabric evolution")

        # Compute new value of time step using CFL criterion
        h_min = mesh.hmin()
        v_max = abs(stokes.vc.vector()[:]).max()
        dt.assign(C_CFL*h_min/v_max)

        # Save model state and metrics

        vs[i],ts[i], *xprofiles = integral_measures(stokes, thermal, ds)
        
        if i%5 == 0: 
            print('*** Dumping model state...')
            kwargs_init = (kwargs_mesh, kwargs_fabric, kwargs_rheology, kwargs_stokes, kwargs_thermal)
            metrics = (vs[:i], ts[:i], *xprofiles)
            save_state(case,i, timevec[:i],metrics, kwargs_init, mesh,stokes,rheology,thermal)
            
        #list_timings()
    
