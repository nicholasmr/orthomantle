#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2022-2024

"""
Routines for saving and loading model state
"""

import copy, os, sys, code # code.interact(local=locals())
import numpy as np
from scipy import integrate
import pickle

from dolfin import *
from mesh import * 
from specfabpy.fenics.olivine import OlivineFabric
from thermal import * 
from stokes import * 
from rheology import *


def fname_h5(case, i, path):  return '%s/case-%i/statedumps/%04d.h5'%(path, case, i)
def fname_pkl(case, i, path): return '%s/case-%i/statedumps/%04d.pkl'%(path, case, i)
    
    
def save_state(case,i, timevec,metrics, kwargs_init, mesh,stokes,rheology,thermal, path='experiments'):
    
    os.system('mkdir -p %s/case-%i/statedumps'%(path,case))

    with open(fname_pkl(case,i,path), 'wb') as f:
        pickle.dump((timevec, metrics, kwargs_init), f)

    hdf5 = HDF5File(mesh.mpi_comm(), fname_h5(case,i,path), "w")
    hdf5.write(mesh, "mesh")
    hdf5.write(stokes.u, "u")
    hdf5.write(thermal.temp, "temp")
    hdf5.write(rheology.fabric.SDM_b.w, "sb")
    hdf5.write(rheology.fabric.SDM_n.w, "sn")
    hdf5.close()
        
        
def load_state(case,i, path='experiments', skipfields=False):

    with open(fname_pkl(case,i,path), 'rb') as f:
        (timevec, metrics, kwargs_init) = pickle.load(f)

    (kwargs_mesh, kwargs_fabric, kwargs_rheology, kwargs_stokes, kwargs_thermal) = kwargs_init

    meshargs = unit_mesh(**kwargs_mesh) 

    if not skipfields:
    
        fabric   = OlivineFabric(meshargs, **kwargs_fabric) # initial state is isotropic, so no need to initialize CPO state vector field
        rheology = Rheology(meshargs, **kwargs_rheology)
        stokes   = Stokes(meshargs, **kwargs_stokes)
        thermal  = Thermal(meshargs, **kwargs_thermal)

        rheology.set_fabric(fabric)
        stokes.set_rheology(rheology)

        ### Initialize with saved fields
    
        mesh = Mesh()
        hdf5 = HDF5File(mesh.mpi_comm(), fname_h5(case,i,path), "r")
        hdf5.read(mesh, '/mesh', False)
        
        u = Function(stokes.W)
        hdf5.read(u, "/u")
        stokes.set_state(u, interp=False)
        
        temp = Function(thermal.T)
        hdf5.read(temp, "/temp")
        thermal.set_state(temp)
        
        sn, sb = Function(rheology.fabric.SDM_n.W), Function(rheology.fabric.SDM_b.W)
        hdf5.read(sn, "/sn")
        hdf5.read(sb, "/sb")
        rheology.fabric.set_state(sb, sn, interp=False)
    else:
        stokes,rheology,thermal = None,None,None
    
    return (meshargs, timevec, metrics, stokes,rheology,thermal)

def integral_measures(stokes, thermal, ds, xlen=100):

    # Notice that there is no need to divide by domain width or area below (to determine averages) since we consider a unit mesh

    ex, ez = Constant((1,0)), Constant((0,1))

    ### Average velocity at top boundary
    vs = assemble(dot(stokes.vc,ex)*ds(DOM_ID__TOP))
    
    ### Average drag (xz shear stress)  at top boundary
    sigma = stokes.sigma(thermal.temp)
    tracvec = dot(sigma,ez) # traction vector
    ts = np.sqrt( assemble(inner(tracvec,tracvec)*ds(DOM_ID__TOP)) )
    sigxz = project(dot(tracvec,ex)) # for x-profile below
    sigzz = project(dot(tracvec,ez))    
    
    ### x-profiles for z=1
    N=xlen
    xvec = np.linspace(0,1,N)
    vs_x, tsx_x, tsz_x = np.zeros((N)),np.zeros((N)),np.zeros((N))
    for ii,x in enumerate(xvec):
        vs_x[ii] = stokes.vc.sub(0)(x, 1)
        tsx_x[ii] = sigxz(x, 1)
        tsz_x[ii] = sigzz(x, 1)
    
    return (vs,ts, xvec,vs_x,tsx_x,tsz_x)

def Ezz_avg(stokes):

    xi, Eij = stokes.rheology.get_Exixj()
    Exx, Ezz, Exz = Eij[0], Eij[2], Eij[4]

    N = 30 # should be a subsampling of mesh resolution to avoid jagged lines because Eij are DG0 elements
    xy0, xy1 = (0,0), (1,1)
    x_1d = np.linspace(xy0[0],xy1[0],N)
    y_1d = np.linspace(xy0[1],xy1[1],N)
    xv, yv = np.meshgrid(x_1d, y_1d)

    Ezz_np = np.zeros((N,N))
    for ii in np.arange(N):
        for jj in np.arange(N):
            Ezz_np[ii,jj] = Ezz(xv[ii,jj], yv[ii,jj])

    ezz = np.zeros((N))
    for ii in np.arange(N):
        ezz[ii] = integrate.trapezoid(Ezz_np[:,ii], y_1d)

    return (ezz, x_1d)    
    
