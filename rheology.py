#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2022-2024

"""
Viscoplastic rheology for polycrystalline olivine
"""

import numpy as np
import math
import copy, sys, time, code # code.interact(local=locals())

from dolfin import *

from specfabpy.fenics.rheology import Isotropic, IsotropicPlastic, Orthotropic

class Rheology():

    def __init__(self, meshargs, rheology='Orthotropic', n=3.5, A0=1, Enb=100, YSTRESS=1.0, ETA_T=1e5, ETA0=1e-3):

        mesh, boundary_parts, ds, norm = meshargs
        self.mesh = mesh

        ### Bulk rheology

        self.rheology_name = rheology
        self.n  = n
        self.A0 = A0
        self.modelplane = 'xz'
        
        self.YSTRESS = YSTRESS
        self.ETA_T   = ETA_T
        self.ETA0    = ETA0
        
        ### Bulk rheology 
        
        if self.rheology_name not in ['Isotropic-Plastic', 'Isotropic-Viscoplastic', 'Orthotropic-Viscoplastic']: 
            raise ValueError('Rheology "%s" not available.'%(self.rheology))
        else:
            if self.rheology_name == 'Isotropic-Plastic':        self.rheology = IsotropicPlastic(n=self.n)
            if self.rheology_name == 'Isotropic-Viscoplastic':   self.rheology = Isotropic(n=self.n)
            if self.rheology_name == 'Orthotropic-Viscoplastic': self.rheology = Orthotropic(n=self.n)           
        

    def set_fabric(self, fabric):
        self.fabric = fabric
        self.L = self.fabric.L # spectral truncation        

    def get_C(self, v):
        # Tensorial part of viscoplastic rheology
        D2 = sym(nabla_grad(v)) # strain-rate tensor (2x2)
        C2 = self.rheology.C_inv(D2, self.fabric.mi, self.fabric.Eij, modelplane=self.modelplane)
        return C2        

    def get_eta(self, v, temp):
        # Scalar viscosity of viscoplastic rheology
        D2 = sym(nabla_grad(v)) # strain-rate tensor (2x2)
        eta_vp = self.rheology.viscosity(D2, self.A0, self.fabric.mi, self.fabric.Eij, modelplane=self.modelplane)
        eta = self.get_eta_harmonicavg(eta_vp, temp)
        return eta
    
    def get_eta_harmonicavg(self, eta0, temp):
        etal = self.A(temp) # linear part
#        etap = ETA0 + YSTRESS/eta0 # plastic part (original version)
        etap = self.ETA0 + self.YSTRESS * eta0 # not YSTRESS/eta0 as in Petra's code since we define viscosity as the entire multiplicative factor of YSTRESS
        eta = 2/(1/etal + 1/etap)
        return eta
    
    def A(self, temp):
        return exp(-temp*math.log(self.ETA_T))     
            
