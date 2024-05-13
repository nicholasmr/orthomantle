#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2022-2024

"""
Stokes flow solver
"""

import copy, sys, time, code # code.interact(local=locals())
import math
from dolfin import *

import socket
if socket.gethostname() == 'rathmann': from ufl_legacy import nabla_div, transpose # my library fix
else:                                  from ufl        import nabla_div, transpose

from mesh import ez, DOM_ID__BOTTOM, DOM_ID__TOP, DOM_ID__LEFT, DOM_ID__RIGHT, DOM_ID__INTERIOR

class Stokes():

    def __init__(self, meshargs, RA=1e2, friction_coef=0):

        mesh, boundary_parts, ds, norm = meshargs
        self.RA = RA
        self.friction_coef = friction_coef
        self.ds = ds
        
        ### FEM spaces
        
        self.Pele = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        self.Vele = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        self.Wele = MixedElement([self.Pele, self.Vele]) 
        self.P = FunctionSpace(mesh, self.Pele) 
        self.V = FunctionSpace(mesh, self.Vele) 
        self.W = FunctionSpace(mesh, self.Wele) 
        
        ### Weight functions and unknowns

        self.w       = TestFunction(self.W)
        self.w_trial = TrialFunction(self.W)
        
        self.wp, self.wv = split(self.w)
        self.u = Function(self.W) # unknowns
        
        ### Boundary conditions

        # Fix the pressure at the top boundary
        bc_p = DirichletBC(self.W.sub(0), Constant(0), "near(x[0],0) && near(x[1],0)", method="pointwise")
                            
        # Free-slip at all boundaries (v dot n = 0)
        bc_v_bot   = DirichletBC(self.W.sub(1).sub(1), Constant(0), boundary_parts, DOM_ID__BOTTOM)
        bc_v_top   = DirichletBC(self.W.sub(1).sub(1), Constant(0), boundary_parts, DOM_ID__TOP)
        bc_v_left  = DirichletBC(self.W.sub(1).sub(0), Constant(0), boundary_parts, DOM_ID__LEFT)
        bc_v_right = DirichletBC(self.W.sub(1).sub(0), Constant(0), boundary_parts, DOM_ID__RIGHT)

        # Collect BCs for mechanical part of the problem
        self.bcs = [bc_p, bc_v_bot, bc_v_top, bc_v_left, bc_v_right]
     
    def set_state(self, u0, interp=True):
        if interp: self.u.interpolate(u0)
        else:      self.u.assign(u0)
        self._set_split()     
        
    def set_rheology(self, rheology):
        self.rheology = rheology
       
    def _set_split(self):
        self.p, self.v = split(self.u)
        self.pc, self.vc = self.u.split() # used as coupling functions with other problems (thermal, CPO)
        
    def solve(self, temp0):

        F = nabla_div(self.v)*self.wp*dx + self.p*nabla_div(self.wv)*dx + Constant(self.RA)*temp0*inner(ez, self.wv)*dx

        if 0: 
            # Debug (Petra's variational form)
            YSTRESS, ETA_T, ETA0 = self.rheology.YSTRESS, self.rheology.ETA_T, self.rheology.ETA0
            u,v,p = self.u, self.v, self.p
            omega = self.wv
            #----
            def eijeij(v): return sqrt(inner(sym(nabla_grad(v)), sym(nabla_grad(v))))
            def viscosity(u, temp):
                p, v = u.split()
                viscosityl = exp(-temp*math.log(ETA_T))
                viscosityp = ETA0 + YSTRESS/eijeij(v)
                return 2/(1/viscosityl + 1/viscosityp)
            F += - viscosity(u, temp0)*inner((nabla_grad(v) + transpose(nabla_grad(v))), transpose(nabla_grad(omega)))*dx
            
        else:
            # My generalization
            tau = self.tau(temp0)
            F -= inner(tau, transpose(nabla_grad(self.wv)))*dx
            
            if self.friction_coef > 0:
                print('*** adding lid friction law: %.4e '%(self.friction_coef))
                F += -Constant(self.friction_coef)*dot(self.vc, self.wv)*self.ds(DOM_ID__TOP)

        solve(F == 0, self.u, self.bcs, solver_parameters={"newton_solver":  {"relative_tolerance": 1e-3, \
            "relaxation_parameter":1.0, "maximum_iterations":100, "convergence_criterion":'incremental', 'linear_solver': 'mumps'}}) # , 'preconditioner': 'ilu'

        self._set_split() # set new guess for next iteration

        return self.u

    def tau(self, temp0):
        eta = self.rheology.get_eta(self.vc, temp0)
        C   = self.rheology.get_C(self.v)
        tau = 2*eta*C
        return tau        

    def sigma(self, temp0):        
        tau = self.tau(temp0)
        sigma = tau - Identity(2)*self.p
        return sigma

