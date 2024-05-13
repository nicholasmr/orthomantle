#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2022-2024

from dolfin import *
from mesh import DOM_ID__BOTTOM, DOM_ID__TOP, DOM_ID__LEFT, DOM_ID__RIGHT, DOM_ID__INTERIOR

class Thermal():
    
    def __init__(self, meshargs, TS=0, TB=1):
 
        mesh, boundary_parts, ds, norm = meshargs
        self.TS, self.TB = TS, TB
 
        ### FEM spaces   
        self.T = FunctionSpace(mesh, "Lagrange", 2) # quadratic Lagrange elements for temperature
        
        ### Weight function and unknown
        self.w          = TestFunction(self.T)
        self.temp_trial = TrialFunction(self.T)
        self.temp       = Function(self.T) # solution container
        self.temp0      = Function(self.T) # previous solution container
                
        ### Boundary conditions
        
        # Fixed temperature at top and bottom boundaries; natural BCs on sidewalls
        bc_top = DirichletBC(self.T, Constant(self.TS), boundary_parts, DOM_ID__TOP)
        bc_bot = DirichletBC(self.T, Constant(self.TB), boundary_parts, DOM_ID__BOTTOM)
        self.bcs = [bc_bot, bc_top] # Collect BCs

    def set_state(self, temp, interp=True):
    
        if interp: # assume temp is an expression or constant
            self.temp.interpolate(temp)
            self.temp0.interpolate(temp)
        else:
            self.temp.assign(temp)
            self.temp0.assign(temp)
        
    def solve(self, v0, dt):
    
        a = self.temp_trial*self.w*dx + dt*inner(v0, nabla_grad(self.temp_trial))*self.w*dx + dt*inner(nabla_grad(self.temp_trial), nabla_grad(self.w))*dx
        L = self.temp0*self.w*dx
        
        problem_T = LinearVariationalProblem(a, L, self.temp, self.bcs)
        solver_T = LinearVariationalSolver(problem_T)
        prm_T = solver_T.parameters
        prm_T["linear_solver"] = "mumps"
        solver_T.solve()
        
        self.temp0.assign(self.temp) # save previous solution
    
