#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2022-2024

from dolfin import *

DOM_ID__BOTTOM   = 0
DOM_ID__TOP      = 1
DOM_ID__LEFT     = 2
DOM_ID__RIGHT    = 3
DOM_ID__INTERIOR = 4

ez = Constant((0,1))
I_t = Constant(((1,0), (0,1)))

def unit_mesh(XDIV=40, ZDIV=40, W=1, H=1, x0=0, z0=0):

    #----------------
    # Boundaries
    #----------------

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[1], z0)

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[1], z0+H)

    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[0], x0)

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[0], x0+W)

    #----------------
    # Mesh
    #----------------

    mesh = RectangleMesh(Point(x0, z0), Point(x0+W, z0+H), XDIV, ZDIV) # XDIV, ZDIV, "crossed") # option "crossed" stands for crossed diagonals (number of elements=XDIV*ZDIV*4)
    cell = triangle
    norm = FacetNormal(mesh) # definition of an outer normal

    #----------------
    # Boundary partitioning
    #----------------

    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

    boundary_parts.set_all(DOM_ID__INTERIOR) # interior of the domain

    Gamma_b = BottomBoundary()
    Gamma_b.mark(boundary_parts, DOM_ID__BOTTOM)

    Gamma_t = TopBoundary()
    Gamma_t.mark(boundary_parts, DOM_ID__TOP)

    Gamma_l = LeftBoundary()
    Gamma_l.mark(boundary_parts, DOM_ID__LEFT)

    Gamma_r = RightBoundary()
    Gamma_r.mark(boundary_parts, DOM_ID__RIGHT)

    ds = Measure("ds")[boundary_parts] 
    
    return (mesh, boundary_parts, ds, norm)

