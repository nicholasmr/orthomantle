#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2022-2024

""" 
Plotting library for model state 
"""

import copy, os, sys, code # code.interact(local=locals())

import numpy as np
import scipy.special as sp
import cartopy.crs as ccrs
import cmasher as cmr
from cmcrameri import cm as cracmap

import warnings
warnings.filterwarnings("ignore")        

from dolfin import *
from scipy.interpolate import griddata

from specfabpy import specfab as sf
from specfabpy import common as sfcom
from specfabpy import discrete as sfdsc
from specfabpy import plotting as sfplt
FS = sfplt.setfont_tex(fontsize=13)
FSLEG = FS-2
FSLBL = FS+2

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
from matplotlib import rcParams, rc
import matplotlib.gridspec as gridspec
from matplotlib import collections as mc

#------------------
# Global 
#------------------

EXPPATH = 'experiments'

lbl_list0  = ['Plastic', 'Isotropic viscoplastic', 'Orthotropic viscoplastic']
lbl_list   = 2*lbl_list0
lbl_list_f = ['%s ($f=0$)'%(l) for l in lbl_list0] + ['%s ($f=2/100$)'%(l) for l in lbl_list0]

ls_list = ['-']*3 + ['--']*3
c_list = ['k', sfplt.c_blue, sfplt.c_red, '0.5', sfplt.c_lblue, sfplt.c_lred]

def get_linestyle(case, flbl=False, lw=1.5):
    # for experiments 1,2,3, 11,12,13
    jj = case - 1 + (0 if case<10 else 3-10)
    c, ls, lbl = c_list[jj], ls_list[jj], lbl_list_f[jj] if flbl else lbl_list[jj]
    return dict(lw=lw, ls=ls, c=c, label=lbl)

#------------------
# Local
#------------------

c_contour = '0.1'

cb_aspect = 20
kwargs_cb = {'pad':0.145, 'aspect':cb_aspect, 'fraction':0.075, 'orientation':'horizontal'}

#kwargs_grid = {'lw':0.075, 'color':'0.5', 'alpha':0.4}
kwargs_grid = {'lw':0.075, 'color':'0.5', 'alpha':0.8}

kwargs_panellbl = dict(fontsize=FSLBL, frameon=False, bbox=(-0.23,1.21))
kwargs_legend = {'frameon':False, 'fancybox':False, 'edgecolor':'none', 'handlelength':1.3, 'labelspacing':0.25, 'columnspacing':1.25}


def plot_diagnostics(u, T, rheology, mesh, t, CASE,i, modelplane='xz', xy0=(0,0), xy1=(1,1), path=EXPPATH, fname=None, annotate=False):

    IS_ANISOTROPIC = True # debug flag: plot CPO fields etc. ?

    if fname is None: fname = '%s/case-%i/diagnostic-%04d.png'%(path,CASE,i)
    else:             fname = '%s/%s.png'%(path,fname)
    print('[->] Plotting model state: %s'%(fname))

    #-------------------------
    # Initialize
    #-------------------------
    
    ### Figure geometry
    figscale = 0.45
    figsize = (2.5/3 * 20*figscale, 3/2 * 1.925 *9*figscale)
    fig = plt.figure(figsize=figsize)

    ### Subplot axes
    gs = gridspec.GridSpec(3, 2, wspace=0.25, hspace=0.29, left=0.07, right=0.82, top=0.968, bottom=0.045)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])

    ### Mesh coordinates and triangulation
    coords = copy.deepcopy(mesh.coordinates().reshape((-1, 2)).T)
    triang = tri.Triangulation(*coords, triangles=mesh.cells())
    
    ### Function space for plotting
    Qele = FiniteElement("CG", mesh.ufl_cell(), 2) 
    Q = FunctionSpace(mesh, Qele)
    ex,ey = Constant((1,0)), Constant((0,1))
    
    ### S2 projection
    geo, prj = sfplt.getprojection(rotation=-90-40, inclination=50)
    
    #-------------------------
    # Generate plots
    #-------------------------
    
#    N = 16 * 1
    N = 31
    x_1d = np.linspace(xy0[0],xy1[0],N)
    y_1d = np.linspace(xy0[1],xy1[1],N)
    xv, yv = np.meshgrid(x_1d, y_1d)

    N_lr = 8
    dxy = (xy1[0]-xy0[0])/(N_lr+2)
    x_1d_lr = np.linspace(xy0[0]+dxy,xy1[0]-dxy,N_lr)
    y_1d_lr = np.linspace(xy0[1]+dxy,xy1[1]-dxy,N_lr)
    xv_lr, yv_lr = np.meshgrid(x_1d_lr, y_1d_lr)
        
    """
    Velocity and temperature
    """

    ax = ax1
    F = project(T, Q)
    lbl='$T$'
    ticks = np.linspace(0,1, 11)
    cmap = cracmap.lipari
    plot_panel(ax, F, mesh, triang, ticks, cmap, lbl, extend='neither', dtickcbar=1, xy0=xy0, xy1=xy1)

    uvec = np.zeros((N_lr,N_lr,2))
    for ii in np.arange(N_lr):
        for jj in np.arange(N_lr):
            x_, y_ = xv_lr[ii,jj], yv_lr[ii,jj]
            uvec[ii,jj,0] = u.sub(0)(x_,y_)
            uvec[ii,jj,1] = u.sub(1)(x_,y_)
            uvec[ii,jj,:] /= np.linalg.norm(uvec[ii,jj,:])
    QV = ax.quiver(xv_lr,yv_lr, uvec[:,:,0], uvec[:,:,1], color=c_contour)
    plt.quiverkey(QV, 0.775, 1.075, 1.5, r'$\vb{u}/\abs{\vb{u}}$', labelpos='E', coordinates='axes', labelsep=0.05)

    sfplt.panellabel(ax, 2, r'\textit{(a)}', **kwargs_panellbl)
    ax.text(0.175, 1.06, r'$t=%.4f$'%(t), fontsize=FS)
    
    """
    CPO field
    """
    
    ax = ax2

    cpo_n,cpo_b = rheology.fabric.SDM_n, rheology.fabric.SDM_b
   
    # Principal directions
    m1_n = np.zeros((N_lr,N_lr,2))
    m1_b = np.zeros((N_lr,N_lr,2))
    for ii in np.arange(N_lr):
        for jj in np.arange(N_lr):
            x_, y_ = xv_lr[ii,jj], yv_lr[ii,jj]
            eigvecs, eigvals = cpo_n.eigenframe(x_, y_, modelplane=modelplane)
            m1_n[ii,jj,:] = eigvecs[[0,2],0] # x-z components of largest nlm eigenvalue
            eigvecs, eigvals = cpo_b.eigenframe(x_, y_, modelplane=modelplane)
            m1_b[ii,jj,:] = eigvecs[[0,2],0] 

    # CPO strength (intensity)
    J_n  = np.zeros((N,N))
    for ii in np.arange(N):
        for jj in np.arange(N):
            x_, y_ = xv[ii,jj], yv[ii,jj]
            Iend = sf.L8len
            nlm = cpo_n.get_nlm(x_,y_)
            J_n[ii,jj] = np.sqrt(4*np.pi)**2*np.sum(np.multiply(nlm[:Iend], np.conj(nlm[:Iend])))

    lvls = np.arange(1,12 +1,1)
    dtickcbar = 1
    #cmap = cmr.get_sub_cmap('Greys', 0, 0.9)
    cmap = cracmap.batlowW_r
    h = ax.contourf(xv,yv, J_n, cmap=cmap, levels=lvls, extend='both')
    cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
    cbar.ax.set_xlabel(r'$J(n)$')
    if dtickcbar>1: 
        for label in cbar.ax.xaxis.get_ticklabels()[1::dtickcbar]: label.set_visible(False)
    set_axes(ax, xy0=xy0, xy1=xy1)
    ax.triplot(triang, **kwargs_grid)
    
    sc = 21
    QV1 = ax.quiver(xv_lr,yv_lr, +m1_n[:,:,0], +m1_n[:,:,1], scale=sc, color=sfplt.c_blue)
    QV1 = ax.quiver(xv_lr,yv_lr, -m1_n[:,:,0], -m1_n[:,:,1], scale=sc, color=sfplt.c_blue)
    QV2 = ax.quiver(xv_lr,yv_lr, +m1_b[:,:,0], +m1_b[:,:,1], scale=sc, color=sfplt.c_red)
    QV2 = ax.quiver(xv_lr,yv_lr, -m1_b[:,:,0], -m1_b[:,:,1], scale=sc, color=sfplt.c_red)

    dx = -0.05
    plt.quiverkey(QV1, 0.58+dx, 1.075, 1.5, r'$\pm\vb{m}_1$', labelpos='E', coordinates='axes', labelsep=0.05)
    plt.quiverkey(QV2, 0.85+dx, 1.075, 1.5, r'$\pm\vb{m}_2$', labelpos='E', coordinates='axes', labelsep=0.05)

    # ODFs
    
    pos = [(0.5,0.5), (0.5,np.amax(yv_lr))]
    axpos = [(0.91, 0.855), (0.91, 0.966)]
    mrk = ['X', 'P']
    CPO_plots = [{ 'nlm':cpo_n.get_nlm(x,y), 'blm': cpo_b.get_nlm(x,y), 'loc':(x,y), 'axpos':axpos[ii], 'mrk':mrk[ii] } for ii,(x,y) in enumerate(pos)]
    mrk_kwargs = dict(markersize=10, markeredgewidth=1.1, markeredgecolor='k', markerfacecolor='w')

    for CPO in CPO_plots:
    
        W = H = 0.075 # ax width
        x0, y0 = CPO['loc'][0], CPO['loc'][1]
        points, = ax.plot([x0,],[y0,], CPO['mrk'], **mrk_kwargs)
        trans = CPO['axpos']
        dy = -0.75*H
        dxax = 0.6*W
        axpos1 = [trans[0]-W/2-dxax, trans[1]-H/2+dy, W,H]
        axpos2 = [trans[0]-W/2+dxax, trans[1]-H/2+dy, W,H]

        axin1 = plt.axes(axpos1, projection=prj) 
        axin1.set_global()
        axin2 = plt.axes(axpos2, projection=prj) 
        axin2.set_global()
        
        sfplt.plotODF(CPO['nlm'], cpo_n.lm, axin1, cmap='Blues', lvlset=(np.linspace(0.1, 0.6, 6), lambda x,p:'%.1f'%x), showcb=False)
        sfplt.plotODF(CPO['blm'], cpo_b.lm, axin2, cmap='Reds',  lvlset=(np.linspace(0.1, 0.6, 6), lambda x,p:'%.1f'%x), showcb=False)
        sfplt.plotcoordaxes(axin1, geo, axislabels='vuxi', color='k', fontsize=FSLEG)
        sfplt.plotcoordaxes(axin2, geo, axislabels='vuxi', color='k', fontsize=FSLEG)

        axin1.set_title(r'$n$', fontsize=FS)
        axin2.set_title(r'$b$', fontsize=FS)
        kwargs = dict(fontsize=FS, va='top', ha='center')
        axin1.text(0.5, -H*1.02, '[010]', transform=axin1.transAxes, **kwargs)
        axin2.text(0.5, -H*1.02, '[100]', transform=axin2.transAxes, **kwargs)


    x0 = 1.1
    y1, y2 = 0.948, 0.39
    ax.text(x0, y1, 'CPO at', fontsize=FS)
    ax.text(x0, y2, 'CPO at', fontsize=FS)
    ax.plot([x0+0.26,],[y1+0.02,], mrk[1], clip_on=False, **mrk_kwargs)
    ax.plot([x0+0.26,],[y2+0.02,], mrk[0], clip_on=False, **mrk_kwargs)

    sfplt.panellabel(ax, 2, r'\textit{(b)}', **kwargs_panellbl)


    """
    Enhancement factors
    """

#    mi, Eij = rheology.mi, rheology.Eij # eigenenhancements
    xi, Eij = rheology.fabric.xi, rheology.fabric.Exij #rheology.get_Exixj()
    Exx, Ezz, Exz = Eij[0], Eij[2], Eij[4]
    
    ### Exz

    ax = ax4
    F = project(Exz, Q) # nb (xz) shear
    lbl='$E_{xz}$'
    ticks = np.arange(1,4.5+1e-3,0.5) #np.linspace(1,2.5, 7)
    cmap = cracmap.oslo_r
    plot_panel(ax, F, mesh, triang, ticks, cmap, lbl, extend='both', dtickcbar=2, xy0=xy0, xy1=xy1)
    sfplt.panellabel(ax, 2, r'\textit{(d)}', **kwargs_panellbl)
    if annotate: ax.text(0.5,1.01, 'Ease of $x$--$z$ shear\n relative to isotropic CPO', va='bottom', ha='center', ma='center', fontsize=FS-1)

    ### Ezz
    
    ax = ax3
    F = project(Ezz, Q) 
    lbl='$E_{zz}$'
    ticks = np.arange(0.2,1.8+1e-3,0.2)
#    cmap = cmr.get_sub_cmap('BrBG', 0, 1) # 
    cmap = cracmap.vik_r
    plot_panel(ax, F, mesh, triang, ticks, cmap, lbl, extend='both', dtickcbar=2, xy0=xy0, xy1=xy1)
    sfplt.panellabel(ax, 2, r'\textit{(c)}', **kwargs_panellbl)
    if annotate: ax.text(0.5,1.01, 'Ease of uniaxial $z$-compression\n relative to isotropic CPO', va='bottom', ha='center', ma='center', fontsize=FS-1)
    
    """
    Seismic velocities
    """
    
    theta, phi = 0, 0 # wave propagation direction (colat and azimuth)
    vP, vS1, vS2 = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
    for ii in np.arange(N):
        for jj in np.arange(N):
            vP[ii,jj], vS1[ii,jj], vS2[ii,jj] = rheology.fabric.get_elastic_velocities(xv[ii,jj], yv[ii,jj], theta, phi)

    vPiso, vSiso, _ = rheology.fabric.get_elastic_velocities__isotropic()
    
    ### P-wave

    ax = ax5
    F = vP/vPiso * 1e2
    delta = 12
    lvls = np.linspace(100-delta,100+delta,13)
    dtickcbar = 1
    h = ax.contourf(xv,yv, F, cmap=cracmap.bam_r, levels=lvls, extend='both')
    cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
    cbar.ax.set_xlabel(r'$V_\mathrm{P}/V_{\mathrm{P},0}$ (\%)')
    if dtickcbar>1: 
        for label in cbar.ax.xaxis.get_ticklabels()[1::dtickcbar]: label.set_visible(False)
    if annotate: ax.text(0.5,1.01, 'Vertical $V_\mathrm{P}$\n relative to $V_\mathrm{P}$ of isotropic CPO', va='bottom', ha='center', ma='center', fontsize=FS-1)
    set_axes(ax, xy0=xy0)
    sfplt.panellabel(ax, 2, r'\textit{(e)}', **kwargs_panellbl)
    
    ### S-wave
    
    ax = ax6
    lvls = np.linspace(0,11,12)
    F = (vS1-vS2)/vSiso * 1e2
    h = ax.contourf(xv,yv, F, cmap=cracmap.acton_r, levels=lvls, extend='both')
    cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
    cbar.ax.set_xlabel(r'$\Delta V_\mathrm{S}/V_{\mathrm{S},0}$ (\%)')
    if annotate: ax.text(0.5,1.01, 'Vertical $\Delta V_\mathrm{S}$\n relative to $V_\mathrm{S}$ of isotropic CPO', va='bottom', ha='center', ma='center', fontsize=FS-1)
    set_axes(ax, xy0=xy0)
    sfplt.panellabel(ax, 2, r'\textit{(f)}', **kwargs_panellbl)
        
    #-------------------
    # SAVE PLOT
    #-------------------
    
    plt.savefig(fname, dpi=250)
    print('[OK] Done')
    plt.close(fig)
        

def plot_panel(ax, z, mesh, triang, ticks, cmap, title, dtickcbar=1, contourlvls=None, extend='max', fmt='%1.1f', cbaspect=cb_aspect, xy0=(0,0), xy1=(+1,+1)):

    Z = z.compute_vertex_values(mesh)
    h = ax.tricontourf(triang, Z, levels=ticks, cmap=cmap, extend=extend, norm=None)
    cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
    cbar.ax.set_xlabel(title)
    set_axes(ax, xy0=xy0)
    if dtickcbar>1:
        for label in cbar.ax.xaxis.get_ticklabels()[1::dtickcbar]: label.set_visible(False)


def set_axes(ax, xy0=(0,0), xy1=(+1,+1)):

    dy_tick=1
    dy_tickminor=dy_tick/4
    ax.set_yticks(np.arange(xy0[1],xy1[1]+dy_tick*3,dy_tick))
    ax.set_yticks(np.arange(xy0[1],xy1[1]+dy_tick*3,dy_tickminor), minor=True)
    ax.set_ylabel('$z$', labelpad=-1, fontsize=FS+1)
    ax.set_ylim([xy0[1],xy1[1]])

    dx_tick=1
    dx_tickminor=dx_tick/4
    ax.set_xticks(np.arange(xy0[0],xy1[0]+1e-5,dx_tick))
    ax.set_xticks(np.arange(xy0[0],xy1[0]+1e-5,dx_tickminor), minor=True)
    ax.set_xlabel('$x$', labelpad=-9, fontsize=FS+1)
    ax.set_xlim([xy0[0],xy1[0]])        
    
