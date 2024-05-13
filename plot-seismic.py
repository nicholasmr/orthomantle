#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2023-2024

"""
Seismic summary figure
"""

import copy, os, sys, code # code.interact(local=locals())
import numpy as np
from scipy import integrate

from dolfin import *
from state import *
from plottools import *

### Config

# Time steps at which steady state has been reached for each experiment
nt = [5395]*6 

# Experiments to plot
cases = [1,2,3, 11,12,13]
#cases = [1,2,3]

### Figure geometry

figscale = 0.45
figsize = (15*figscale, 17.5*figscale)
fig = plt.figure(figsize=figsize)

### Subplot axes

gs = gridspec.GridSpec(6, 2, wspace=0.3, hspace=3, left=0.08, right=0.97, top=0.95, bottom=0.07)
ax1 = fig.add_subplot(gs[0:3,0])
ax2 = fig.add_subplot(gs[0:3,1])
ax3 = fig.add_subplot(gs[3:,0])
ax4 = fig.add_subplot(gs[3:5,1])

kwargs_panellbl    = dict(fontsize=FS+2, frameon=False, bbox=(-0.26,1.16))
kwargs_panellblalt = dict(fontsize=FS+2, frameon=False, bbox=(-0.26,1.2))

annotate = True

#-------------------------
# Generate plots
#-------------------------

N = 31 
xy0, xy1 = (0,0), (1,1)
x_1d = np.linspace(xy0[0],xy1[0],N)
y_1d = np.linspace(xy0[1],xy1[1],N)
xv, yv = np.meshgrid(x_1d, y_1d)

def get_velocities(case, i, theta=0, phi=0):

    (meshargs, timevec,metrics, stokes,rheology,thermal) = load_state(case, i, path=EXPPATH)
    
    vP, vS1, vS2 = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
    for ii in np.arange(N):
        for jj in np.arange(N):
            vP[ii,jj], vS1[ii,jj], vS2[ii,jj] = rheology.fabric.get_elastic_velocities(xv[ii,jj], yv[ii,jj], theta, phi)
            
    # propagation times across vertical columns
    dtx_vP, dtx_vS1, dtx_vS2 = np.zeros((N)), np.zeros((N)), np.zeros((N))
    I0 = 0 # 0 => bottom to top
    zvec = yv[I0:,0]
    for ii in np.arange(N):
        dtx_vP[ii]  = integrate.trapezoid(np.divide(1,vP[I0:,ii] ), zvec)
        dtx_vS1[ii] = integrate.trapezoid(np.divide(1,vS1[I0:,ii]), zvec)
        dtx_vS2[ii] = integrate.trapezoid(np.divide(1,vS2[I0:,ii]), zvec)

    # ... and in the isotropic case
    vPiso, vSiso, _ = rheology.fabric.get_elastic_velocities__isotropic()
    H = np.amax(y_1d)
    dtx_vP_iso = 1/vPiso * H
    dtx_vS_iso = 1/vSiso * H

    return vP,vS1,vS2, dtx_vP,dtx_vS1,dtx_vS2, vPiso,vSiso, dtx_vP_iso,dtx_vS_iso, meshargs
    
### Plot panels

C = len(cases)
vP, vS1, vS2, dvS = [None]*C, [None]*C, [None]*C, [None]*C
dtx_vP, dtx_vS1, dtx_vS2 = [None]*C, [None]*C, [None]*C
dtx_vP_ref, dtx_vS_ref = None, None
vPiso, vSiso = None, None

# Propagation time plot 

for ii, case in enumerate(cases):
    vP[ii], vS1[ii], vS2[ii], dtx_vP[ii],dtx_vS1[ii],dtx_vS2[ii], vPiso,vSiso,dtx_vP_ref,dtx_vS_ref,_ = get_velocities(case, nt[ii])
    dvS[ii] = vS1[ii] - vS2[ii]
    kwargs_line = get_linestyle(case, flbl=True)
    F = (dtx_vS1[ii] - dtx_vS2[ii])/dtx_vS_ref * 1e2
    ax4.plot(x_1d, F, **kwargs_line)

ax = ax4
sfplt.panellabel(ax, 2, r'\textit{(d)}', **kwargs_panellblalt)
kwargs_leg = {'fontsize':FS-1, 'frameon':False, 'fancybox':False, 'edgecolor':'k', 'handlelength':1.3, 'labelspacing':0.35, 'columnspacing':1.25}
hleg = ax.legend(loc=3, bbox_to_anchor=(-0.1, -1.16), ncol=1, **kwargs_leg)
hleg._legend_box.sep = 9
ax.text(0.070,-0.16, 'Increasing vertical S1--S2 travel-time\n difference relative to isotropic S-wave', va='top', ha='left', ma='left', fontsize=FS-1.5)
arrprops = dict(arrowstyle="-|>", mutation_scale=18, connectionstyle="arc3", linewidth=1, edgecolor='k', facecolor='k')
kwargs = dict(xycoords='data', textcoords='data', ma='center', zorder=20, arrowprops=arrprops)
ax.annotate(r'', xytext=(0.035, -0.1), xy=(0.035,-2.1), **kwargs)

ax.set_xlabel('$x$')
ax.set_ylabel('$(t_{\mathrm{S1}}-t_{\mathrm{S2}})/t_{\mathrm{S},0}$ (\%)')  
xticks = np.arange(0,1+1e-5,0.125)
ax.set_xticks(xticks[::2])
ax.set_xticks(xticks, minor=True)
ax.set_xlim([0,1])
yticks = np.arange(-8,-0+1e-2,1)
ax.set_yticks(yticks[::2])
ax.set_yticks(yticks, minor=True)
      
# \Delta V_S plots for plastic and isotropic rheology
      
def plot_dvS(ax, dvS,vSiso, lbl='', cblbl='?'):

    lvls = np.linspace(0,11,12)
    F = dvS/vSiso * 1e2
    h = ax.contourf(xv,yv, F, cmap=cracmap.acton_r, levels=lvls, extend='both')
    cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
    cbar.ax.set_xlabel('$\Delta V_{\mathrm{S}}^\mathrm{(%s)}/V_{\mathrm{S},0}$ (%s)'%(cblbl,'\%'))
    sfplt.panellabel(ax, 2, lbl, **kwargs_panellbl)
    set_axes(ax)
      

case = 0
plot_dvS(ax1, dvS[case],vSiso, cblbl='pla', lbl=r'\textit{(a)}')
if annotate: ax1.text(0.5,1.01, 'Vertical $\Delta V_\mathrm{S}$ for plastic rheology\n relative to $V_\mathrm{S}$ of isotropic CPO', va='bottom', ha='center', ma='center', fontsize=FS-1)

case = 1
plot_dvS(ax2, dvS[case],vSiso, cblbl='iso', lbl=r'\textit{(b)}')
if annotate: ax2.text(0.5,1.01, 'Vertical $\Delta V_\mathrm{S}$ for isotropic rheology\n relative to $V_\mathrm{S}$ of isotropic CPO', va='bottom', ha='center', ma='center', fontsize=FS-1)

# Relative \Delta V_S

ax = ax3
F = np.divide(dvS[2],dvS[1]) # orthotropic/isotropic in pct
lvls = np.linspace(0.5,1.5,11)
dtickcbar = 1
h = ax.contourf(xv,yv, F, cmap=cracmap.bam_r, levels=lvls, extend='both')
cbar = plt.colorbar(h, ax=ax, **kwargs_cb)
cbar.ax.set_xlabel(r'$\Delta V_\mathrm{S}^\mathrm{(ort)}/\Delta V_\mathrm{S}^\mathrm{(iso)}$')
if annotate: ax.text(0.5,1.01, 'Vertical $\Delta V_\mathrm{S}$ of orthotropic rheology\n relative to isotropic rheology', va='bottom', ha='center', ma='center', fontsize=FS-1)
set_axes(ax)
sfplt.panellabel(ax, 2, r'\textit{(c)}', **kwargs_panellbl)
    
### Save figure

fname = '%s/seismic-summary.png'%(EXPPATH)
plt.savefig(fname, dpi=250)
plt.close(fig)

