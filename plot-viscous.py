#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2023-2024

"""
Viscous summary figure
"""

import copy, os, sys, code # code.interact(local=locals())
import numpy as np
import pickle

from dolfin import *
from state import *
from plottools import *
from specfabpy import plotting as sfplt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


kwargs_leg = {'fontsize':FS, 'frameon':False, 'fancybox':False, 'edgecolor':'k', 'handlelength':1.3, 'labelspacing':0.25, 'columnspacing':1.25}

def plot_state_compare(cases, cases_ref, nt, path=EXPPATH):

    s = 5.4
    fig = plt.figure(figsize= (1.3*s, 1.1*s))
    gs = gridspec.GridSpec(2, 2, wspace=0.30, hspace=0.35, left=0.09, right=0.975, top=0.97, bottom=0.26)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    timemax = 0

    for case in cases:

        print('Processing case %i ...'%(case))

        case_ref = cases_ref[0 if case <10 else 1]
        (meshargs, timevec,metrics, stokes,rheology,thermal) = load_state(case_ref, nt, path=path, skipfields=True)
        (vs_ref,ts_ref, xvec_ref,vs_x_ref,tsx_x_ref,tsz_x_ref) = metrics
        
        (meshargs, timevec,metrics, stokes,rheology,thermal) = load_state(case, nt, path=path, skipfields=False)
        (mesh, boundary_parts, ds, norm) = meshargs # unpack
        (vs,ts, xvec,vs_x,tsx_x,tsz_x) = metrics

        kwargs_line = get_linestyle(case, flbl=False)

        ax1.plot(timevec, vs, **kwargs_line)
        
        if case not in cases_ref:
            ts_x     = np.sqrt(np.power(tsx_x,2)     + np.power(tsz_x,2))
            ts_x_ref = np.sqrt(np.power(tsx_x_ref,2) + np.power(tsz_x_ref,2))
            ax2.plot(xvec, np.divide(vs_x,vs_x_ref), **kwargs_line)
            ax3.plot(xvec, np.divide(ts_x,ts_x_ref), **kwargs_line)
            
        Ezz_, xvec = Ezz_avg(stokes)
#        print(Ezz_, xvec)
        ax4.plot(xvec, Ezz_, **kwargs_line)
        

    ax = ax1
    ax.set_xticks(np.arange(0,1.1,0.04))
    ax.set_xticks(np.arange(0,1.1,0.02), minor=True)
    ax.set_xlim([0, 0.18])
    ax.set_xlabel('$t$')

    for ax in [ax2,ax3,ax4]: 
        ax.set_xticks(np.arange(0,1.1,0.2))
        ax.set_xticks(np.arange(0,1.1,0.1), minor=True)
        ax.set_xlim([0, 1])
        ax.set_xlabel('$x$')

    ax = ax2
    ax.set_yticks(np.arange(0,2,0.2))
    ax.set_yticks(np.arange(0,2,0.1), minor=True)

    ax = ax3
    ax.set_yticks(np.arange(0,2,0.1))
    ax.set_yticks(np.arange(0,2,0.05), minor=True)
    
    ax = ax4
    ax.set_yticks(np.arange(0,2,0.2))
    ax.set_yticks(np.arange(0,2,0.1), minor=True)

    for jj, ax in enumerate([ax1,ax2,ax3,ax4]):
        kwargs_panellbl = dict(fontsize=FS+2, frameon=False, bbox=(-0.26,1.135))
        sfplt.panellabel(ax, 2, r'\textit{(%s)}'%(chr(ord('a')+jj)), **kwargs_panellbl)

    ax1.set_yticks(np.arange(0,600,100))
    ax1.set_yticks(np.arange(0,600,50), minor=True)

    ax1.set_ylim([0,450])    
    ax2.set_ylim([0.1,0.95])
    ax3.set_ylim([0.95,1.25])
    ax4.set_ylim([0.5,1.3])

    ax1.set_ylabel(r'$\ev*{u_x}$ at $z=1$')
    ax2.set_ylabel(r'$u_x/u_{x}^{(\mathrm{pla})}$ at $z=1$')
    ax3.set_ylabel(r'$\abs*{\vb{t}}/\abs*{\vb{t}^{(\mathrm{pla})}}$ at $z=1$')
    ax4.set_ylabel(r'$\ev*{E_{zz}}$')

    hleg = ax1.legend(loc=3, bbox_to_anchor=(0.2, -2.28), ncol=2, title='$f=0$\quad\qquad\qquad\qquad\qquad\qquad$f=2/100$', **kwargs_leg)
    hleg._legend_box.sep = 9
    

#    if len(cases) > 1 and nt > 1300:
    if True:
        dx, dy = +0.015, +110
        I = np.nanargmax(vs)
        xy0 = (timevec[I], 240)
        arrprops = dict(arrowstyle="-|>", mutation_scale=18, connectionstyle="arc3", linewidth=1, edgecolor='k', facecolor='k')
        kwargs = dict(xycoords='data', textcoords='data', ma='center', zorder=20, arrowprops=arrprops)
        ax1.annotate(r'', xy=xy0, xytext=(xy0[0]+dx, xy0[1]+dy), **kwargs)
        dx, dy = -0.010, +110
        xy0 = (timevec[int(2*I + 0.7*np.nanargmax(vs[int(2*I):]))], 240)
        xy1 = (xy0[0]+dx, xy0[1]+dy)
        ax1.annotate(r'', xy=xy0, xytext=xy1, **kwargs)
        ax1.text(xy1[0]-4*dx, xy1[1], 'First and second plume\n passing the lid', ha='center', ma='left', va='bottom', fontsize=FS-1.5)

        ax2.text(0.9, 0.125, 'Slower lid velocities\n compared to plastic rheology', ha='right', ma='right', va='bottom', fontsize=FS-1.5)
        ax2.annotate(r'', xytext=(0.95, 0.3), xy=(0.95,0.11), **kwargs)
        
        ax3.text(0.1, 1.2425, 'Larger lid traction\n compared to plastic rheology', ha='left', ma='left', va='top', fontsize=FS-1.5)
        ax3.annotate(r'', xytext=(0.05, 1.18), xy=(0.05, 1.245), **kwargs)
        
        ax4.text(0.1, 0.525, 'Larger column-averaged viscosity\n for uniaxial $z$ compression\n compared to isotropic olivine', ha='left', ma='left', va='bottom', fontsize=FS-1.5)
        ax4.annotate(r'', xytext=(0.05, 0.7), xy=(0.05, 0.515), **kwargs)
        
    fname = '%s/viscous-summary.png'%(path)
    plt.savefig(fname, transparent=False, dpi=250)
    plt.close(fig)
    

if __name__ == "__main__":

    nt = 5395
    plot_state_compare((1,2,3, 11,12,13), (1,11), nt, path=EXPPATH)
#    plot_state_compare((1,), (1,), nt, path=EXPPATH)

