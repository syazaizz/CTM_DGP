## Import the required python packages

import numpy as np
import math

import sys
import classylss
import classylss.binding as CLASS

from ctm import CTM, Cosmo
from scipy.integrate import quad

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import animation
from IPython.display import HTML

from matplotlib import rc
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.transforms import blended_transform_factory

from scipy.interpolate import interp1d as interp

import seaborn as sns
cmap = sns.color_palette("muted")

mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'cm'
mpl.rcParams["lines.linewidth"] = 2.2
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["axes.labelsize"] = 14.
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.labelsize"] = 14.
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.labelsize"] = 14.
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.minor.bottom"] = False
mpl.rcParams["xtick.minor.top"] = False
mpl.rcParams["ytick.minor.left"] = False
mpl.rcParams["ytick.minor.right"] = False

colours = ["black", cmap[4], cmap[1], cmap[6], cmap[-1]]
linestyles = ["-", "--", "-.", ":"]


"""
Define k values and z values
"""
k_vals = np.logspace(-3, np.log10(0.9), 1000)

z_vals = np.linspace(0, 200, 1000)


"""
## Load the .txt files
"""
## Load in the DGP power spectra for DGP here

## DGP undamped data, z = 0,1,2,3,4,5

P_dgp_0=np.loadtxt(your_path + "/dgp_runs/P_dgp_0.txt")
P_dgp_1=np.loadtxt(your_path + "/dgp_runs/P_dgp_1.txt")
P_dgp_2=np.loadtxt(your_path + "/dgp_runs/P_dgp_2.txt")
P_dgp_3=np.loadtxt(your_path + "/dgp_runs/P_dgp_3.txt")
P_dgp_4=np.loadtxt(your_path + "/dgp_runs/P_dgp_4.txt")
P_dgp_5=np.loadtxt(your_path + "/dgp_runs/P_dgp_5.txt")

## DGP damped data, kc = 5.0, z = 0,1,2,3,4,5

P_dgp_0_k5=np.loadtxt(your_path + "/dgp_runs/P_dgp_0_k5.txt")
P_dgp_1_k5=np.loadtxt(your_path + "/dgp_runs/P_dgp_1_k5.txt")
P_dgp_2_k5=np.loadtxt(your_path + "/dgp_runs/P_dgp_2_k5.txt")
P_dgp_3_k5=np.loadtxt(your_path + "/dgp_runs/P_dgp_3_k5.txt")
P_dgp_4_k5=np.loadtxt(your_path + "/dgp_runs/P_dgp_4_k5.txt")
P_dgp_5_k5=np.loadtxt(your_path + "/dgp_runs/P_dgp_5_k5.txt")

## Load in the CTM power spectra for CDM here

## CDM undamped data, z = 0,1,2,3,4,5

P_cdm_0=np.loadtxt(your_path + "/dgp_runs/P_cdm_0.txt")
P_cdm_1=np.loadtxt(your_path + "/dgp_runs/P_cdm_1.txt")
P_cdm_2=np.loadtxt(your_path + "/dgp_runs/P_cdm_2.txt")
P_cdm_3=np.loadtxt(your_path + "/dgp_runs/P_cdm_3.txt")
P_cdm_4=np.loadtxt(your_path + "/dgp_runs/P_cdm_4.txt")
P_cdm_5=np.loadtxt(your_path + "/dgp_runs/P_cdm_5.txt")

## CDM Damped data, z = 0,1,2,3,4,5

P_cdm_0_k5=np.loadtxt(your_path + "/dgp_runs/P_cdm_0_k5.txt")
P_cdm_1_k5=np.loadtxt(your_path + "/dgp_runs/P_cdm_1_k5.txt")
P_cdm_2_k5=np.loadtxt(your_path + "/dgp_runs/P_cdm_2_k5.txt")
P_cdm_3_k5=np.loadtxt(your_path + "/dgp_runs/P_cdm_3_k5.txt")
P_cdm_4_k5=np.loadtxt(your_path + "/dgp_runs/P_cdm_4_k5.txt")
P_cdm_5_k5=np.loadtxt(your_path + "/dgp_runs/P_cdm_5_k5.txt")

## Load Emulator data for z = 0,1,2,3,4,5

k_emu_0=np.loadtxt(your_path + "/emulator_runs/k_emu.txt")
P_emu_0=np.loadtxt(your_path + "/emulator_runs/P_emu_0.txt")
P_emu_1=np.loadtxt(your_path + "/emulator_runs/P_emu_1.txt")
P_emu_2=np.loadtxt(your_path + "/emulator_runs/P_emu_2.txt")
P_emu_3=np.loadtxt(your_path + "/emulator_runs/P_emu_3.txt")
P_emu_4=np.loadtxt(your_path + "/emulator_runs/P_emu_4.txt")
P_emu_5=np.loadtxt(your_path + "/emulator_runs/P_emu_5.txt")


"""
Calculate the linear power spectrum at z = 0,1,2,3
"""
P_lin_0=CTM().linear_power(k_vals)
P_lin_1=CTM().linear_power(k_vals, z_val=1)
P_lin_2=CTM().linear_power(k_vals, z_val=2)
P_lin_3=CTM().linear_power(k_vals, z_val=3)


"""
D(z) for DGP gravity, to calculate renormalisation factor for DGP
"""
def scale_factor(z_val):

    return (1+z_val)**(-1)

def D1_dgp (z_vals, alpha, Ho=67.37, omega0_m=(0.11933+0.02242)/(0.6737**2), omega0_lamda=1.0-(0.11933+0.02242)/(0.6737**2)):

    def rc(alpha, Ho=67.37, omega0_m=omega0_m):

        return ((1-omega0_m)**(1/(alpha-2)))/Ho

    def hubble_dgp(z_val, alpha, Ho=67.37, omega0_m=omega0_m, omega0_lamda=omega0_lamda):

        A = (Ho**2)*((omega0_m/(scale_factor(z_val)**3)))
        B = np.sqrt(A)**alpha/((rc(alpha))**(2-alpha))
        H = (A+B)**(1/2)

        return H

# Solving integral

    def integrand(zz, alpha):

        return (1+zz)/(hubble_dgp(zz, alpha))**3

    def I(z, alpha):

        return quad(integrand, z, 150, args=(alpha))[0]

# Solving D(z)

    def linear_growth_factor(z_val, alpha, Ho=67.37, omega0_m=(0.11933+0.02242)/(0.6737**2)):

        return ((5*omega0_m*Ho**2)/2)*hubble_dgp(z_val, alpha)*I(z_val, alpha)

    def D1_dgp(z_vals, alpha):

        D=np.zeros_like(z_vals)

        for i in range(len(z_vals)):

            D[i] = linear_growth_factor(z_vals[i], alpha=alpha)

        D_vals = interp(z_vals, D)

        D_norm = D/D_vals(0.0)

        return D_norm

    return D1_dgp(z_vals, alpha)

D1_dgp_array=D1_dgp(z_vals, 1.0)

D1_dgp_func=interp(z_vals, D1_dgp_array)

## renormalisation factor for DGP

renorm_factor_1=(CTM().linear_growth_factor(z_val=1.0)/D1_dgp_func(1.0))**2
renorm_factor_2=(CTM().linear_growth_factor(z_val=2.0)/D1_dgp_func(2.0))**2
renorm_factor_3=(CTM().linear_growth_factor(z_val=3.0)/D1_dgp_func(3.0))**2
renorm_factor_4=(CTM().linear_growth_factor(z_val=4.0)/D1_dgp_func(4.0))**2
renorm_factor_5=(CTM().linear_growth_factor(z_val=5.0)/D1_dgp_func(5.0))**2



"""
1. Plot of DGP power at z=0,1,2,3 with no damping
"""
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)

gs.update(wspace=0.05, hspace=0.1)

ax1 = fig.add_subplot(gs[0]) #z=0 panel
ax2 = fig.add_subplot(gs[1]) #z=1 panel
ax3 = fig.add_subplot(gs[2]) #z=2 panel
ax4 = fig.add_subplot(gs[3]) #z=3 panel

ax1.loglog(k_vals, k_vals**3*P_dgp_0/(2.0*np.pi**2), color="black", alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax1.loglog(k_vals, k_vals**3*P_lin_0/(2.0*np.pi**2), color="black", alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax1.loglog(k_vals, k_vals**3*P_cdm_0/(2.0*np.pi**2), color="black", alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax1.set_ylabel(r"$\Delta^2\left(k\right)$")
ax1.set_xlim([1e-2, 0.9])
ax1.set_ylim([1e-3, 5])
ax1.legend(loc="lower right", frameon=False, fontsize=14.)
plt.setp(ax1.get_xticklabels(), visible=False)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor="grey", alpha=0.5)
label=r"$\mathrm{z=0}$"
ax1.text(0.05, 0.95, label, transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)

ax2.loglog(k_vals, k_vals**3*P_dgp_1*renorm_factor_1/(2.0*np.pi**2), color=colours[1], alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax2.loglog(k_vals, k_vals**3*P_lin_1/(2.0*np.pi**2), color=colours[1], alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax2.loglog(k_vals, k_vals**3*P_cdm_1/(2.0*np.pi**2), color=colours[1], alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax2.set_xlim([1e-2, 0.9])
ax2.set_ylim([1e-3, 5])
ax2.legend(loc="lower right", frameon=False, fontsize=14.)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[1], alpha=0.5)
label=r"$\mathrm{z=1}$"
ax2.text(0.05, 0.95, label, transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax3.loglog(k_vals, k_vals**3*P_dgp_2*renorm_factor_2/(2.0*np.pi**2), color=colours[2], alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax3.loglog(k_vals, k_vals**3*P_lin_2/(2.0*np.pi**2), color=colours[2], alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax3.loglog(k_vals, k_vals**3*P_cdm_2/(2.0*np.pi**2), color=colours[2], alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax3.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax3.set_ylabel(r"$\Delta^2\left(k\right)$")
ax3.set_xlim([1e-2, 0.9])
ax3.set_ylim([1e-3, 5])
ax3.legend(loc="lower right", frameon=False, fontsize=14.)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[2], alpha=0.5)
label=r"$\mathrm{z=2}$"
ax3.text(0.05, 0.95, label, transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax4.loglog(k_vals, k_vals**3*P_dgp_3*renorm_factor_3/(2.0*np.pi**2), color=colours[3], alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax4.loglog(k_vals, k_vals**3*P_lin_3/(2.0*np.pi**2), color=colours[3], alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax4.loglog(k_vals, k_vals**3*P_cdm_3/(2.0*np.pi**2), color=colours[3], alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax4.set_xlim([1e-2, 0.9])
ax4.set_ylim([1e-3, 5])
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax4.legend(loc="lower right", frameon=False, fontsize=14.)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[3], alpha=0.5)
label=r"$\mathrm{z=3}$"
ax4.text(0.05, 0.95, label, transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)

plt.savefig("undamped_power_spec.pdf", bbox_inches="tight")



"""
2. Plot of DGP power at z=0,1,2,3 with damping, kc=5.0
"""
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)

gs.update(wspace=0.05, hspace=0.1)

ax1 = fig.add_subplot(gs[0]) #z=0 panel
ax2 = fig.add_subplot(gs[1]) #z=1 panel
ax3 = fig.add_subplot(gs[2]) #z=2 panel
ax4 = fig.add_subplot(gs[3]) #z=3 panel

ax1.loglog(k_vals, k_vals**3*P_dgp_0_k5/(2.0*np.pi**2), color="black", alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax1.loglog(k_vals, k_vals**3*P_lin_0/(2.0*np.pi**2), color="black", alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax1.loglog(k_vals, k_vals**3*P_cdm_0_k5/(2.0*np.pi**2), color="black", alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax1.set_ylabel(r"$\Delta^2\left(k\right)$")
ax1.set_xlim([1e-2, 0.9])
ax1.set_ylim([1e-3, 5])
ax1.legend(loc="lower right", frameon=False, fontsize=14.)
plt.setp(ax1.get_xticklabels(), visible=False)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor="grey", alpha=0.5)
label=r"$\mathrm{z=0}$"
ax1.text(0.05, 0.95, label, transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax2.loglog(k_vals, k_vals**3*P_dgp_1_k5*renorm_factor_1/(2.0*np.pi**2), color=colours[1], alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax2.loglog(k_vals, k_vals**3*P_lin_1/(2.0*np.pi**2), color=colours[1], alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax2.loglog(k_vals, k_vals**3*P_cdm_1_k5/(2.0*np.pi**2), color=colours[1], alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax2.set_xlim([1e-2, 0.9])
ax2.set_ylim([1e-3, 5])
ax2.legend(loc="lower right", frameon=False, fontsize=14.)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[1], alpha=0.5)
label=r"$\mathrm{z=1}$"
ax2.text(0.05, 0.95, label, transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax3.loglog(k_vals, k_vals**3*P_dgp_2_k5*renorm_factor_2/(2.0*np.pi**2), color=colours[2], alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax3.loglog(k_vals, k_vals**3*P_lin_2/(2.0*np.pi**2), color=colours[2], alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax3.loglog(k_vals, k_vals**3*P_cdm_2_k5/(2.0*np.pi**2), color=colours[2], alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax3.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax3.set_ylabel(r"$\Delta^2\left(k\right)$")
ax3.set_xlim([1e-2, 0.9])
ax3.set_ylim([1e-3, 5])
ax3.legend(loc="lower right", frameon=False, fontsize=14.)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[2], alpha=0.5)
label=r"$\mathrm{z=2}$"
ax3.text(0.05, 0.95, label, transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax4.loglog(k_vals, k_vals**3*P_dgp_3_k5*renorm_factor_3/(2.0*np.pi**2), color=colours[3], alpha=0.8, label=r"$\mathrm{DGP\ CTM}$", linewidth=2.2)
ax4.loglog(k_vals, k_vals**3*P_lin_3/(2.0*np.pi**2), color=colours[3], alpha=0.8, label=r"$\mathrm{linear\ \Lambda -CDM}$", linewidth=2.2, linestyle='--')
ax4.loglog(k_vals, k_vals**3*P_cdm_3_k5/(2.0*np.pi**2), color=colours[3], alpha=0.8, label=r"$\mathrm{\Lambda -CDM\ CTM}$", linewidth=2.2, linestyle='dotted')

ax4.set_xlim([1e-2, 0.9])
ax4.set_ylim([1e-3, 5])
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax4.legend(loc="lower right", frameon=False, fontsize=14.)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[3], alpha=0.5)
label=r"$\mathrm{z=3}$"
ax4.text(0.05, 0.95, label, transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)

plt.savefig("k1_damped_power_spec.pdf", bbox_inches="tight")



"""
3. Plot of DGP power spec at z=0,1,2,3, Comparison between undamped and damped power spec
"""
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)

gs.update(wspace=0.05, hspace=0.1)

ax1 = fig.add_subplot(gs[0]) #z=0 panel
ax2 = fig.add_subplot(gs[1]) #z=1 panel
ax3 = fig.add_subplot(gs[2]) #z=2 panel
ax4 = fig.add_subplot(gs[3]) #z=3 panel

ax1.loglog(k_vals, k_vals**3*P_dgp_0/(2.0*np.pi**2), color="black", alpha=0.8, label=r"$\mathrm{Undamped}$", linewidth=2.2)
ax1.loglog(k_vals, k_vals**3*P_dgp_0_k5/(2.0*np.pi**2), color="black", linestyle='--', alpha=0.8, label=r"$\mathrm{Damped}$", linewidth=2.2)
ax1.set_ylabel(r"$\Delta^2\left(k\right)$")
ax1.set_xlim([0.1, 0.9])
ax1.set_ylim([0.55, 1.6])
ax1.legend(loc="lower right", frameon=False, fontsize=14.)
# plt.setp(ax1.get_xticklabels(), visible=False)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor="grey", alpha=0.5)
label=r"$\mathrm{z=1}$"
ax1.text(0.05, 0.95, label, transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax2.loglog(k_vals, k_vals**3*P_dgp_1*renorm_factor_1/(2.0*np.pi**2), color=colours[1], alpha=0.8, label=r"$\mathrm{Undamped}$", linewidth=2.2)
ax2.loglog(k_vals, k_vals**3*P_dgp_1_k5*renorm_factor_1/(2.0*np.pi**2), color=colours[1], linestyle='--', alpha=0.8, label=r"$\mathrm{Damped}$", linewidth=2.2)
ax2.set_xlim([1e-2, 0.9])
ax2.set_ylim([1e-3, 5])
ax2.legend(loc="lower right", frameon=False, fontsize=14.)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[1], alpha=0.5)
label=r"$\mathrm{z=1}$"
ax2.text(0.05, 0.95, label, transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax3.loglog(k_vals, k_vals**3*P_dgp_2*renorm_factor_2/(2.0*np.pi**2), color=colours[2], alpha=0.8, label=r"$\mathrm{Undamped}$", linewidth=2.2)
ax3.loglog(k_vals, k_vals**3*P_dgp_2_k5*renorm_factor_2/(2.0*np.pi**2), color=colours[2], linestyle='--', alpha=0.8, label=r"$\mathrm{Damped}$", linewidth=2.2)
ax3.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax3.set_ylabel(r"$\Delta^2\left(k\right)$")
ax3.set_xlim([1e-2, 0.9])
ax3.set_ylim([1e-3, 5])
ax3.legend(loc="lower right", frameon=False, fontsize=14.)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[2], alpha=0.5)
label=r"$\mathrm{z=2}$"
ax3.text(0.05, 0.95, label, transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)


ax4.loglog(k_vals, k_vals**3*P_dgp_3_k5*renorm_factor_3/(2.0*np.pi**2), color=colours[3], alpha=0.8, label=r"$\mathrm{Undamped}$", linewidth=2.2)
ax4.loglog(k_vals, k_vals**3*P_dgp_3_k5*renorm_factor_3/(2.0*np.pi**2), color=colours[3], linestyle='--', alpha=0.8, label=r"$\mathrm{Damped}$", linewidth=2.2)
ax4.set_xlim([1e-2, 0.9])
ax4.set_ylim([1e-3, 5])
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax4.legend(loc="lower right", frameon=False, fontsize=14.)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor=colours[3], alpha=0.5)
label=r"$\mathrm{z=3}$"
ax4.text(0.05, 0.95, label, transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)

plt.savefig("Undamped_damped.pdf", bbox_inches="tight")



"""
Difference between calculated spectrum and emulator
"""
def difference(P_true, P_approx):

  return P_approx/P_true-1.0



"""
4. Plot the difference between DGP and Emulator (solid lines) and the difference between \Lambda CDM and Emulator (dashed lines)
"""
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2)

gs.update(wspace=0.05, hspace=0.1)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

######## undamped plot ###########

ax1.semilogx(k_vals, difference(P_emu_0, P_dgp_0), color="black", alpha=0.8, label=r"$\mathrm{z=0}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_emu_1, P_dgp_1*renorm_factor_1), color=colours[1], alpha=0.8, label=r"$\mathrm{z=1}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_emu_2, P_dgp_2*renorm_factor_2), color=colours[2], alpha=0.8, label=r"$\mathrm{z=2}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_emu_3, P_dgp_3*renorm_factor_3), color=colours[3], alpha=0.8, label=r"$\mathrm{z=3}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_emu_4, P_dgp_4*renorm_factor_4), color=colours[4], alpha=0.8, label=r"$\mathrm{z=4}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_emu_5, P_dgp_5*renorm_factor_5), color="maroon", alpha=0.8, label=r"$\mathrm{z=5}$", linewidth=2.2)

ax1.semilogx(k_vals, difference(P_emu_0, P_cdm_0), color="black", linestyle='--', alpha=0.8)
ax1.semilogx(k_vals, difference(P_emu_1, P_cdm_1), color=colours[1], linestyle='--', alpha=0.8)
ax1.semilogx(k_vals, difference(P_emu_2, P_cdm_2), color=colours[2], linestyle='--', alpha=0.8)
ax1.semilogx(k_vals, difference(P_emu_3, P_cdm_3), color=colours[3], linestyle='--', alpha=0.8)
ax1.semilogx(k_vals, difference(P_emu_4, P_cdm_4), color=colours[4], linestyle='--', alpha=0.8)
ax1.semilogx(k_vals, difference(P_emu_5, P_cdm_5), color="maroon", linestyle='--', alpha=0.8)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor="khaki", alpha=0.5)
label=r"$\mathrm{Undamped}$"
ax1.text(0.05, 0.7, label, transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)

ax1.set_ylabel(r"$\mathrm{\frac{P_{calc}(k)-P_{emu}(k)}{P_{emu}(k)}}$")
ax1.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
# ax1.set_title('Undamped')
ax1.legend(loc="lower left", frameon=False, fontsize=14.)
ax1.set_xlim([1e-2, 0.9])
ax1.fill_between(k_vals, -0.05, 0.05, color="black", alpha=0.2)

######## damped plot ###########

ax2.semilogx(k_vals, difference(P_emu_0, P_dgp_0_k5), color="black", alpha=0.8, label=r"$\mathrm{z=0}$", linewidth=2.2)
ax2.semilogx(k_vals, difference(P_emu_1, P_dgp_1_k5*renorm_factor_1), color=colours[1], alpha=0.8, label=r"$\mathrm{z=1}$", linewidth=2.2)
ax2.semilogx(k_vals, difference(P_emu_2, P_dgp_2_k5*renorm_factor_2), color=colours[2], alpha=0.8, label=r"$\mathrm{z=2}$", linewidth=2.2)
ax2.semilogx(k_vals, difference(P_emu_3, P_dgp_3_k5*renorm_factor_3), color=colours[3], alpha=0.8, label=r"$\mathrm{z=3}$", linewidth=2.2)
ax2.semilogx(k_vals, difference(P_emu_4, P_dgp_4_k5*renorm_factor_4), color=colours[4], alpha=0.8, label=r"$\mathrm{z=4}$", linewidth=2.2)
ax2.semilogx(k_vals, difference(P_emu_5, P_dgp_5_k5*renorm_factor_5), color="maroon", alpha=0.8, label=r"$\mathrm{z=5}$", linewidth=2.2)

ax2.semilogx(k_vals, difference(P_emu_0, P_cdm_0_k5), color="black", linestyle='--', alpha=0.8)
ax2.semilogx(k_vals, difference(P_emu_1, P_cdm_1_k5), color=colours[1], linestyle='--', alpha=0.8)
ax2.semilogx(k_vals, difference(P_emu_2, P_cdm_2_k5), color=colours[2], linestyle='--', alpha=0.8)
ax2.semilogx(k_vals, difference(P_emu_3, P_cdm_3_k5), color=colours[3], linestyle='--', alpha=0.8)
ax2.semilogx(k_vals, difference(P_emu_4, P_cdm_4_k5), color=colours[4], linestyle='--', alpha=0.8)
ax2.semilogx(k_vals, difference(P_emu_5, P_cdm_5_k5), color="maroon", linestyle='--', alpha=0.8)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor="khaki", alpha=0.5)
label=r"$\mathrm{Damped}$"
ax2.text(0.05, 0.7, label, transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', bbox=props)

ax2.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.legend(loc="ulower left", frameon=False, fontsize=14.)
ax2.set_xlim([1e-2, 0.9])
ax2.fill_between(k_vals, -0.05, 0.05, color="black", alpha=0.2)

plt.savefig("diff_emulator.pdf", bbox_inches="tight")



"""
5. Plot the difference between DGP and CDM at z=0,1,2,3,4,5, undamped kc=0
"""
fig = plt.figure(figsize=(6, 5))
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.05, hspace=0.1)

ax1 = fig.add_subplot(gs[0])

ax1.semilogx(k_vals, difference(P_dgp_0, P_cdm_0), color="black",  alpha=0.8, label=r"$\mathrm{z=0}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_dgp_1*renorm_factor_1, P_cdm_1), color=colours[1], alpha=0.8, label=r"$\mathrm{z=1}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_dgp_2*renorm_factor_2, P_cdm_2), color=colours[2], alpha=0.8, label=r"$\mathrm{z=2}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_dgp_3*renorm_factor_3, P_cdm_3), color=colours[3], alpha=0.8, label=r"$\mathrm{z=3}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_dgp_4*renorm_factor_4, P_cdm_4), color=colours[4], alpha=0.8, label=r"$\mathrm{z=4}$", linewidth=2.2)
ax1.semilogx(k_vals, difference(P_dgp_5*renorm_factor_5, P_cdm_5,), color="maroon", alpha=0.8, label=r"$\mathrm{z=5}$", linewidth=2.2)

ax1.set_ylabel(r"$\mathrm{\frac{P_{cdm}(k)-P_{dgp}(k)}{P_{dgp}(k)}}$")
ax1.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax1.legend(loc="lower left", frameon=False, fontsize=14.)

ax1.set_xlim([1e-2, 0.9])
ax1.fill_between(k_vals, -0.05, 0.05, color="black", alpha=0.2)

plt.savefig("P_cdm_P_dgp.pdf", bbox_inches="tight")



"""
6. plot P(k) for DGP
"""
fig = plt.figure(figsize=(6, 5))
gs = gridspec.GridSpec(1, 1)

gs.update(wspace=0, hspace=0)

ax1 = fig.add_subplot(gs[0])

ax1.loglog(k_vals, P_dgp_0 , label=r"$\mathrm{z=0}$", color="black", linewidth=2.2)
ax1.loglog(k_vals, P_dgp_1*renorm_factor_1, label=r"$\mathrm{z=1}$", color=colours[1],  linewidth=2.2)
ax1.loglog(k_vals, P_dgp_2*renorm_factor_2,  label=r"$\mathrm{z=2}$", color=colours[2], linewidth=2.2)
ax1.loglog(k_vals, P_dgp_3*renorm_factor_3,  label=r"$\mathrm{z=3}$", color=colours[3], linewidth=2.2)
ax1.loglog(k_vals, P_dgp_4*renorm_factor_4, label=r"$\mathrm{z=4}$", color=colours[4], linewidth=2.2)
ax1.loglog(k_vals, P_dgp_5*renorm_factor_5,  label=r"$\mathrm{z=5}$", color="maroon", linewidth=2.2)

ax1.set_xlabel(r"$k\ [\mathrm{h}\ \mathrm{Mpc}^{-1}]$")
ax1.set_ylabel(r"$\mathrm{P}\left(k\right)\ [\mathrm{Mpc}^3\ \mathrm{h}^{-3}]$")
ax1.set_xlim([1e-3, 1])
ax1.legend(loc="upper right left", frameon=False, fontsize=14.)

plt.savefig('P(k)_pdf.pdf', bbox_inches="tight")



"""
7. PLot linear power and Zel'dovich spectrum
"""
## Calculate the linear power spectrum at z=0 using GCTM

P_lin=CTM().linear_power(k_vals)

# Calculate the Zel'dovich power spectrum using GCTM

P_zel=CTM().zeldovich_power(input_k=k_vals)

# PLot

fig = plt.figure(figsize=(6, 5))
gs = gridspec.GridSpec(1, 1)

gs.update(wspace=0.0, hspace=0.0)

ax1 = fig.add_subplot(gs[0])

ax1.loglog(k_vals, P_lin, color=colours[1], alpha=0.8, linestyle='-', linewidth=2.2, label=r"$\mathrm{Linear}$")
ax1.loglog(k_vals, P_zel, color=colours[2], alpha=0.8, linestyle='--', linewidth=2.2, label=r"$\mathrm{Zeldovich}$")
ax1.legend(loc="upper right", frameon=False, fontsize=14.)
ax1.set_xlim([1e-3, 1])
ax1.set_ylim([1e2, 4e4])
ax1.set_xlabel(r"$k\ [\mathrm{h}^3\ \mathrm{Mpc}^{-3}]$")
ax1.set_ylabel(r"$\mathrm{P}(k)\ [\mathrm{Mpc}^3\ \mathrm{h}^{-3}]$")

plt.savefig('lin_zel.pdf', bbox_inches="tight")



"""
8. Plot DGP, alpha = 0 and  1  and CDM
"""
z_vals=np.linspace(0.0, 200.0, 500)

D1_lambdacdm = CTM().linear_growth_factor(z_vals)

plt.plot(z_vals, D1_dgp(z_vals, 0.0), label=r"$\mathrm{\alpha = 0}$", linewidth=2.2, color=colours[1])
plt.plot(z_vals, D1_dgp(z_vals, 1.0), label=r"$\mathrm{\alpha = 1}$", linewidth=2.2, color=colours[2])
plt.plot(z_vals, D1_lambdacdm, linestyle='dotted', label=r"$\mathrm{\Lambda -CDM}$", color=colours[3], linewidth=2.2)


plt.xlabel(r"$\mathrm{redshift,\ z}$")
plt.ylabel(r"$\mathrm{Linear\ Growth\ Factor,\ D(z)}$")
plt.legend(loc="upper right", frameon=False, fontsize=14.)
plt.xlim([0,10])

plt.savefig('DGP.pdf', bbox_inches="tight")
