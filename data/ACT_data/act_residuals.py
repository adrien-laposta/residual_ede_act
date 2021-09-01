import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import FortranFile
import matplotlib

matplotlib.rcParams['axes.linewidth'] = 2 #10 / 5

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'serif'

matplotlib.rcParams['axes.labelsize'] = 16 #75 / 5

matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.axis'] = 'both'
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.linestyle'] = "dotted"
matplotlib.rcParams['grid.linewidth'] = 0.8
matplotlib.rcParams['grid.alpha'] = 0.5
matplotlib.rcParams['xtick.labelsize'] = 15 #65 / 5
matplotlib.rcParams['xtick.major.size'] = 5 #25 / 5
matplotlib.rcParams['xtick.major.width'] = 2 #10 / 5
matplotlib.rcParams['xtick.major.pad'] = 6 #30 / 5
matplotlib.rcParams['xtick.direction'] = 'in'

matplotlib.rcParams['ytick.labelsize'] = 16 #65 / 5
matplotlib.rcParams['ytick.major.size'] = 5 #25 / 5
matplotlib.rcParams['ytick.major.width'] = 2 #10 / 5
matplotlib.rcParams['ytick.major.pad'] = 6 #30 / 5
matplotlib.rcParams['ytick.direction'] = 'in'


matplotlib.rcParams['legend.fontsize'] = 16 #65 / 5
matplotlib.rcParams['legend.frameon'] = False

###################
#### CONSTANTS ####
###################
tt_lmax = 5000
bmin = 0
b0 = 5
nbin = 260
nbinw = 130
nbintt = 40
nbinte = 45
nbinee = 45
lmax_win = 7925
bmax_win = 520
bmax = 52
##################
##################
##################

###############
#### FILES ####
###############
like_file = "cl_cmb_ap.dat"
cov_file = "c_matrix_ap.dat"
bbldeep_file = "coadd_bpwf_15mJy_191127_lmin2.npz"
bblwide_file = "coadd_bpwf_100mJy_191127_lmin2.npz"
camb_file = "bf_ACTPol_lcdm.minimum.theory_cl"
ell_act_file = "Binning.dat"
##############
##############
##############

#######################
#### LOAD ACT DATA ####
#######################
bval, X_data, X_sig = np.genfromtxt(like_file, max_rows=nbin,
                                    delimiter=None,unpack=True)
ell_act = np.loadtxt("Binning.dat")

f = FortranFile(cov_file, "r")
cov = f.read_reals(dtype = float).reshape((nbin, nbin))
for i_index in range(nbin):
    for j_index in range(i_index, nbin):
        cov[i_index, j_index] = cov[j_index, i_index]
err_act = np.sqrt(cov.diagonal())
bbldeep = np.load(bbldeep_file)["bpwf"]
win_func_d = np.zeros((bmax_win, lmax_win))
win_func_d[:bmax_win, 1:lmax_win] = bbldeep[:bmax_win, :lmax_win]

bblwide = np.load(bblwide_file)["bpwf"]
win_func_w = np.zeros((bmax_win, lmax_win))
win_func_w[:bmax_win, 1:lmax_win] = bblwide[:bmax_win, :lmax_win]
#######################
#######################
#######################

######################
#### CAMB BINNING ####
######################
camb = np.loadtxt(camb_file)
ellcamb = camb[:, 0]
dltt = camb[:, 1]
dlte = camb[:, 2]
dlee = camb[:, 3]

X_model = np.zeros(nbin)
Y = np.zeros(nbin)

l_list = np.arange(2, tt_lmax + 1)
cltt = np.zeros(lmax_win)
clte = np.zeros(lmax_win)
clee = np.zeros(lmax_win)

cltt[1:tt_lmax] = dltt[:tt_lmax-1]/l_list/(l_list+1)*2*np.pi
clte[1:tt_lmax] = dlte[:tt_lmax-1]/l_list/(l_list+1)*2*np.pi
clee[1:tt_lmax] = dlee[:tt_lmax-1]/l_list/(l_list+1)*2*np.pi

cltt_d = win_func_d[2*bmax:3*bmax, 1:lmax_win]@cltt[1:lmax_win]
clte_d = win_func_d[6*bmax:7*bmax, 1:lmax_win]@clte[1:lmax_win]
clee_d = win_func_d[9*bmax:10*bmax, 1:lmax_win]@clee[1:lmax_win]

cltt_w = win_func_w[2*bmax:3*bmax, 1:lmax_win]@cltt[1:lmax_win]
clte_w = win_func_w[6*bmax:7*bmax, 1:lmax_win]@clte[1:lmax_win]
clee_w = win_func_w[9*bmax:10*bmax, 1:lmax_win]@clee[1:lmax_win]

X_model[:nbintt]=cltt_d[b0:b0+nbintt]
X_model[nbintt:nbintt+nbinte]=clte_d[:nbinte]
X_model[nbintt+nbinte:nbintt+nbinte+nbinee]=clee_d[:nbinee]
X_model[nbintt+nbinte+nbinee:2*nbintt+nbinte+nbinee]=cltt_w[b0:b0+nbintt]
X_model[2*nbintt+nbinte+nbinee:2*nbintt+2*nbinte+nbinee]=clte_w[:nbinte]
X_model[2*nbintt+2*nbinte+nbinee:2*nbintt+2*nbinte+2*nbinee]=clee_w[:nbinee]
#####################
#####################
#####################


##############
#### PLOT ####
##############
def plot(spec, dls, scale = "log"):

    fig = plt.figure(figsize = (8, 7))

    ids_d = {"tt": [0, 40],
             "te": [40, 85],
             "ee": [85, 130]}

    ids_w = {"tt": [130, 170],
             "te": [170, 215],
             "ee": [215, 260]}
    ids_ell = {"tt": [5, 45],
               "te": [0, 45],
               "ee": [0, 45]}

    id_d0, id_d1 = ids_d[spec][0], ids_d[spec][1]
    id_w0, id_w1 = ids_w[spec][0], ids_w[spec][1]
    id_l0, id_l1 = ids_ell[spec][0], ids_ell[spec][1]
    dl = dls[spec]
    ellcamb = dls['ell']

    ax = fig.add_axes((.15, .3, .8, .6))
    id = np.where(ellcamb >= 100)
    ellcamb = ellcamb[id]
    dl = dl[id]
    ax.plot(ellcamb, dl/ellcamb/(ellcamb + 1)*2*np.pi, color = "k", ls = "dotted", lw = 0.7, label = "Best fit from CAMB")
    ax.errorbar(ell_act[:, 2][id_l0:id_l1], X_data[id_d0:id_d1], yerr = err_act[id_d0:id_d1], label = "deep", ls = "None", marker = "o",
            elinewidth = 1.5, markersize = 4)
    ax.errorbar(ell_act[:, 2][id_l0:id_l1], X_data[id_w0:id_w1], yerr = err_act[id_w0:id_w1], label = "wide", ls = "None",
            marker = "o", elinewidth = 1.5, markersize = 4)
    ax.legend()
    ax.set_xticklabels([])
    ax.set_ylabel(r"$C_\ell^{\rm %s}$"%(spec.upper()))
    if scale == "log":
        ax.set_yscale('log')
    ax.set_xlim(100, 4500)

    ax2 = fig.add_axes((.15, .1, .8, .2))
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$\Delta C_\ell^{\rm %s}$"%(spec.upper()))
    ax2.set_xlim(100, 4500)
    ax2.axhline(0, xmin = -100, xmax = 1e4, color = "k", alpha = 0.6)
    ax2.errorbar(ell_act[:, 2][id_l0:id_l1], X_data[id_d0:id_d1] - X_model[id_d0:id_d1], yerr = err_act[id_d0:id_d1], ls = "None", marker = "o",
             elinewidth = 1.5, markersize = 4)
    ax2.errorbar(ell_act[:, 2][id_l0:id_l1], X_data[id_w0:id_w1] - X_model[id_w0:id_w1], yerr = err_act[id_w0:id_w1], ls = "None",
             marker = "o", elinewidth = 1.5, markersize = 4)
    plt.tight_layout()
    plt.show()

plot("tt", {"ell": ellcamb, "tt": dltt, "te": dlte, "ee": dlee}, scale = "nolog")
plot("te", {"ell": ellcamb, "tt": dltt, "te": dlte, "ee": dlee}, scale = "nolog")
plot("ee", {"ell": ellcamb, "tt": dltt, "te": dlte, "ee": dlee}, scale = "log")
