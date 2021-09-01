import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

### rcParams ###
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.axis'] = 'both'
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.linestyle'] = "dotted"
matplotlib.rcParams['grid.linewidth'] = 0.8
matplotlib.rcParams['grid.alpha'] = 0.5
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['xtick.major.pad'] = 6
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['ytick.major.pad'] = 6
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['legend.frameon'] = False



def get_planck_dict(freq, modes):
    ids = [0]
    names = []
    file = "spectrum_%s_%dx%d.dat"

    for mode in modes:
        for f1 in freq:
            for f2 in freq:
                if f2 < f1:
                    continue
                if os.path.exists(file%(mode, f1,f2)):
                    names.append("%s_%dx%d"%(mode,f1,f2))
                    spec = np.loadtxt(file%(mode,f1,f2))
                    ids.append(ids[-1] + len(spec))
    print(ids)
    invcovmat = np.load("covmat.npy")
    covmat = np.linalg.inv(invcovmat)
    outputs = {}

    for i, name in enumerate(names):
        spec = np.loadtxt("spectrum_%s.dat"%name)
        pr = spec[:, 0] * (spec[:, 0] + 1) / 2 / np.pi
        outputs[name] = {"ell": spec[:, 0],
                        "Dl": spec[:, 1] * pr,
                        "cov": covmat[ids[i]:ids[i+1],
                                      ids[i]:ids[i+1]]*np.outer(pr,pr)}
    return(outputs)

def bin_spec(x, y, yerr, binning):

    lowlims = binning[:, 0]
    highlims = binning[:, 1]
    Nb = len(highlims)
    binsize = [highlims[i] + 1 - lowlims[i] for i in range(Nb)]

    yb = [np.mean(y[((x >= lowlims[i]) & (x <= highlims[i]))]) for i in range(Nb)]
    covb = [
        1/binsize[i]*np.mean(
            yerr[((x >= lowlims[i]) & (x <= highlims[i]))]**2
            ) for i in range(Nb)]

    yb = np.array(yb)
    yerrb = np.array(np.sqrt(covb))
    lb = binning[:, 2]

    return(lb, yb, yerrb)

def plot_res(xbf, ybf, ybfb, x, y, ycov, lab_BF, lab_data, mode):

    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_axes((.15, .3, .8, .6))
    ax.plot(xbf, ybf, color = "k", lw = 0.8,
            label = lab_BF)
    yerr = np.sqrt(ycov.diagonal())
    ax.errorbar(x, y, yerr=yerr, color = "tab:red", ls = "None",
                marker = ".", capsize = 1.0, elinewidth = 1.0,
                label = lab_data)
    ax.legend()
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"$D_\ell^{\mathrm{%s}}$" % mode)


    ax2 = fig.add_axes((.15, .1, .8, .2), sharex = ax)
    ax2.set_xlabel(r"$\ell$")
    ax2.axhline(0, xmin = -100, xmax = 1e4, color = "k")

    delta = y - ybfb
    print("chi2 %s - %s - %s : " % (lab_data, lab_BF, mode),
          delta @ np.linalg.solve(ycov, delta),
          " / %d" % len(delta[~np.isnan(delta)]))

    ax2.errorbar(x, delta, yerr = yerr, ls = "None", color = "tab:red",
                     marker = ".", elinewidth = 1.0, capsize = 0.8)
    ax2.set_ylabel(r"$\Delta D_\ell^{%s}}$" % mode)
    ax2.set_ylim(-3, 3)
    #plt.savefig(plot_dir + "%s_%s_%s.png" % (lab_BF, lab_data, mode), dpi=300)
    plt.show()

### ACT EDE best-fits ###
act_dir = "/home/laposta-l/Desktop/residual_ede_act/ACT_EDE/"
yp = {"LCDM": 1.000934985,
      "EDE": 9.951140073e-01,
      "LCDM-WMAP": 1.001913951,
      "EDE-WMAP": 1.005842680}
def load_act_best_fits(act_dir):

    spec_file = act_dir + "ACT_%s_bf_%s_ThibautAdrien_26Aug2021.txt"
    kind = {"ACT": "only",
            "ACT+WMAP": "P18TTlmax650"}
    model = ["LCDM", "EDE"]

    outputs = {}
    for k in kind:
        for m in model:
            ell, TT, TE, EE = np.loadtxt(spec_file%(kind[k], m)).T
            outputs[k, m] = {"ell": ell,
                             "TT": TT,
                             "TE": TE,
                             "EE": EE}
    return(outputs)

best_fits = load_act_best_fits(act_dir)

kind = "ACT+WMAP"
model = "EDE"
ell_ACT, DlEE_ACT = best_fits[kind, model]["ell"], best_fits[kind, model]["EE"]


binning_file = "../bin_planck.dat"
binning = np.loadtxt(binning_file)
lb, yEE_ACT, _ = bin_spec(ell_ACT, DlEE_ACT, ell_ACT, binning)
outputs = get_planck_dict([100, 143, 217], ["TT", "EE", "TE"])

spec = "EE_143x143"

x = outputs[spec]["ell"]
id = np.where((x >= 30) & (x <= 1500))
y = outputs[spec]["Dl"][id]
ycov = outputs[spec]["cov"][np.ix_(id[0], id[0])]
x = x[id]

id_BF = np.where((lb >= 30) & (lb <= 1500))
lb = lb[id_BF]
yEE_ACT = yEE_ACT[id_BF]

id_BF = np.where((ell_ACT >= 30) & (ell_ACT <= 1500))
ell_ACT = ell_ACT[id_BF]
DlEE_ACT = DlEE_ACT[id_BF]

plot_res(ell_ACT, DlEE_ACT, yEE_ACT, x, y, ycov, "%s [%s]"%(kind, model), "Planck EE 143x143", "EE")
