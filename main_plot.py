import load_tools
import bin_tools
import tools
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./custom.mplstyle")

#####################
## Plot parameters ##
#####################
bf_kind = "ACT_only"
mode = "EE"
lims = [50, 1200]

save_plots = False
plot_dir = "plots/"

exp = "Planck"
#exp = "SPT"

xfreq = "143x143"
#xfreq = "150x150"

act_pol_eff = {("ACT_only", "LCDM"): 1.000934985,
               ("ACT_only", "EDE"): 0.9951140073,
               ("ACT+WMAP", "LCDM"): 1.001913951,
               ("ACT+WMAP", "EDE"): 1.005842680}
if mode == "EE":
    pol_eff = act_pol_eff[bf_kind, "LCDM"] ** 2
elif mode == "TE":
    pol_eff = act_pol_eff[bf_kind, "LCDM"]

if exp == "Planck":
    spec = f"{mode}_{xfreq}"
elif exp == "SPT":
    spec = f"{mode} {xfreq}"

if save_plots:
    import os
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
#####################
#####################
#####################

data_dir = "data/"
act_bf_dir = data_dir + "ACT_bestfits/"
planck_data_dir = data_dir + "Planck_data/"
planck_bf_dir = data_dir + "PLA/"
act_data_dir = data_dir + "ACT_data/"
spt_data_dir = data_dir + "SPT3G_data/"

binning_act = np.loadtxt(act_data_dir + "Binning.dat")
binning_planck = np.loadtxt(planck_data_dir + "bin_planck.dat")

planck_data = load_tools.load_planck_data(planck_data_dir)
act_bf = load_tools.load_act_best_fits(act_bf_dir)
act_data = load_tools.load_act_data(act_data_dir)
spt_data = load_tools.load_spt_data(spt_data_dir)
planck_bf = load_tools.load_planck_best_fit_PLA(planck_bf_dir)

#### Plot ####
fig = plt.figure(figsize = (12, 8))
ax = fig.add_axes((.15, .3, .8, .6))

### ACT best fits ###

### CDM ###
x_cdm, y_cdm = tools.cut_spec(lims, act_bf[bf_kind, "LCDM"]["ell"],
                              act_bf[bf_kind, "LCDM"][mode])
ax.plot(x_cdm, y_cdm, color = "k", lw = 0.7,
        label = bf_kind + " [LCDM]")
###########

### EDE ###
x_ede, y_ede = tools.cut_spec(lims, act_bf[bf_kind, "EDE"]["ell"],
                              act_bf[bf_kind, "EDE"][mode])
ax.plot(x_ede, y_ede, color = "k", lw = 0.7,
        ls = "dotted", label = bf_kind + " [EDE]")
###########

### ACT data ###

### Deep ###
xd, yd, covd = (act_data["deep", mode]["ell"],
               act_data["deep", mode]["Dl"],
               act_data["deep", mode]["cov"])
yerrd = np.sqrt(covd.diagonal())
xd, yd, yerrd = tools.cut_spec(lims, xd, yd, yerrd)
ax.errorbar(xd, yd, yerr = yerrd, ls = "None",
            marker = ".", capsize = 1., elinewidth = 1.,
            label = "ACT deep")
############

### Wide ###
xw, yw, covw = (act_data["wide", mode]["ell"],
               act_data["wide", mode]["Dl"],
               act_data["wide", mode]["cov"])
yerrw = np.sqrt(covw.diagonal())
xw, yw, yerrw = tools.cut_spec(lims, xw, yw, yerrw)
ax.errorbar(xw, yw, yerr = yerrw, ls = "None",
            marker = ".", capsize = 1., elinewidth = 1.,
            label = "ACT wide")
############

### Planck ###
if exp == "Planck":
    xp, yp, ycovp = (planck_data[spec]["ell"],
                     planck_data[spec]["Dl"],
                     planck_data[spec]["cov"])

    n=4 ## large binning for planck
    xp, yp, ycovp = bin_tools.rebin_spec(xp, yp, ycovp, n)

### SPT ###
elif exp == "SPT":
    xp, yp, ycovp = (spt_data[spec]["ell"],
                     spt_data[spec]["Dl"],
                     spt_data[spec]["cov"])

yerrp = np.sqrt(ycovp.diagonal())
xp, yp, yerrp = tools.cut_spec(lims, xp, yp, yerrp)
ax.errorbar(xp, yp, yerrp, ls = "None",
            marker = ".", capsize = 1.,
            elinewidth = 1., label = f"{exp} {spec}")
##############

ax.legend()
ax.tick_params(labelbottom=False)
ax.set_xlim(left = 200)
ax.set_ylabel(r"$D_\ell^{\mathrm{%s}}$" % mode)

### Residual plot ###
ax2 = fig.add_axes((.15, .1, .8, .2), sharex = ax)
ax2.set_xlabel(r"$\ell$")
ax2.axhline(0, xmin = -100, xmax = 1e4, color = "k", ls = "--", lw = 1.)

### Bestfit difference ###
ax2.plot(x_ede, y_ede - y_cdm, color = "k",
         lw = 0.8, label = "ACT EDE - ACT LCDM")

if exp == "Planck":
    ax2.plot(x_ede, planck_bf["EE"][lims[0]-2:lims[1]-1] - y_cdm,
             lw = 0.8, ls = "dotted", color = "k",
             label = "Planck LCDM - ACT LCDM")
    ax2.legend(fontsize = 10)
##########################

### Residual LCDM ###
lb_cdm, yb_cdm = bin_tools.bin_actlike(act_data_dir,
                                       act_bf[bf_kind, "LCDM"][mode],
                                       mode, "deep", binning_act)
lb_cdm, yb_cdm = tools.cut_spec(lims, lb_cdm, yb_cdm)

### Deep ###
ax2.errorbar(lb_cdm, yd / pol_eff - yb_cdm , yerr = yerrd, ls = "None",
             marker = ".", capsize = 1., elinewidth = 1.)
############

### Wide ###
ax2.errorbar(lb_cdm, yw / pol_eff - yb_cdm, yerr = yerrw, ls = "None",
             marker = ".", capsize = 1., elinewidth = 1.)
############

### Planck ###
if exp == "Planck":
    bin_pk = binning_planck[
        np.argwhere(binning_planck[:, 2] == planck_data[spec]["ell"][0])[0,0]:]
    rebin_min_planck = np.array([bin_pk[i*n,0] for i in range(len(bin_pk)//n)])
    rebin_max_planck = np.array([bin_pk[(i+1)*n - 1, 1] for i in range(len(bin_pk)//n)])
    rebin_mean_planck = (rebin_min_planck + rebin_max_planck) / 2
    rebin_planck = np.array([rebin_min_planck, rebin_max_planck, rebin_mean_planck]).T

    lb_cdm, yb_cdm, _ = bin_tools.bin_spec(act_bf[bf_kind, "LCDM"]["ell"],
                                           act_bf[bf_kind, "LCDM"][mode],
                                           act_bf[bf_kind, "LCDM"]["ell"],
                                           rebin_planck)

if exp == "SPT":
    lb_cdm, yb_cdm = bin_tools.bin_sptlike(spt_data_dir,
                                           act_bf[bf_kind, "LCDM"][mode],
                                           mode, xfreq)

lb_cdm, yb_cdm = tools.cut_spec(lims, lb_cdm, yb_cdm)
lb_cdm, yb_cdm = lb_cdm[:len(yp)], yb_cdm[:len(yp)]

ax2.errorbar(lb_cdm, yp - yb_cdm, yerr = yerrp, ls = "None",
             marker = ".", capsize = 1., elinewidth = 1.)
##############

ax2.set_ylabel(r"$\Delta D_\ell^{%s}}$" % mode)
ax2.set_ylim(-3, 3)
ax2.set_xlim(left = 200)
if save_plots:
    plt.savefig(plot_dir + f"{bf_kind}_{mode}_{exp}_{xfreq}.png", dpi = 300)
plt.show()
