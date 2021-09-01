import numpy as np
import os
from scipy.io import FortranFile
import pandas as pd

def load_planck_data(planck_data_dir):
    """
    Args:
        planck_data_dir: str - name of planck data directory

    Returns:
        outputs: dict - contains the planck multipoles,
                        power spectra, and covariances
                        for each cross frequency
    """
    file = planck_data_dir + "spectrum_%s_%dx%d.dat"

    ids = [0]
    names = []
    freqs = [100, 143, 217]

    for mode in ["TT", "EE", "TE"]:
        for f1 in freqs:
            for f2 in freqs:
                if f2 < f1: continue
                if os.path.exists(file % (mode, f1, f2)):

                    names.append("%s_%dx%d" % (mode, f1, f2))
                    spec = np.loadtxt(file % (mode, f1, f2))
                    ids.append(ids[-1] + len(spec))

    invcovmat = np.load(planck_data_dir + "covmat.npy")
    covmat = np.linalg.inv(invcovmat)

    outputs = {}

    for i, name in enumerate(names):
        mode, f1xf2 = name.split("_")
        f1, f2 = f1xf2.split("x")
        f1, f2 = int(f1), int(f2)
        spec = np.loadtxt(file % (mode, f1, f2))
        pr = spec[:, 0] * (spec[:, 0] + 1) / 2 / np.pi
        outputs[name] = {"ell": spec[:, 0],
                         "Dl": spec[:, 1] * pr,
                         "cov": covmat[ids[i]:ids[i+1],
                                       ids[i]:ids[i+1]] * np.outer(pr, pr)}
    return(outputs)

def load_planck_best_fit_PLA(PLA_dir):
    """
    Args:
        PLA_dir: str - name of the PLA directory

    Returns:
        outputs: dict - contains the PLA multipoles and
                        best fit power spectra
    """
    PLA_bf_file = PLA_dir + ("COM_PowerSpect_CMB-base-plikHM-"
                             "TTTEEE-lowl-lowE-lensing-"
                             "minimum-theory_R3.01.txt")
    PLA_bf = pd.read_csv(PLA_bf_file, delim_whitespace = True,
                         index_col = 0)

    ell = PLA_bf.index
    cal2 = (1.000442)**2  ###calibration from the PLA
    TT = PLA_bf.TT / cal2
    TE = PLA_bf.TE / cal2
    EE = PLA_bf.EE / cal2

    outputs = {"ell": ell,
               "TT": TT,
               "EE": EE,
               "TE": TE}

    return(outputs)

def load_act_best_fits(act_dir):
    """
    Args:
        act_dir: str - name of act best-fits directory

    Returns:
        outputs: dict - contains the ACT multipoles and
                        power spectra for each kind of
                        best-fit
    """
    spec_file = act_dir + "ACT_%s_bf_%s_26Aug2021.txt"
    kind = {"ACT_only": "only",
            "ACT+WMAP": "P18TTlmax650"}
    model = ["LCDM", "EDE"]

    outputs = {}
    for k in kind:
        for m in model:
            ell, TT, TE, EE = np.loadtxt(spec_file % (kind[k], m)).T
            outputs[k, m] = {"ell": ell,
                             "TT": TT,
                             "TE": TE,
                             "EE": EE}
    return(outputs)

def load_act_data(act_data_dir):
    """
    Args:
        act_data_dir: str - name of ACT data directory

    Returns:
        outputs: dict - contains the ACT multipoles,
                        power spectra, and covariances
                        for deep and wide patches
    """
    nbin = 260

    like_file = act_data_dir + "cl_cmb_ap.dat"
    cov_file = act_data_dir + "c_matrix_ap.dat"
    bin_file = act_data_dir + "Binning.dat"

    ell = np.loadtxt(bin_file)[:, 2]

    bval, X_data, X_sig = np.genfromtxt(like_file, max_rows = nbin,
                                        delimiter = None, unpack = True)
    f = FortranFile(cov_file, "r")
    cov = f.read_reals(dtype = float).reshape((nbin,nbin))
    for i_index in range(nbin):
        for j_index in range(i_index, nbin):
            cov[i_index, j_index] = cov[j_index, i_index]

    ids = {"deep": {"TT": [0, 40],
                    "TE": [40, 85],
                    "EE": [85, 130]},
           "wide": {"TT": [130, 170],
                    "TE": [170, 215],
                    "EE": [215, 260]}}

    ids_ell = {"TT": [5, 45],
               "TE": [0, 45],
               "EE": [0, 45]}

    outputs = {}

    for s in ["deep", "wide"]:
        for mode in ["TT", "TE", "EE"]:
            id0, id1 = ids[s][mode][0], ids[s][mode][1]
            lb = ell[ids_ell[mode][0]:ids_ell[mode][1]]
            pref = lb * (lb + 1) / (2 * np.pi)
            Dl = X_data[id0:id1] * pref
            covDl = cov[id0:id1,
                        id0:id1] * np.outer(pref, pref)

            outputs[s, mode] = {"ell": lb,
                                "Dl": Dl,
                                "cov": covDl}

    return(outputs)

def load_spt_data(spt_data_dir):
    """
    Args:
        spt_data_dir: str - name of SPT3G data directory

    Returns:
        outputs: dict - contains the SPT3G multipoles,
                        power spectra, and covariances
                        for each cross frequency
    """
    spt_bandpowers = spt_data_dir + "SPT3G_Y1_EETE_bandpowers.dat"
    spt_cov = spt_data_dir + "SPT3G_Y1_EETE_covariance.dat"
    spt_win_dir = spt_data_dir + "windows/"

    bmin, bmax = 1, 44
    nbins = bmax + 1 - bmin
    win_lmin, win_lmax = 1, 3200
    ells = np.arange(win_lmin, win_lmax + 1)
    bp = np.loadtxt(spt_bandpowers, unpack=True)[1:]
    cov = np.loadtxt(spt_cov)
    windows = np.array(
        [np.loadtxt(
            spt_win_dir + f"window_{i}.txt", unpack = True
            )[1:] for i in range(bmin, bmax + 1)])

    vec_indices = np.arange(12) # take all spectra
    bp = bp[vec_indices].flatten()
    windows = windows[:, vec_indices, :]
    cov_indices = np.array(
        [np.arange(i * nbins, (i+1) * nbins) for i in vec_indices]
                          )
    cov_indices = cov_indices.flatten()
    cov = cov[np.ix_(cov_indices, cov_indices)]
    lbs = [windows[:, i, :] @ ells for i in vec_indices]

    spectra = ["EE 90x90",
               "TE 90x90",
               "EE 90x150",
               "TE 90x150",
               "EE 90x220",
               "TE 90x220",
               "EE 150x150",
               "TE 150x150",
               "EE 150x220",
               "TE 150x220",
               "EE 220x220",
               "TE 220x220"]

    outputs = {}
    for i, spec in enumerate(spectra):
        vmin = i * int(len(bp) / len(spectra))
        vmax = (i + 1) * int(len(bp) / len(spectra))
        outputs[spec] = {"ell": lbs[i],
                         "Dl": bp[vmin: vmax],
                         "cov": cov[vmin:vmax,
                                    vmin:vmax]}
    return(outputs)
