import numpy as np

def bin_spec(x, y, yerr, binning):
    """
    Args:
        x: 1D array - x-values
        y: 1D array - y-values
        yerr: 1D array - errors on y
        binning: 2D array - each line is [lmin, lmax, (lmin + lmax)/2]

    Returns:
        lb: 1D-array - binned x-values
        yb: 1D array - binned y-values
        yerrb: 1D array - binned errors on y
    """
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

def rebin_spec(x, y, cov, n):
    """
    Args:
        x: 1D array - x-values
        y: 1D array - y-values
        cov: 2D array - covariance of y
        n: int - merge bins (x) n by n

    Returns:
        x, y, cov with merged bins
    """
    N = len(y)
    nbins = int(N / n)
    xp = x[:nbins*n]
    yp = y[:nbins*n]
    covp = cov[:nbins*n, :nbins*n]

    P = np.zeros((nbins, len(yp)))
    for i in range(nbins):
        P[i, n*i:n*(i+1)] = np.ones(n)
    invcov = P @ np.linalg.solve(covp, P.T)
    y = np.linalg.solve(invcov, P) @ np.linalg.solve(covp, yp)
    x = np.array([np.mean(xp[i*n:(i+1)*n]) for i in range(nbins)])

    return(x, y, np.linalg.inv(invcov))

def bin_actlike(act_data_dir, Dl, mode, kind, binning, tt_lmax = 5000):
    """
    Args:
        act_data_dir: str - name of ACT data directory
        Dl: 1D array - power spectrum to bin (starts at l=2)
        mode: str - "TT", "EE", "TE"
        kind: str - "deep" or "wide"
        binning: 2D array - each line is [lmin, lmax, (lmin + lmax)/2]
        tt_lmax: int - cut in multipoles

    Returns:
        lb, Db: binned values of multipoles and Dl
    """
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

    bbldeep_file = act_data_dir + "coadd_bpwf_15mJy_191127_lmin2.npz"
    bblwide_file = act_data_dir + "coadd_bpwf_100mJy_191127_lmin2.npz"

    bbl_file = {"wide": bblwide_file,
                "deep": bbldeep_file}

    bbl = np.load(bbl_file[kind])["bpwf"]
    win_func = np.zeros((bmax_win, lmax_win))
    win_func[:bmax_win, 1:lmax_win] = bbl[:bmax_win, :lmax_win]


    l_list = np.arange(2, tt_lmax + 1)

    cl = np.zeros(lmax_win)
    cl[1:tt_lmax] = Dl[:tt_lmax - 1] / l_list / (l_list + 1) * 2 * np.pi

    index_win = {"TT": [2, 3],
                 "TE": [6, 7],
                 "EE": [9, 10]}
    id0, id1 = index_win[mode]
    cl_b = win_func[id0*bmax:id1*bmax, 1:lmax_win]@cl[1:lmax_win]
    Dl_b = binning[:, 2] * (binning[:, 2] + 1) / 2 / np.pi * cl_b

    return(binning[:, 2], Dl_b)

def bin_sptlike(spt_data_dir, Dl, mode, xfreq):
    """
    Args:
        spt_data_dir: str - name of SPT3G data directory
        Dl: 1D array - power spectrum to bin (starts at l=2)
        mode: str - "TT", "EE", "TE"
        xfreq: str - "f1xf2"
        
    Returns:
        lb, Db: binned values of multipoles and Dl
    """
    spt_win_dir = spt_data_dir + "windows/"

    bmin, bmax = 1, 44
    nbins = bmax + 1 - bmin
    win_lmin, win_lmax = 1, 3200
    ells = np.arange(win_lmin, win_lmax + 1)

    windows = np.array(
        [np.loadtxt(
            spt_win_dir + f"window_{i}.txt", unpack = True
            )[1:] for i in range(bmin, bmax + 1)])

    vec_indices = np.arange(12) # take all spectra
    windows = windows[:, vec_indices, :]

    Dl_sptlike = np.empty_like(ells)
    Dl_sptlike[1:] = Dl[:win_lmax - 1]

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

    index = spectra.index(f"{mode} {xfreq}")
    Db = windows[:, index, :] @ Dl_sptlike
    lb = windows[:, index, :] @ ells

    return(lb, Db)
