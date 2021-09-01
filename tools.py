import numpy as np

def cut_spec(lims, x, y, yerr = None):
    """
    Args:
        lims: list containing min and max value of x
        x: 1D array for x-values
        y: 1D array for y-values
        yerr (optional): 1D array for errors on y

    Returns:
        x_o: 1D array cutted x-values
        y_o: 1D array cutted y-values
        yerr_o (optional): 1D array cutted errors on y
    """
    id = np.where((x>=lims[0])&(x<=lims[1]))
    x_o = x[id]
    y_o = y[id]
    if yerr is not None:
        yerr_o = yerr[id]
        return(x_o, y_o, yerr_o)
    
    return(x_o, y_o)
