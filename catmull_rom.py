import numpy as np

def catrom(t, y):
    """
    Perform Catmull-Rom interpolation on a regularly sampled series.

    Parameters
    ----------
    t : numpy.ndarray
        The abscissae at which to evaluate the interpolant.
    y : numpy.ndarray
        The function f(t') sampled at integer values t', such that
        y[0] = f(0), y(1) = f(1), etc.

    Returns
    -------
    yinterp : numpy.ndarray
        The interpolated function, f(t)
    """

    #Decimal and Integer Parts of t
    t_dec, t_low = np.modf(t)
    t_low = t_low.astype(int)
    N = len(y)

    #Bounds for Left Case, Middle Case, Right Case
    left = t_low < 1
    middle = (t_low >= 1) & (t_low <= N - 3)
    right = t_low > N - 3

    yinterp = np.empty_like(t, dtype=float)

    #Left Case
    t_il = t_low[left]
    t_dl = t_dec[left]
    m1 = (y[t_il + 1] - y[t_il])
    m2 = (y[t_il + 2] - y[t_il]) / 2 
    yinterp[left] = (
        (2*t_dl**3 - 3*t_dl**2 + 1) * y[t_il]
        + (t_dl**3 - 2*t_dl**2 + t_dl) * m1
        + (-2*t_dl**3 + 3*t_dl**2) * y[t_il + 1]
        + (t_dl**3 - t_dl**2) * m2
    )

    #Middle Case --> Traditional Catmull-Rom Formula
    t_im = t_low[middle]
    t_dm = t_dec[middle]
    m1 = (y[t_im + 1] - y[t_im - 1]) / 2
    m2 = (y[t_im + 2] - y[t_im]) / 2

    yinterp[middle] = (
        (2*t_dm**3 - 3*t_dm**2 + 1) * y[t_im]
        + (t_dm**3 - 2*t_dm**2 + t_dm) * m1
        + (-2*t_dm**3 + 3*t_dm**2) * y[t_im + 1]
        + (t_dm**3 - t_dm**2) * m2
    )

    #Right Case
    t_ir = t_low[right]
    t_dr = t_dec[right]
    m1 = (y[t_ir + 1] - y[t_ir - 1])/2
    m2 = (y[t_ir + 1] - y[t_ir]) 
    yinterp[right] = (
        (2*t_dr**3 - 3*t_dr**2 + 1) * y[t_ir]
        + (t_dr**3 - 2*t_dr**2 + t_dr) * m1
        + (-2*t_dr**3 + 3*t_dr**2) * y[t_ir + 1]
        + (t_dr**3 - t_dr**2) * m2
    )

    #For When Point to Interpolate Over is Given On Array
    given = (t_dec == 0)
    yinterp[given] = (y[t_low[given]])
    
    return yinterp 
