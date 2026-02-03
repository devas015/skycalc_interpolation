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

    t1, k = np.modf(t)
    k = k.astype(int)
    k = k.flatten()
    y = y.flatten()
    N = len(y)

    left = k < 1
    middle = (k >= 1) & (k <= N - 3)
    right = k > N - 3

    yinterp = np.empty_like(t, dtype=float)

    a = -0.25*y[0] + 0.5*y[1] - 0.25*y[2]
    b = y[0] - 2*y[1] + y[2]
    c = -1.75*y[0] + 2.5*y[1] - 0.75*y[2]
    d = y[0]
    yinterp[left] = a*t1[left]**3 + b*t1[left]**2 + c*t1[left] + d

    km = k[middle]
    tm = t1[middle]
    m1 = (y[km + 1] - y[km - 1]) / 2
    m2 = (y[km + 2] - y[km]) / 2

    yinterp[middle] = (
        (2*tm**3 - 3*tm**2 + 1) * y[km]
        + (tm**3 - 2*tm**2 + tm) * m1
        + (-2*tm**3 + 3*tm**2) * y[km + 1]
        + (tm**3 - tm**2) * m2
    )

    tr = t1[right]
    #Check if right edge solution is correct
    a =  0.25*y[-1] - 0.5*y[-2] + 0.25*y[-3]
    b =  y[-1] - 2*y[-2] + y[-3]
    c =  1.75*y[-1] - 2.5*y[-2] + 0.75*y[-3]
    d =  y[-1]
    yinterp[right] = (
        a*tr**3 +
        b*tr**2 +
        c*tr +
        d
    )
    return yinterp 


if __name__ == '__main__':
    x = np.arange(0, 100)

    def f(t):
        return t**3 + 2* t**2 

    y = f(x)

    t = np.asarray([0.8, 0.9, 1.2, 3.5, 4.7, 5.9, 96.5, 99.1])
    yinterp = catrom(t, y)

    print(f'f({t}) == {f(t)}')
    print(f'f({t}) ~= {yinterp}')
