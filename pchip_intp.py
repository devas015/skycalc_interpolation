import numpy as np
from scipy.interpolate import PchipInterpolator
import astropy
from astropy.io import fits
import matplotlib.pyplot as plt

#Grid Values
pwvs = np.array([0.05,0.10,0.25,0.5,1.0,1.5,2.5,3.5,5.0,7.5,10.0,20.0,30.0])
moon_sun_sep = np.arange(37) * 5
moon_target_sep = np.arange(37) * 5
alts = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

#Loading Array and Actual Numbers
brightness_array = np.load('../SkyCalcFiles/Updated_SkyCalc_Array.npy', mmap_mode='r')
with astropy.io.fits.open("sky_model_pwv0.05_phase159_sep18_alt47.fits") as data:
    wavelengths = data[1].data["LAM"].astype(np.float64)
    actual = data[1].data["FLUX"].astype(np.float64)

#PCHIP Function (assuming PWV is on grid)
def pchip_intp(
    moon_sun_sep,
    moon_target_sep,
    alts,
    brightness_array,
    pwv_index,
    moon_sun0,
    moon_target0,
    alt0,
):
    #Slice of Array at PWV
    grid = brightness_array[:, pwv_index, :, :, :]  # shape (Nλ, Nmoon_sun, Nmoon_target, Nalt)

    #Interpolate in order of phase, separation, altitude
    grid = PchipInterpolator(moon_sun_sep, grid, axis=1)(moon_sun0)
    grid = PchipInterpolator(moon_target_sep, grid, axis=1)(moon_target0)
    grid = PchipInterpolator(alts, grid, axis=1)(alt0)
    return grid

#Test Case
pwv_index = np.where(pwvs == 0.05)[0][0]
sep1 = 159
sep2 = 18
alt = 47

# Interpolate spectrum at this geometry
pchip_pred = pchip_intp(
    moon_sun_sep=moon_sun_sep,
    moon_target_sep=moon_target_sep,
    alts=alts,
    brightness_array=brightness_array,
    pwv_index=pwv_index,
    #test case
    moon_sun0=sep1,
    moon_target0=sep2,
    alt0=alt,
)
