# -*- coding: utf-8 -*-
"""
This stand-alone file contains functions for calculating drop shapes
using the axisymmetric drop-shape analysis (ADSA) methodology. In
addition, command-like execution mode is provided as well as a
main-function that calculates and plots the drop shape for a given set
of volume and contact angle.
"""
import sys
if sys.version_info.major < 3:
    raise AssertionError("Use Python 3")
import argparse
import csv
import numpy as np
import scipy
from scipy.integrate import odeint, trapz, cumtrapz
from scipy.optimize import minimize, minimize_scalar, least_squares
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
##import seaborn as sns
##sns.set_style("whitegrid")

# Define physical constants
gamma = 72.8e-3  # N/m
rho = 1000.0  # kg/m3
g = 9.81  # m/s2


def yl(y, s, b, c=rho*g/gamma):
    """
    Compute the derivate of the Young-Laplace equation.

    Parameters
    ----------
    y - (phi, x, z) array_like
    s - Integration parameter (unused)
    b - 1/R0, ie. the curvature at top of drop
    c - rho g / gamma, ie. the capillary constant
    
    Returns
    -------
    dy/ds : (dphi/ds, dx/ds, dz/ds) a tuple
    """
 
    # Extract coordinates
    (phi, x, z) = y
 
    # Handle special case of x=0 and phi=0
    if x == 0 and phi == 0:
        dphi_ds = 2*b + c*z
    else:
        dphi_ds = 2*b + c*z - np.sin(phi)/x
    dx_ds = np.cos(phi)
    dz_ds = np.sin(phi)

    return (dphi_ds, dx_ds, dz_ds)


def calc_volume(y):
    """
    Calculate the drop volume from a shape matrix.

    Parameters
    ----------
    y - Shape matrix, ndarray (n, 3)

    Returns
    -------
    Integrated volume using for the given shape matrix.

    """
    return trapz(np.pi*y[:, 1]**2, y[:, 2])


def drop_shape_estimate(R, ca, c=rho*g/gamma):
    """
    Calculate approximate values based on given radius of curvature and contact 
    angle.

    Parameters
    ----------
    R - Radius of curvature at the top of drop
    ca - Contact angle of the drop (in degrees)
    c - (rho*g/gamma) The capillary constant

    Returns
    -------
    (h, R, vol, l_c, ca) - Estimated values
        h - Maximum height of drop
        R - Radius of curvature at the top of drop (not changed)
        vol - Volume of drop
        l_c = Total path length of drop perimeter
        ca - Contact angle of drop (not changed)
    """
    x = np.cos(np.deg2rad(ca))
    # Spherical cap approximation
    vol = np.pi/3.0*R*(x**3-3*x+2)
    # Height from spherical cap approximation
    h = R*(1-x)
    # Path length from spherical cap approximation
    l_c = R*np.deg2rad(ca)
    
    return (h, R, vol, l_c, ca)


def drop_shape_estimate_for_volume(vol, ca, c=rho*g/gamma):
    """
    Calculate approximate values based on given volume and contact angle.

    Parameters
    ----------
    vol - Volume of drop
    ca - Contact angle (in degrees)
    c - (rho*g/gamma) The capillary constant

    Returns
    -------
    (h, R, vol, l_c, ca) - Estimated values
        h - Maximum height of drop
        R - Radius of curvature at the top of drop
        vol - Volume of drop after estimation
        l_c = Total path length of drop perimeter
        ca - Contact angle of drop (not changed)    
    """
    x = np.cos(np.deg2rad(ca))
    # Spherical cap approximation with static R != R(z)
    R = np.power(3 * vol / (np.pi*(x**3-3*x+2)), 1/3)
    # Height of drop from spherical cap approximation
    h = R*(1-x)
    # Path length from spherical cap approximation
    l_c = R*np.deg2rad(ca)
    return (h, R, vol, l_c, ca)


def drop_shape(R0, ca, s=None, volume=None):
    """
    Calculate the drop shape for given R0 and ca.

    Parameters
    ----------
    R0 - Radius of curvature at the top of the drop
    ca - Contact angle (in degrees)
    s - Integration space (set to None for estimation)
    volume - Volume of drop (set to None for estimation)

    Returns
    -------
    y - ndarray, [..., [s, x, z]] 
        2D array containing the path of the drop

    Raises
    ------
    RuntimeError
        If stop condition cannot be reached. Usually the provided path, s, is
        too short when this exception occurs. Try setting the path manually.
    """
    if s is None:
        (h_R, R0, vol_est, lc_R, ca) = drop_shape_estimate(R0, ca)
        if volume is not None:
            # Calculate another estimate
            (h_vol, R0_est, volume, lc_vol,
                ca) = drop_shape_estimate_for_volume(volume, ca)
        else:
            lc_vol = lc_R
        s = np.linspace(0, max(lc_R, lc_vol), 1000)
    y = odeint(yl, [0.0, 0.0, 0.0], s, args=(1/R0,))

    # Find stop condition
    found_end = False
    for imax, yval in enumerate(y):
        if yval[0] >= np.deg2rad(ca):
            found_end = True
            break

    if found_end:
        # Remove excess points
        y = y[:imax]  # Move points to baseline
        y = y - [0, 0, max(y[:, 2])]
    else:
        # s = np.linspace(0, max(s), 100)
        # return drop_shape(R0, ca, s)
        raise RuntimeError("Stop condition not found. Perhaps increase "
                "s_max? Maximum theta={}, while expecting {}".format(
                    np.rad2deg(y[-1][0]), ca))

    return y


def guess_R0(volume, ca):
    """
    Calculate estimate for R0 from volume using the spherical cap approximation.

    Parameters
    ----------
    volume - Volume of drop
    ca - Contact angle (in degrees)

    Returns
    -------
    R0 - Estimate for the radius of curvature at the top of drop
    """
    # Maximum height we allow for a spherical droplet
    h_max = 3.0 * np.sqrt(gamma/(rho*g))
    x = np.cos(np.deg2rad(ca))
    R0 = np.power(3*volume/(np.pi*(x**3-3*x+2)), 1/3)
    h = np.abs(R0*(1-np.cos(np.deg2rad(ca))))  # Height of sphere

    if h > h_max:
        # We use a ellipsoid approximation (scale: h/2R0 = h_max/2c)
        c = R0/h * h_max
        # Equal volumes --> abc = R0**3 --> a = b = sqrt(R0**3/h_max)
        b = R0*np.sqrt(h/h_max)
        # Ellipse radius of curvature at pole
        Re = b**2 / c
        return Re

    return R0


def fmin(V, volume):
    """
    Cost function for the minimization algorithm. 

    Parameters
    ----------
    V - Calculated volume
    volume - Target volume

    Return
    ------
    Cost estimation for the given parameters.

    Notes
    -----
    Tweaking this function might be necessary to provide more robust results.
    """
    return (volume - V)**2/V


def drop_shape_for_volume(volume, ca, xtol=1e-9, method='nelder-mead'):
    """
    Calculate the drop shape for given volume and ca. Since calculating 
    directly from volume is not possible with the current method, this 
    algorithm minimizes the error in the target volume and calculated volume.

    Stop condition, xtol, should in general be adequate, but might require
    tweaking if the algorithm fails to converge.

    Parameters
    ----------
    volume - Target volume
    ca - Target contact angle (in radians)
    xtol - Tolerance for the stop condition in the scipy.optimize.minimize
           function.
    method - Minimization method for the scipy.optimize.minimize function.

    Returns
    -------
    y - ndarray, [..., [s, x, z]] 
        2D array containing the path of the drop    
    """
    # Make a guess for the radius of curvature
    R0 = guess_R0(volume, ca)

    # Minimize error
    res = minimize(lambda x: fmin(calc_volume(drop_shape(
        x, ca, volume=volume)), volume), R0, method=method, 
        options={'xtol': xtol})

    # Calculate resulting shape
    R0 = res.x[0]
    y = drop_shape(R0, ca)

    return y


def plot_drop(y, color='k', label=None, annotate=True, ax=None, autoscale=True):
    """
    Plot the given drop shape.

    Parameters
    ----------
    y - 2D array of (s, x, z) elements
    color - Color of the lines
    label - Label for the plotted line (set to None for empty)
    ax - matplotlib.pyplot.Axes instance (set to None for new axis)
    autoscale - Scale according to drop height

    Notes
    -----
    Pauses execution to show the plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    x_data = list(-y[:, 1][::-1]) + list(y[:, 1])
    y_data = list(-y[:, 2][::-1]) + list(-y[:, 2])

    ax.plot(x_data, y_data, ls='-', marker='o', color=color,
            mec=color, mfc='w', mew=1.0, label=label)
    ax.axhline(0, ls='--', color='k')
    if autoscale:
        ax.set_ylim(min(y_data), 1.1*max(y_data))
    ax.set_aspect('equal')
    ax.set_xlabel("mm")
    ax.set_xticklabels(1000*ax.get_xticks())
    ax.set_ylabel("mm")
    ax.set_yticklabels(1000*ax.get_yticks())

    volume = calc_volume(y)

    if annotate:
        ax.text(0.5, 0.5, "Contact angle: {:.4}\nVolume: {:.4} uL".format(np.rad2deg(
            y[-1, 0]), volume*1e9), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.show()


def main(volume, ca):
    """
    Run an example calculation using the provided volume and contact angle.

    Parameters
    ----------
    volume - Target volume (in litres)
    ca - Contact angle (in degrees)
    """

    print("Solving for volume {:.3} uL, and ca {}°".format(volume*1e9, ca))
    y = drop_shape_for_volume(volume, ca)
    
    true_volume=calc_volume(y)
    print("Volume {:.3} uL".format(true_volume*1e9))
    
    # Plot data
    plot_drop(y)
    
    # Save data to excel/csv
    with open('output.csv', 'w') as fl:
        writer=csv.writer(fl)
        writer.writerow(['s', 'x', 'y'])  
        for values in y:
            writer.writerow(values)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", help="volume of drop in uL", type=float)
    parser.add_argument("contact_angle", help="contact angle in degrees", type=float)

    args = parser.parse_args()

    main(volume=1e-9*args.volume, ca=args.contact_angle)
