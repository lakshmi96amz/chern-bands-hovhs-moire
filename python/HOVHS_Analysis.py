#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def get_band_energy(nx, ny, kx, ky, hz, V, band_index):
    """
    Get the energy of a specific band at given k-point.

    Parameters
    ----------
    band_index : int
        Index of the band (0 = lowest energy band)

    Returns
    -------
    energy : float
        Energy eigenvalue of the specified band
    wavefunction : ndarray
        Corresponding eigenvector
    """
    H = build_hamiltonian(nx, ny, kx, ky, hz, V)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[band_index]

def compute_gradient(nx, ny, kx, ky, hz, V, band_index, dk=1e-5):
    """
    Compute the gradient of band energy with respect to kx and ky using finite differences.

    Parameters
    ----------
    dk : float
        Small increment for finite difference calculation

    Returns
    -------
    grad : ndarray
        Gradient vector [dE/dkx, dE/dky]
    """

    # Energy at the point
    E0 = get_band_energy(nx, ny, kx, ky, hz, V, band_index)

    # Forward differences
    E_kx_plus = get_band_energy(nx, ny, kx + dk, ky, hz, V, band_index)
    E_ky_plus = get_band_energy(nx, ny, kx, ky + dk, hz, V, band_index)

    # Compute derivatives
    dE_dkx = (E_kx_plus - E0) / dk
    dE_dky = (E_ky_plus - E0) / dk

    grad = np.array([dE_dkx, dE_dky])

    return grad


def compute_hessian(nx, ny, kx, ky, hz, V, band_index, dk=1e-5):
    r"""
    Compute the determinant of the Hessian matrix of band energy with respect to kx and ky using finite differences.

    Parameters
    ----------
    dk : float
        Small increment for finite difference calculation

    Returns
    -------
    hessian : float
        determinant of Hessian matrix, d^2E/dkx^2 d^2E/dky^2 - (d^2E/dkxdky)^2
    """

    # Energies at neighboring points
    E0 = get_band_energy(nx, ny, kx, ky, hz, V, band_index)

    E_kx_plus = get_band_energy(nx, ny, kx + dk, ky, hz, V, band_index)
    E_kx_minus = get_band_energy(nx, ny, kx - dk, ky, hz, V, band_index)

    E_ky_plus = get_band_energy(nx, ny, kx, ky + dk, hz, V, band_index)
    E_ky_minus = get_band_energy(nx, ny, kx, ky - dk, hz, V, band_index)

    E_kx_plus_ky_plus = get_band_energy(nx, ny, kx + dk, ky + dk, hz, V, band_index)
    E_kx_plus_ky_minus = get_band_energy(nx, ny, kx + dk, ky - dk, hz, V, band_index)
    E_kx_minus_ky_plus = get_band_energy(nx, ny, kx - dk, ky + dk, hz, V, band_index)
    E_kx_minus_ky_minus = get_band_energy(nx, ny, kx - dk, ky - dk, hz, V, band_index)

    # Second derivatives using central differences
    d2E_dkx2 = (E_kx_plus - 2*E0 + E_kx_minus) / (dk**2)
    d2E_dky2 = (E_ky_plus - 2*E0 + E_ky_minus) / (dk**2)

    # Mixed derivative
    d2E_dkxdky = (E_kx_plus_ky_plus - E_kx_plus_ky_minus - E_kx_minus_ky_plus + E_kx_minus_ky_minus) / (4 * dk**2)

    hessian = d2E_dkx2 * d2E_dky2 - (d2E_dkxdky)**2

    return hessian


# In[ ]:


import matplotlib.pyplot as plt
from scipy.optimize import brentq


def get_Vcritical(nx, ny, hz, band_index, V_min=1.3, V_max=1.6, tol=1e-6,
                  NV=101, dk=1e-4):
    """
    Compute the critical value of V for a given hz that corresponds to the
    determinant of the Hessian vanishing at the K valley.

    Parameters
    ----------
    V_min, V_max : float
        Search range for V (default: 1.3 to 1.6).
    tol : float
        Tolerance on V for root finding (default: 1e-6).
    NV : int
        Number of V values in the coarse sign-change scan (default: 101).
    dk : float
        Step size for the finite-difference Hessian (default: 1e-4).

    Returns
    -------
    V_crit : float
        Critical value of V where det(Hessian) = 0 at the K valley for the
        given hz, or np.nan if no root is found in [V_min, V_max].

    Notes
    -----
    The K valley point for the triangular lattice is at k = (4π/3a, 0).
    This function finds V such that the band structure has a higher-order
    saddle point at the K valley.
    """
    kx_K, ky_K = 4*np.pi/3, 0.0  # k at the K valley

    def det_hess(V):
        return compute_hessian(nx, ny, kx_K, ky_K, hz, V, band_index, dk=dk)

    # Coarse scan over [V_min, V_max] to find where det(Hessian) changes sign
    V_list = np.linspace(V_min, V_max, NV)
    d = np.array([det_hess(V) for V in V_list])
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]

    if len(idx) == 0:
        return np.nan

    # Refine the first sign change with a root finder
    i = idx[0]
    return brentq(det_hess, V_list[i], V_list[i + 1], xtol=tol)


def get_hovhs(nx, ny, band_index, V_min=1.3, V_max=1.6, hz_min=0.0, hz_max=1.0,
              Nhz=101, tol=1e-6):
    """
    Compute the (hz, V) values that correspond to a higher-order Van Hove
    singularity around the K valley point for the band given by band_index.

    Parameters
    ----------
    V_min, V_max : float
        Search range for V (default: 1.3 to 1.6).
    hz_min, hz_max : float
        Range of hz values (default: 0.0 to 1.0).
    Nhz : int
        Number of hz values in the range (default: 101).
    tol : float
        Tolerance on V for root finding (default: 1e-6).

    Returns
    -------
    hz_list : ndarray
        hz values
    Vc_list : ndarray
        Critical V values that correspond to the HOVHS
    """
    # List of hz values within the given range
    hz_list = np.linspace(hz_min, hz_max, Nhz)

    # List of critical V values
    Vc_list = np.array([
        get_Vcritical(nx, ny, hz, band_index, V_min=V_min, V_max=V_max, tol=tol)
        for hz in hz_list
    ])

    return hz_list, Vc_list


def plot_hovhs(hz_list, Vc_list):
    """
    Plot the HOVHS line in the (hz, V) phase space.

    Parameters
    ----------
    hz_list : array_like
        hz values
    Vc_list : array_like
        Critical V values that correspond to the HOVHS
    """
    plt.plot(hz_list, Vc_list, 'k')

    plt.xlabel(r"$h_z \,/\, (v_F/a)$")
    plt.ylabel(r"$V \,/\, (v_F/a)$")

    plt.show()


# In[ ]:





# In[ ]:




