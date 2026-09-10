#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def compute_dos(nx, ny, hz, V, E_min = -3, E_max = 5, nk=100, nbins=200):
    """
    Compute density of states within a given energy range, using histogram method.

    Parameters
    ----------
    nx, ny : int
        Plane-wave cutoffs along reciprocal lattice directions.
    hz : float
        magnetization parameter.
    V : float
        Lattice potential amplitude.
    E_min, E_max : float
        Energy window within which density of states is calculated. 
        (default: -3 to 5)
    nk : int
        Number of k-points along each direction in the Brillouin zone.
    nbins : int
        Number of bins for the histogram.

    Returns
    -------
    energy_bins : ndarray
        Energy bin centers.
    dos : ndarray
        Density of states.
    """

    # Reciprocal lattice vectors for triangular lattice
    a = 1
    ax = a
    ay = np.sqrt(3) * a

    b1 = np.array([2*np.pi/ax, 0])
    b2 = np.array([2*np.pi/ax * (-1/2), 2*np.pi/ay * (1/2)])

    # First Brillouin zone vertices (hexagon)
    # For triangular lattice, the BZ is a hexagon
    # We'll sample the parallelogram defined by b1 and b2, 
    # which tiles the BZ properly

    # Collect all eigenvalues
    all_eigenvalues = []
    k_points = []

    print(f"Computing eigenvalues for triangular lattice BZ...")

    count = 0
    # Sample the first BZ as a parallelogram
    for i in range(nk):
        for j in range(nk):
            # Map to [0,1] x [0,1] and then to BZ
            u = i / nk
            v = j / nk

            # k-point in the first BZ
            k = u * b1 + v * b2
            kx, ky = k[0], k[1]

            k_points.append([kx, ky])

            H = build_hamiltonian(nx, ny, kx, ky, hz, V)
            eigenvalues = np.linalg.eigvalsh(H)
            all_eigenvalues.extend(eigenvalues)
            count += 1

    all_eigenvalues = np.array(all_eigenvalues)
    k_points = np.array(k_points)

    print(f"Total k-points sampled: {count}")

    #Energy values within the given energy window
    Energy = [];
    for val in all_eigenvalues:
        if E_min <= val <= E_max:
            Energy.append(val)

    # Create histogram
    dos, bin_edges = np.histogram(Energy, bins=nbins, density=True)
    energy_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    return energy_bins, dos


def plot_dos(energy_bins, dos):
    """
    Plot the density of states.

    Parameters
    ----------
    energy_bins : ndarray
        Energy bin centers.
    dos : ndarray
        Density of states.
    """

    plt.plot(energy_bins,dos,'k')

    plt.ylim(0,np.max(dos)+0.05)

    plt.xlabel('Energy')
    plt.ylabel('Density of states')

    plt.show()


# In[ ]:




