#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def build_hamiltonian(nx, ny, kx, ky, hz, V):
    """
    Construct the Hamiltonian for a single Dirac fermion
    in a C3-symmetric moiré lattice potential V(r) under a constant magnetization hz.

    Model:
        H = (k × σ)_z + V(r) σ_0 + hz σ_3
    where
        V(r) = 2 V Σⱼ [cos(Gⱼ·r) - cos(Gⱼ·r + φ)]  with φ = 2π/3

    Parameters
    ----------
    nx, ny : int
        Plane-wave cutoffs along reciprocal lattice directions.
    kx, ky : float
        Crystal momentum.
    hz : float
        uniform magnetization strength.
    V : float
        lattice potential amplitude.

    Returns
    -------
    H : ndarray (complex)
        Hamiltonian matrix of dimension 2 * (nx+1) * (ny+1).
    """

    # Lattice constants
    a = 1  # length is scaled by a
    ax = a
    ay = np.sqrt(3) * a

    # Phase for C3 symmetry
    phi = 2 * np.pi / 3

    # C3 symmetric coupling (replacing V in C6 case)
    V_c3 = V * (1 - np.exp(1j * phi))

    #Reciprocal lattice vectors
    n = int(2*(nx+1)*(ny+1))

    R = []

    dR = [-2*np.pi/ax,-2*np.pi/ay]
    xlist = np.linspace(-nx/2,nx/2,nx+1)
    ylist = np.linspace(-ny/2,ny/2,ny+1)
    for i in xlist:
        for j in ylist :
            value = [(i-j)*dR[0],(i+j)*dR[1]]
            R.append(value)

    # Constructing Hamiltonian
    H = np.zeros((n, n), complex)

    # Diagonal terms
    for l in range(nx+1):
        for m in range(ny+1):
            H[l*(ny+1) + m, l*(ny+1) + m] = hz
            H[l*(ny+1)+m,l*(ny+1)+m + (nx+1)*(ny+1)] = -(1.j*(kx-R[l*(ny+1)+m][0])+ (ky - R[l*(ny+1)+m][1]))

            H[l*(ny+1) + m+(nx+1)*(ny+1), l*(ny+1) + m+(nx+1)*(ny+1)] = -hz
            H[l*(ny+1)+m+ (nx+1)*(ny+1),l*(ny+1)+m ] = (1.j*(kx-R[l*(ny+1)+m][0])- (ky - R[l*(ny+1)+m][1]))

    # Hopping along first direction (x-direction)
    for l in range(nx):
        for m in range(ny + 1):
            H[l*(ny+1) + m, (l+1)*(ny+1) + m] = V_c3 
            H[(l+1)*(ny+1) + m, l*(ny+1) + m] = np.conj(V_c3 )

            H[l*(ny+1) + m + (nx+1)*(ny+1), (l+1)*(ny+1) + m + (nx+1)*(ny+1)] = V_c3 
            H[(l+1)*(ny+1) + m + (nx+1)*(ny+1), l*(ny+1) + m + (nx+1)*(ny+1)] = np.conj(V_c3)

    # Hopping along second direction (y-direction)
    for l in range(nx + 1):
        for m in range(ny):
            H[l*(ny+1) + m, l*(ny+1) + m+1] = V_c3
            H[l*(ny+1) + m+1, l*(ny+1) + m] = np.conj(V_c3)

            H[l*(ny+1) + m + (nx+1)*(ny+1), l*(ny+1) + m+1 + (nx+1)*(ny+1)] = V_c3 
            H[l*(ny+1) + m+1 + (nx+1)*(ny+1), l*(ny+1) + m + (nx+1)*(ny+1)] = np.conj(V_c3)

    # Hopping along third direction (diagonal)
    for l in range(nx):
        for m in range(ny):
            H[l*(ny+1) + m, (l+1)*(ny+1) + m+1] = V_c3 
            H[(l+1)*(ny+1) + m+1, l*(ny+1) + m] = np.conj(V_c3 )

            H[l*(ny+1) + m + (nx+1)*(ny+1), (l+1)*(ny+1) + m+1 + (nx+1)*(ny+1)] = V_c3 
            H[(l+1)*(ny+1) + m+1 + (nx+1)*(ny+1), l*(ny+1) + m + (nx+1)*(ny+1)] = np.conj(V_c3)

    return H

