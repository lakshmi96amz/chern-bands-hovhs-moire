#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def get_band_structure(nx,ny,hz,V,bands = []):

    """
    Compute and plot the band structure along the high-symmetry path
    Γ → K → M → Γ for the moiré surface Hamiltonian.

    Parameters
    ----------
    nx, ny : int
        Plane-wave cutoffs.
    hz : float
        magnetization parameter.
    V : float
        lattice potential amplitude.
    bands : list of int, optional
        Band indices to plot. If None, all bands are plotted.
    """

    #No. of k-points per segment
    N1 = 306
    N2 = 153
    N3 = 265

    a = 1 #Lattice constant

    #Building a list of k-points along the high-symmetry path
    Kx1 = np.linspace(0,2*np.pi/(a*3),N1)
    Kx2 = np.linspace(2*np.pi/(a*3),0,N2)
    Ky3 = np.linspace(2*np.pi/(np.sqrt(3)*a),0,N3)

    Kx = []
    Ky = []

    # Γ → K
    for kx in Kx1 :
        Kx.append(kx)
        Ky.append(np.sqrt(3)*kx)
    # K → M
    for kx in Kx2 :
        Kx.append(kx)
        Ky.append(2*np.pi/(np.sqrt(3)*a))
    # M → Γ
    for ky in Ky3 :
        Kx.append(0)
        Ky.append(ky)

    #Diagonalize the Hamiltonian along the path
    Energy = []
    for i in range(len(Kx)):
        eigval = np.sort(np.linalg.eigvalsh(build_hamiltonian(nx,ny,Kx[i],Ky[i],hz,V)))
        Energy.append(eigval)

    #Plot the bands
    x = np.linspace(1,len(Kx),len(Kx))
    for r in bands :
        y = []
        for i in range(len(Kx)):
            value = Energy[i][r]
            y.append(value)
        plt.plot(x,y)

    plt.ylabel('Energy')

    locs, labels = plt.xticks()
    plt.xticks([0,N1,N1+N2,N1+N2+N3],['$\Gamma$','K','M','$\Gamma$'])
    plt.show()

