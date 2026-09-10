#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matploltib.pyplot as plt


# In[ ]:


def compute_berry_curvature(nx, ny, kx, ky, hz, V, band_index, dk=1e-4):
    """
    Compute the Berry curvature Ω(k) for a given band using the Fukui method.

    Ω(k) = Im[ln(U_1 U_2 U_1^† U_2^†)]

    where U_i are link variables (overlaps between neighboring states).

    Parameters
    ----------
    dk : float
        Step size for finite differences in k-space.

    Returns
    -------
    berry_curvature : float
        Berry curvature at the given k-point for the specified band.
    """

    # Get eigenstates at k-point and neighbors
    H = build_hamiltonian(nx, ny, kx, ky, hz, V)
    _, psi = eigh(H)
    psi_n = psi[:, band_index]

    H_kx = build_hamiltonian(nx, ny, kx + dk, ky, hz, V)
    _, psi_kx = eigh(H_kx)
    psi_n_kx = psi_kx[:, band_index]

    H_ky = build_hamiltonian(nx, ny, kx, ky + dk, hz, V)
    _, psi_ky = eigh(H_ky)
    psi_n_ky = psi_ky[:, band_index]

    H_kxky = build_hamiltonian(nx, ny, kx + dk, ky + dk, hz, V)
    _, psi_kxky = eigh(H_kxky)
    psi_n_kxky = psi_kxky[:, band_index]

    # Compute link variables (overlaps)
    U1 = np.vdot(psi_n, psi_n_kx)
    U2 = np.vdot(psi_n_kx, psi_n_kxky)
    U3 = np.vdot(psi_n_kxky, psi_n_ky)
    U4 = np.vdot(psi_n_ky, psi_n)

    # Berry curvature from plaquette
    # Ω = Im[ln(U1 * U2 * U3 * U4)] / dk^2
    F = U1 * U2 * U3 * U4
    berry_curvature = np.angle(F) / (dk**2)

    return berry_curvature


# In[ ]:


def compute_hall_conductivity(nx, ny, hz, V,band_index, mu, nk=50, dk=1e-4):
    """
    Compute the intrinsic anomalous Hall conductivity.

    σ^int_xy(μ) = (e²/h) * (1/2π) ∫ d²k Ω(k) Θ(μ - ε(k))

    Parameters
    ----------
    mu : float
        Chemical potential (Fermi energy).
    nk : int
        Number of k-points along each direction for BZ integration.
    dk : float
        Step size for Berry curvature calculation.

    Returns
    -------
    sigma_xy : float
        Anomalous Hall conductivity in units of e²/h.
    """

    # Reciprocal lattice vectors for triangular lattice
    a = 1
    ax = a
    b1 = np.array([2*np.pi/ax, 0])
    b2 = np.array([2*np.pi/ax * (-1/2), 2*np.pi/(np.sqrt(3)*ax) * (1/2)])

    # Sample the first Brillouin zone
    total_berry_flux = 0.0
    num_occupied = 0

    print(f"Computing Hall conductivity with {nk}x{nk} k-points...")

    berry_curvatures = []
    energies_list = []
    k_points = []

    for i in range(nk):
        if i % 10 == 0:
            print(f"Progress: {i}/{nk}")

        for j in range(nk):
            # Map to first BZ
            u = i / nk
            v_frac = j / nk

            k = u * b1 + v_frac * b2
            kx, ky = k[0], k[1]

            # Get energy at this k-point
            H = build_hamiltonian(nx, ny, kx, ky, hz, V, s, v)
            energies = np.linalg.eigvalsh(H)
            energy = energies[band_index]

            energies_list.append(energy)
            k_points.append([kx, ky])

            # Check if state is occupied
            if energy <= mu:
                # Compute Berry curvature
                if method == 'fukui':
                    omega = compute_berry_curvature(nx, ny, kx, ky, hz, V, s, v, 
                                                   band_index, dk)
                else:  # velocity method
                    omega = compute_berry_curvature_v2(nx, ny, kx, ky, hz, V, s, v, 
                                                       band_index, dk)

                total_berry_flux += omega
                berry_curvatures.append(omega)
                num_occupied += 1
            else:
                berry_curvatures.append(0.0)

    # Normalize by BZ area and number of k-points
    # BZ area for triangular lattice
    BZ_area = np.linalg.norm(np.cross(b1, b2))

    # Hall conductivity in units of e²/h
    sigma_xy = total_berry_flux / (2 * np.pi * nk**2) * BZ_area

    print(f"\nComputation complete!")
    print(f"Number of occupied states: {num_occupied}/{nk**2}")
    print(f"Total Berry flux: {total_berry_flux:.6f}")
    print(f"BZ area: {BZ_area:.6f}")

    return sigma_xy, np.array(k_points), np.array(energies_list), np.array(berry_curvatures)


def plot_hall_vs_mu(nx, ny, hz, V, band_index, mu_list, nk=30, dk=1e-4):
    """
    Plot Hall conductivity as a function of chemical potential.
    """

    sigma_list = []

    print("Computing Hall conductivity vs chemical potential...")

    for i, mu in enumerate(mu_list):
        print(f"\nμ = {mu:.4f} ({i+1}/{len(mu_list)})")
        sigma_xy, _, _, _ = compute_hall_conductivity(nx, ny, hz, V, 
                                                       band_index, mu, nk, dk)
        sigma_list.append(sigma_xy)

    sigma_list = np.array(sigma_list)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mu_list, sigma_list, 'b-', linewidth=2.5, marker='o', markersize=6)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax.set_xlabel(r'Chemical Potential $\mu$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Hall Conductivity $\sigma_{xy}$ (e²/h)', fontsize=14, fontweight='bold')
    ax.set_title(f'Anomalous Hall Conductivity vs Chemical Potential\n' + 
                 f'Band {band_index}, V={V}, h₀={hz[0]}, h₁={hz[1]}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return mu_list, sigma_list

