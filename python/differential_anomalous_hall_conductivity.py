#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matploltib.pyplot as plt


# In[ ]:


def compute_differential_hall_conductivity(nx, ny, hz, V, band_index, 
                                          mu_list, nk=50, dk=1e-4, 
                                          delta_width=0.01):
    """
    Compute the differential intrinsic anomalous Hall conductivity.

    dσ^int_xy(μ)/dμ = (e²/h) * (1/2π) ∫ d²k Ω(k) δ(μ - ε(k))

    The delta function is approximated by a Lorentzian or Gaussian.

    Parameters
    ----------
    mu_list : array-like
        Array of chemical potential values to evaluate.
    nk : int
        Number of k-points along each direction for BZ integration.
    dk : float
        Step size for Berry curvature calculation.
    delta_width : float
        Width parameter for delta function approximation (broadening).

    Returns
    -------
    mu_array : ndarray
        Chemical potential values.
    dsigma_dmu : ndarray
        Differential Hall conductivity dσ_xy/dμ in units of e²/h.
    """

    # Reciprocal lattice vectors for triangular lattice
    a = 1
    ax = a
    b1 = np.array([2*np.pi/ax, 0])
    b2 = np.array([2*np.pi/ax * (-1/2), 2*np.pi/(np.sqrt(3)*ax) * (1/2)])

    # BZ area
    BZ_area = np.linalg.norm(np.cross(b1, b2))

    print(f"Computing differential Hall conductivity...")
    print(f"Sampling {nk}x{nk} k-points in the Brillouin zone...")

    # First pass: collect all energies and Berry curvatures at k-points
    energies_k = []
    berry_k = []
    k_points = []

    for i in range(nk):
        if i % 10 == 0:
            print(f"Progress (data collection): {i}/{nk}")

        for j in range(nk):
            # Map to first BZ
            u = i / nk
            v_frac = j / nk

            k = u * b1 + v_frac * b2
            kx, ky = k[0], k[1]

            k_points.append([kx, ky])

            # Get energy at this k-point
            H = build_hamiltonian(nx, ny, kx, ky, hz, V)
            energies = np.linalg.eigvalsh(H)
            energy = energies[band_index]
            energies_k.append(energy)

            # Compute Berry curvature
            omega = compute_berry_curvature(nx, ny, kx, ky, hz, V, band_index, dk)
            berry_k.append(omega)

    energies_k = np.array(energies_k)
    berry_k = np.array(berry_k)
    k_points = np.array(k_points)

    print(f"Energy range: [{energies_k.min():.4f}, {energies_k.max():.4f}]")

    # Second pass: for each μ, compute dσ/dμ using delta function
    dsigma_dmu = []

    print(f"\nComputing dσ/dμ for {len(mu_list)} chemical potential values...")

    for idx, mu in enumerate(mu_list):
        if idx % 5 == 0:
            print(f"Progress (integration): {idx}/{len(mu_list)}, μ = {mu:.4f}")

        # Delta function approximation (Gaussian approximation)
        gamma = delta_width
        delta_approx = (1/(delta_width * np.sqrt(2*np.pi))) * np.exp(-0.5*((mu - energies_k)/delta_width)**2)

        # Integrate: Σ_k Ω(k) δ(μ - ε(k))
        integrand = berry_k * delta_approx
        integral = np.sum(integrand)

        # Normalize by number of k-points and multiply by BZ area
        dsigma = integral * BZ_area / (2 * np.pi * nk**2)
        dsigma_dmu.append(dsigma)

    dsigma_dmu = np.array(dsigma_dmu)
    mu_array = np.array(mu_list)

    return mu_array, dsigma_dmu, k_points, energies_k, berry_k

def plot_differential_hall_conductivity(mu_array, dsigma_dmu, hz, V, band_index,
                                        save_fig=False, filename='differential_hall.png'):
    """
    Plot the differential Hall conductivity dσ_xy/dμ.

    Parameters
    ----------
    mu_array : ndarray
        Chemical potential values.
    dsigma_dmu : ndarray
        Differential Hall conductivity values.
    hz, V, band_index : parameters for plot annotation
    save_fig : bool
        Whether to save the figure.
    filename : str
        Filename for saving.
    """

    fig, ax = plt.subplots(figsize=(12, 7))

    # Main plot
    ax.plot(mu_array, dsigma_dmu, 'b-', linewidth=2.5, label=r'$d\sigma_{xy}/d\mu$')
    ax.fill_between(mu_array, 0, dsigma_dmu, alpha=0.3)

    # Zero line
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)

    # Labels and title
    ax.set_xlabel(r'Chemical Potential $\mu$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$d\sigma_{xy}/d\mu$ (e²/h per energy)', fontsize=14, fontweight='bold')
    ax.set_title(f'Differential Anomalous Hall Conductivity\n' + 
                 f'Band {band_index}, V={V}, h₀={hz[0]}, h₁={hz[1]}', 
                 fontsize=16, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')

    # Annotations
    ax.text(0.02, 0.98, 
            f'Positive peaks: electron-like topology\n' +
            f'Negative peaks: hole-like topology',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    plt.show()


# In[ ]:


def compute_thermal_differential_hall_conductivity(nx, ny, hz, V, band_index, 
                                                   mu_list, T_list, nk=50, dk=1e-4, 
                                                   T0=1.0):
    """
    Compute the differential intrinsic anomalous Hall conductivity with thermal broadening.

    dσ^int_xy/dμ(μ; T) = (e²/2πh) ∫ d²k Ω(k) * (1/k_BT) * sech²((ε(k) - μ)/(2k_BT))

    where sech²(x) = 1/cosh²(x) and k_B T = T/T_0 with T_0 = 1/k_B.

    Parameters
    ----------
    mu_list : array-like
        Array of chemical potential values to evaluate.
    T_list : array-like
        Array of temperature values (in units where k_B T_0 = 1).
    nk : int
        Number of k-points along each direction for BZ integration.
    dk : float
        Step size for Berry curvature calculation.
    T0 : float
        Temperature scale T_0 = 1/k_B (default: 1.0).

    Returns
    -------
    mu_array : ndarray
        Chemical potential values.
    dsigma_dmu_thermal : dict
        Dictionary with T as keys and dσ_xy/dμ arrays as values.
    """

    # Reciprocal lattice vectors for triangular lattice
    a = 1
    ax = a
    b1 = np.array([2*np.pi/ax, 0])
    b2 = np.array([2*np.pi/ax * (-1/2), 2*np.pi/(np.sqrt(3)*ax) * (1/2)])

    # BZ area
    BZ_area = np.linalg.norm(np.cross(b1, b2))

    print(f"Computing thermal differential Hall conductivity...")
    print(f"Sampling {nk}x{nk} k-points in the Brillouin zone...")

    # First pass: collect all energies and Berry curvatures at k-points
    energies_k = []
    berry_k = []
    k_points = []

    for i in range(nk):
        if i % 10 == 0:
            print(f"Progress (data collection): {i}/{nk}")

        for j in range(nk):
            # Map to first BZ
            u = i / nk
            v_frac = j / nk

            k = u * b1 + v_frac * b2
            kx, ky = k[0], k[1]

            k_points.append([kx, ky])

            # Get energy at this k-point
            H = build_hamiltonian(nx, ny, kx, ky, hz, V)
            energies = np.linalg.eigvalsh(H)
            energy = energies[band_index]
            energies_k.append(energy)

            # Compute Berry curvature
            omega = compute_berry_curvature(nx, ny, kx, ky, hz, V, 
                                           band_index, dk)
            berry_k.append(omega)

    energies_k = np.array(energies_k)
    berry_k = np.array(berry_k)
    k_points = np.array(k_points)

    print(f"Energy range: [{energies_k.min():.4f}, {energies_k.max():.4f}]")

    # Second pass: for each T and μ, compute dσ/dμ with thermal broadening
    dsigma_dmu_thermal = {}
    mu_array = np.array(mu_list)

    for T in T_list:
        print(f"\nComputing for T = {T:.4f} T_0...")
        dsigma_dmu = []

        kB_T = T / T0  # Effective thermal energy

        for idx, mu in enumerate(mu_list):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(mu_list)}, μ = {mu:.4f}")

            # Thermal broadening function: (1/k_BT) * sech²((ε - μ)/(2k_BT))
            # sech²(x) = 1/cosh²(x)
            if kB_T < 1e-10:  # Zero temperature limit
                # Use delta function approximation
                delta_width = 0.01
                gamma = delta_width
                thermal_factor = (1/np.pi) * (gamma/2) / ((mu - energies_k)**2 + (gamma/2)**2)
            else:
                x = (energies_k - mu) / (2 * kB_T)
                # Compute sech²(x) = 1/cosh²(x)
                cosh_x = np.cosh(x)
                sech_sq = 1.0 / (cosh_x**2)
                thermal_factor = sech_sq / (2 * kB_T)  # Include 1/(2k_BT) factor

            # Integrate: Σ_k Ω(k) * thermal_factor
            integrand = berry_k * thermal_factor
            integral = np.sum(integrand)

            # Normalize by number of k-points and multiply by BZ area
            # Note: factor of 1/2π comes from the formula
            dsigma = integral * BZ_area / (2 * np.pi * nk**2)
            dsigma_dmu.append(dsigma)

        dsigma_dmu_thermal[T] = np.array(dsigma_dmu)

    print("\nComputation complete!")

    return mu_array, dsigma_dmu_thermal, k_points, energies_k, berry_k

def plot_thermal_differential_hall_conductivity(mu_array, dsigma_dmu_thermal, 
                                               T_list, hz, V, band_index,
                                               save_fig=False, 
                                               filename='thermal_differential_hall.png'):
    """
    Plot the differential Hall conductivity with thermal broadening for multiple temperatures.

    Parameters
    ----------
    mu_array : ndarray
        Chemical potential values.
    dsigma_dmu_thermal : dict
        Dictionary with T as keys and dσ_xy/dμ arrays as values.
    T_list : array-like
        List of temperature values.
    hz, V, band_index : parameters for plot annotation
    save_fig : bool
        Whether to save the figure.
    filename : str
        Filename for saving.
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for different temperatures
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(T_list)))

    # Plot for each temperature
    for i, T in enumerate(T_list):
        dsigma_dmu = dsigma_dmu_thermal[T]
        label = f'T = {T:.3f} T₀' if T > 0 else 'T = 0 (zero temp.)'
        ax.plot(mu_array, dsigma_dmu, linewidth=2.5, color=colors[i], 
                label=label, alpha=0.8)

    # Zero line
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)

    # Labels and title
    ax.set_xlabel(r'Chemical Potential $\mu


# Main execution
if __name__ == "__main__":
    # Parameters
    nx, ny = 3, 3
    h0, h1 = 0.1, 0.05
    hz = (h0, h1)
    V = 0.2
    s, v = 1.0, 1.0
    band_index = 5

    # First, determine energy range by sampling
    print("Determining energy range...")
    test_energies = []
    for i in range(10):
        for j in range(10):
            kx = i * 0.2
            ky = j * 0.2
            H = build_hamiltonian(nx, ny, kx, ky, hz, V, s, v)
            E = np.linalg.eigvalsh(H)
            test_energies.append(E[band_index])

    E_min = min(test_energies)
    E_max = max(test_energies)
    print(f"Approximate energy range: [{E_min:.4f}, {E_max:.4f}]")

    # Chemical potential range
    mu_list = np.linspace(E_min - 0.1, E_max + 0.1, 100)

    print("\n" + "=" * 70)
    print("Computing Differential Intrinsic Anomalous Hall Conductivity")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  nx, ny = {nx}, {ny}")
    print(f"  hz = {hz}")
    print(f"  V = {V}")
    print(f"  band_index = {band_index}")
    print(f"  μ range: [{mu_list[0]:.4f}, {mu_list[-1]:.4f}]")
    print("-" * 70)

    # Compute differential Hall conductivity
    mu_array, dsigma_dmu, k_points, energies_k, berry_k = \
        compute_differential_hall_conductivity(
            nx, ny, hz, V, s, v, band_index, 
            mu_list, nk=30, dk=1e-4, delta_width=0.02
        )

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Maximum dσ/dμ = {np.max(dsigma_dmu):.6f} e²/h")
    print(f"Minimum dσ/dμ = {np.min(dsigma_dmu):.6f} e²/h")
    print(f"Total Berry flux = {np.sum(berry_k):.6f}")
    print("=" * 70)

    # Plot results
    plot_differential_hall_conductivity(mu_array, dsigma_dmu, hz, V, band_index,
                                       save_fig=True, 
                                       filename='differential_hall_conductivity.png')

    # Comprehensive analysis
    plot_combined_analysis(mu_array, dsigma_dmu, energies_k, berry_k, 30,
                          hz, V, band_index)

    # DOS comparison
    plot_berry_curvature_weighted_dos(mu_array, dsigma_dmu, energies_k, berry_k)
, fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$d\sigma_{xy}/d\mu$ (e²/h per energy)', fontsize=14, fontweight='bold')
    ax.set_title(f'Thermal Differential Anomalous Hall Conductivity\n' + 
                 f'Band {band_index}, V={V}, h₀={hz[0]}, h₁={hz[1]}', 
                 fontsize=16, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)

    # Annotations
    ax.text(0.02, 0.98, 
            f'Thermal broadening: sech²((ε-μ)/(2k_BT))\n' +
            f'Higher T → broader peaks',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    plt.show()

