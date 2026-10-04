import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.linalg import eigh
from ..models.base import Hamiltonian


def compute_momentum_spectrum(model: Hamiltonian):
    """Return (k, energies, weights) for a finite chain's static eigenstates.

    k is in inverse bohr on the unfolded interval [-pi/a, pi/a), where a
    is the mean site spacing (the SSH unit-cell length is 2*a). The N bins
    have spacing 2*pi/(N*a). weights[k_index, state_index] is
    abs(sum_j exp(-1j*k*x_j)*psi_j / sqrt(N))**2, using actual positions.
    This is a point-orbital spectral weight, not momentum-resolved HHG.
    On a dimerized, nonuniform grid this transform is not unitary; weights
    should not be interpreted as normalized probabilities over these bins.
    """
    a = model.a
    if not np.isfinite(a) or a <= 0:
        raise ValueError("model.a must be finite and positive.")
    positions = np.asarray(model.positions, dtype=float)
    if positions.shape != (model.N,) or not np.all(np.isfinite(positions)):
        raise ValueError("model.positions must contain N finite positions.")
    h = model.build_time_dependent_hamiltonian(0, lambda t: 0).toarray()
    energies, eigenvectors = eigh(h)
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(model.N, d=a))
    
    transform = np.exp(-1j * np.outer(k, positions)) / np.sqrt(model.N)
    weights = np.abs(transform @ eigenvectors) ** 2
    return k, energies, weights


def plot_band_structure(model: Hamiltonian, label: str = None, ax=None, show: bool = True):
    """Plot finite-chain energy/momentum weights in the unfolded k interval.

    All states share one color scale, log(1 + N*weight). Returns the axes.
    The logarithmic display reveals weak weights without rescaling each state.
    See compute_momentum_spectrum for the position and normalization conventions.
    """
    k, energies, weights = compute_momentum_spectrum(model)
    colors = np.log1p(model.N * weights)
    norm = Normalize(vmin=0.0, vmax=float(colors.max()))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for i, energy in enumerate(energies):
        ax.scatter(k, np.full_like(k, energy), c=colors[:, i],
                   cmap='gray_r', norm=norm, s=3, marker='.')
    if label:
        ax.set_title(label, fontsize=14)
    ax.set_xlim(-np.pi / model.a, np.pi / model.a)
    ax.set_xlabel(r"$k\ (\mathrm{bohr}^{-1})$", fontsize=14)
    ax.set_ylabel("Energy (a.u.)", fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5)
    if show:
        plt.show()
    return ax
