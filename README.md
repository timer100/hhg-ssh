# SSH–HHG

**High-order harmonic generation in one-dimensional dimerized chains**

A Python research toolkit developed to reproduce the SSH-chain studies of Jürß and Bauer [1] and Yu [2]. It connects tight-binding electronic structure, laser-driven dynamics, and coherent bulk–edge contributions to HHG.

## Physical framework

For an open chain of $N$ sites, the SSH Hamiltonian is

$$
H_0 = \sum_{n=1}^{N/2} v\,c^\dagger_{n,A}c_{n,B}
    + \sum_{n=1}^{N/2-1} w\,c^\dagger_{n,B}c_{n+1,A}
    + \mathrm{H.c.}
$$

Here $v$ and $w$ are intracell and intercell hopping amplitudes. With zero on-site potential, $|v|<|w|$ supports topological edge states; in this implementation it corresponds to `delta < 0`. Their presence can strongly modify the subgap harmonic yield [1].

In atomic units, the occupied orbitals evolve in the length gauge:

$$
i\partial_t|\psi_m(t)\rangle = [H_0+E(t)\hat{x}]|\psi_m(t)\rangle,
\qquad E(t)=-\partial_t A(t).
$$

The solver uses midpoint Crank–Nicolson propagation and a sine-squared vector-potential envelope. At half filling, the spectrum follows from the total position expectation and a Hann window $W(t)$:

$$
X(t)=\sum_{m=1}^{N/2}\langle\psi_m(t)|\hat{x}|\psi_m(t)\rangle,
\qquad S(\Omega)\propto\left|\mathcal{F}[W(t)\ddot{X}(t)](\Omega)\right|^2.
$$

## Implementation

- **Models:** finite, open-boundary SSH chains with optional staggered on-site potentials.
- **Dynamics:** sparse Crank–Nicolson evolution of half-filled occupied states.
- **Analysis:** HHG spectra, VB/ES decomposition, static energy/momentum spectral weights, and wavefunctions.

The present package implements nearest-neighbor SSH hopping. Extended hopping and the full simulation workflows of [3] are beyond its current scope.

### System-size scope

This toolkit is intended for extended, even-site chains and is not suitable for
small-system simulations. The tutorials use `N=100` and `N=800`; these are example
sizes, not convergence criteria. Very short chains, including `N=2` and `N=4`,
are outside the intended analysis workflow, and some analysis routines may fail.

`compute_momentum_spectrum` uses the physical site positions in
`|sum_j exp(-i*k*x_j) * psi_j / sqrt(N)|^2`. The unfolded momentum interval is
`[-pi/a, pi/a)`, where `a` is the mean site spacing and the SSH unit cell has
length `2*a`. This is a finite-chain static spectral weight, not momentum-resolved
HHG; it is not a normalized momentum probability on a nonuniform site grid.

`compute_hhg_spectrum` returns `|dt * FFT(Hann * acceleration)|^2` at nonnegative
frequencies, including DC and the even-length Nyquist bin. There is no one-sided
factor of two or window-gain correction. This approximates a continuous Fourier
integral, not a power spectral density; timestep comparisons require the same
physical duration and window. Compared with the previous raw FFT convention,
shared-bin intensities are multiplied by `dt**2` (0.01 for `dt=0.1`).

### Edge-state parity

Nearly degenerate edge states can be mixed by a numerical eigensolver. For the
inversion-symmetric topological chain (`V_A == 0`, `delta < 0`), we explicitly
resolve their odd/even parity and select the occupied edge state using Rutkevich's
eigenvalue-ordering result [4, Eq. (35)]: **odd for `N = 4m`, even for `N = 4m + 2`**.
Here `N` counts sites and the nearest-neighbor hoppings are negative. This avoids
ranking an edge-energy splitting below floating-point resolution; the rule is
not imposed when inversion symmetry is broken. Only even `N` is supported.

## Quick start

Use Python 3.12 or newer. The local verification baseline is Python 3.13.16.

```bash
git clone https://github.com/timer100/High-Oder-Harmonic-Generation-SSH.git hhg-ssh
cd hhg-ssh
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS), then install the pinned environment:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m jupyter lab --ip=127.0.0.1
```

`requirements.txt` contains the numerical and notebook dependencies. Install
`requirements-dev.txt` instead for development and notebook validation. The package
build requires `setuptools==84.0.0`; pip installs that pinned backend in an isolated
build environment, while `uv sync` also includes it in the development environment.

Both requirements files list pinned versions exported from `uv.lock` without hashes.

Explore the simulation and analysis workflows in the tutorial notebooks linked below.

![Example SSH harmonic spectra](Images/HHG_SSH.png)

**Tutorials:** [HHG & decomposition](Tutorial_SSH.ipynb) · [Electronic structure](Tutorial_Static_Analysis.ipynb)
**Repository example spectra**: See also the [VB/ES decomposition](Images/HHG_SSH_CD.png).

## Verification and dependencies

[`pyproject.toml`](pyproject.toml) defines dependency ranges; [`uv.lock`](uv.lock)
records exact versions and distribution hashes. With `uv==0.12.23`:

```bash
uv sync --locked --extra notebook
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/validate_notebooks.py
```

The tests cover occupied-state/parity references, pulse support, physical momentum
calibration, FFT conventions, CN convergence/orthonormality, and coherent
decomposition. Full notebook validation writes copies under ignored `.validation/`.
CI runs tests on Windows/Linux with Python 3.12–3.14 and executes full tutorials.
Passing these checks does not establish convergence or
paper agreement for every parameter set.

## References

1. H. Jürß and D. Bauer, “High-harmonic generation in Su-Schrieffer-Heeger chains,” *Physical Review B* **99**, 195428 (2019). [doi:10.1103/PhysRevB.99.195428](https://doi.org/10.1103/PhysRevB.99.195428)
2. C. Yu, “High-order harmonic generation from Su-Schrieffer-Heeger chains with edge states: Three-step processes and their interference,” *Physical Review A* **110**, 033113 (2024). [doi:10.1103/PhysRevA.110.033113](https://doi.org/10.1103/PhysRevA.110.033113)
3. C.-T. Liu, J-S You, and H.-C. Hsu, “Even-harmonic generation from topological edge states in generalized Su-Schrieffer-Heeger models,” *arXiv:2608.22136* (2026), preprint. [doi:10.48550/arXiv.2608.22136](https://doi.org/10.48550/arXiv.2608.22136)
4. S. B. Rutkevich, “A Formula for Eigenvalues of Jacobi Matrices with a Reflection Symmetry,” *Advances in Mathematical Physics* **2018**, 9784091 (2018). [doi:10.1155/2018/9784091](https://doi.org/10.1155/2018/9784091) · [Full text, Eq. (35)](https://arxiv.org/pdf/1510.01860#page=6)

## Intended use & disclaimer

This code is intended for academic reference, research, and education and is distributed under the [MIT License](LICENSE), which also permits commercial use subject to its terms. It is provided on an **“AS IS” basis, without warranty of any kind**. Liability is governed by the MIT License. Users are responsible for independently verifying the implementation and physical interpretation before relying on or publishing results. Please cite the relevant original studies when using their methods or findings.

## License

[MIT](LICENSE)
