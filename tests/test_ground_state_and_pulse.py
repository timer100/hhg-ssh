"""Physical reference checks for review findings R1 and R2.

Run from the repository root: python -m unittest discover -s tests -v
"""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import eigsh as scipy_eigsh

from hhg.fields import SinSquaredPulse
from hhg.models import SSHModel


class GroundStateTests(unittest.TestCase):
    def test_half_filling_matches_dense_ground_energy(self):
        for n in (2, 4, 6, 8, 10, 22, 100):
            for delta, potential in ((-0.15, 0.0), (0.15, 0.0),
                                     (-0.15, 0.2), (-0.15, 1e-10)):
                with self.subTest(n=n, delta=delta, potential=potential):
                    model = SSHModel(N=n, delta=delta, V_A=potential)
                    h = model.build_time_dependent_hamiltonian(
                        0, lambda t: 0.0
                    ).toarray()
                    energies, vectors = np.linalg.eigh(h)
                    occupied = model.get_ground_state()
                    count = n // 2
                    self.assertEqual(occupied.shape, (n, count))
                    np.testing.assert_allclose(
                        occupied.conj().T @ occupied, np.eye(count), atol=1e-12
                    )
                    projected = occupied.conj().T @ h @ occupied
                    np.testing.assert_allclose(
                        np.linalg.eigvalsh(projected), energies[:count],
                        rtol=1e-10, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        h @ occupied, occupied @ projected, atol=1e-12
                    )
                    # Compare physical occupied subspaces, not arbitrary
                    # eigenvector signs. Skip unresolved Fermi-level gaps.
                    if energies[count] - energies[count - 1] > 1e-8:
                        reference = vectors[:, :count]
                        np.testing.assert_allclose(
                            occupied @ occupied.conj().T,
                            reference @ reference.conj().T, atol=1e-10
                        )

    def test_lower_edge_parity_changes_with_chain_length(self):
        for n, expected_parity in ((6, 1), (8, -1), (10, 1), (22, 1)):
            with self.subTest(n=n):
                model = SSHModel(N=n)
                edge = model.get_ground_state()[:, -1]
                self.assertLess(float(edge @ model.static_H @ edge), 0.0)
                np.testing.assert_allclose(
                    edge[::-1], expected_parity * edge, atol=1e-12
                )

    def test_default_near_degenerate_edge_remains_odd(self):
        model = SSHModel()
        for _ in range(3):
            edge = model.get_ground_state()[:, -1]
            np.testing.assert_allclose(edge[::-1], -edge, atol=1e-12)
            self.assertLess(float(edge @ model.static_H @ edge), 0.0)

    def test_unresolved_long_chain_edges_have_required_parity(self):
        # Regression at the paper's length and its 4m+2 counterpart.
        # Energy sorting cannot reliably distinguish these central branches.
        for n, expected_parity in ((800, -1), (802, 1)):
            for seed in (0, 1, 2):
                with self.subTest(n=n, seed=seed):
                    def controlled_eigsh(h, k, **kwargs):
                        start = np.random.default_rng(seed + n + k).standard_normal(n)
                        return scipy_eigsh(h, k=k, v0=start, **kwargs)

                    model = SSHModel(N=n)
                    with patch('hhg.models.ssh.eigsh', side_effect=controlled_eigsh):
                        occupied = model.get_ground_state()
                    edge = occupied[:, -1]
                    np.testing.assert_allclose(
                        edge[::-1], expected_parity * edge, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        occupied.conj().T @ occupied, np.eye(n // 2), atol=1e-12
                    )
                    self.assertLess(np.linalg.norm(model.static_H @ edge), 1e-12)
                    density = np.sum(np.abs(occupied) ** 2, axis=1)
                    np.testing.assert_allclose(density, density[::-1], atol=1e-12)


class FinitePulseTests(unittest.TestCase):
    def test_zero_outside_support_and_at_endpoints(self):
        for cycles in (5.0, 2.5):
            pulse = SinSquaredPulse(ncyc=cycles)
            duration = 2 * np.pi * pulse.ncyc / pulse.omega
            for t in (-duration, -duration / 4, -1e-12, 0.0, duration,
                      np.nextafter(duration, np.inf), 1.25 * duration,
                      2 * duration):
                with self.subTest(cycles=cycles, t=t):
                    self.assertEqual(pulse(t), 0.0)

    def test_field_matches_negative_vector_potential_derivative(self):
        for omega, cycles in ((0.0075, 5.0), (0.7, 2.5)):
            pulse = SinSquaredPulse(A0=0.2, omega=omega, ncyc=cycles)
            duration = 2 * np.pi * cycles / omega

            def vector_potential(t):
                return (pulse.A0 * np.sin(np.pi * t / duration) ** 2
                        * np.sin(omega * t))

            step = 1e-5 / omega
            for t in np.linspace(0.03 * duration, 0.97 * duration, 31):
                with self.subTest(omega=omega, cycles=cycles, t=t):
                    reference = -(vector_potential(t + step)
                                  - vector_potential(t - step)) / (2 * step)
                    np.testing.assert_allclose(
                        pulse(t), reference, rtol=1e-7, atol=1e-10
                    )

    def test_rejects_invalid_frequency_or_cycles(self):
        for name in ('omega', 'ncyc'):
            for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        SinSquaredPulse(**{name: value})


if __name__ == '__main__':
    unittest.main()
