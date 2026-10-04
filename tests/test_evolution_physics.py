"""Independent propagation and coherent-decomposition reference checks."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from hhg.models import SSHModel
from hhg.fields import SinSquaredPulse
from hhg.solvers import TimeEvolver
from hhg.analysis.decomposition import run_current_decomposition


class EvolutionPhysicsTests(unittest.TestCase):
    def test_static_cn_has_second_order_global_error_and_preserves_gram(self):
        model = SSHModel(N=20, delta=0.15)
        initial = model.get_ground_state().astype(complex)
        # A spatial phase gives a nonstationary, orthonormal reference state.
        initial *= np.exp(0.17j * model.positions[:, None])
        duration = 4.0
        exact = expm(-1j * duration * model.static_H.toarray()) @ initial
        errors = []
        for dt in (0.4, 0.2, 0.1):
            for _, _, final in TimeEvolver(model).evolve(
                    lambda t: 0.0, duration, dt, False, initial):
                pass
            errors.append(np.linalg.norm(final - exact))
            np.testing.assert_allclose(final.conj().T @ final,
                                       initial.conj().T @ initial, atol=1e-12)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 3.9)
            self.assertLess(coarse / fine, 4.1)

    def test_driven_propagation_preserves_occupied_orthonormality(self):
        model = SSHModel(N=20)
        initial = model.get_ground_state()
        pulse = SinSquaredPulse(A0=0.2, omega=1.0, ncyc=1.0)
        for _, _, final in TimeEvolver(model).evolve(pulse, 6.0, 0.05, False, initial):
            pass
        np.testing.assert_allclose(final.conj().T @ final, np.eye(10), atol=1e-12)

    def test_coherent_acceleration_decomposition_adds_to_total(self):
        model = SSHModel(N=20)
        initial = model.get_ground_state()
        pulse = SinSquaredPulse(A0=0.2, omega=1.0, ncyc=0.5)
        with patch.object(model, 'get_ground_state', return_value=initial):
            _, acceleration = run_current_decomposition(
                model, pulse, dt=0.05, omega=1.0, ncyc=0.5)
        np.testing.assert_allclose(acceleration['occ'],
                                   acceleration['VB'] + acceleration['ES'],
                                   rtol=1e-9, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
