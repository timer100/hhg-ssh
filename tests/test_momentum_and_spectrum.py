"""Independent physical references for R4 momentum and R8 time spectra."""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hhg.models import SSHModel
from hhg.analysis import (compute_momentum_spectrum, plot_band_structure,
                          compute_hhg_spectrum)


class PeriodicSSH(SSHModel):
    """Test-only ring, whose dispersion is known analytically."""
    def build_static_hamiltonian(self):
        h = super().build_static_hamiltonian().tolil()
        h[0, -1] = h[-1, 0] = self.w
        return h.tocsr()


class MomentumTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_uniform_open_chain_ground_weight_is_at_zero(self):
        for spacing in (1.3, 2.0, 3.1):
            model = SSHModel(N=100, a=spacing, delta=0)
            k, _, weights = compute_momentum_spectrum(model)
            self.assertEqual(k[np.argmax(weights[:, 0])], 0.0)
            self.assertAlmostEqual(k[0], -np.pi / spacing)
            np.testing.assert_allclose(np.diff(k), 2*np.pi/(100*spacing))
            np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-12)

    def test_periodic_chain_weights_follow_analytic_ssh_dispersion(self):
        for spacing in (1.5, 2.5):
            for delta in (-0.15, 0.15):
                model = PeriodicSSH(N=80, a=spacing, delta=delta, V_A=0.1)
                k, energies, weights = compute_momentum_spectrum(model)
                expected_squared = (model.v**2 + model.w**2 + model.V_A**2
                                    + 2*model.v*model.w*np.cos(2*spacing*k))
                error = energies[None, :]**2 - expected_squared[:, None]
                np.testing.assert_allclose(error[weights > 1e-10], 0, atol=1e-12)

    def test_uses_physical_positions_and_is_origin_invariant(self):
        model = SSHModel(N=20, delta=-0.2)
        _, _, reference = compute_momentum_spectrum(model)
        model.positions += 13.25
        _, _, shifted = compute_momentum_spectrum(model)
        np.testing.assert_allclose(shifted, reference, atol=1e-12)
        # Keep the same Hamiltonian but remove intra-cell position offsets.
        model.positions = np.arange(model.N) * model.a
        _, _, uniform = compute_momentum_spectrum(model)
        self.assertGreater(np.max(abs(uniform-reference)), 1e-3)

    def test_plot_shares_color_scale_and_respects_show(self):
        model = SSHModel(N=12, a=3)
        with patch('matplotlib.pyplot.show') as show:
            ax = plot_band_structure(model, show=False)
            show.assert_not_called()
            self.assertEqual(len({id(c.norm) for c in ax.collections}), 1)
            np.testing.assert_allclose(ax.get_xlim(), [-np.pi/3, np.pi/3])
            plot_band_structure(model, ax=ax, show=True)
            show.assert_called_once()


class FourierIntensityTests(unittest.TestCase):
    def test_dc_matches_continuous_hann_integral_across_timesteps(self):
        duration, amplitude = 10.0, 2.0
        for count in (101, 201, 401):
            dt = duration/(count-1)
            _, intensity = compute_hhg_spectrum(np.full(count, amplitude), dt, 1)
            # Integral of A*sin(pi*t/T)^2 over [0,T] is A*T/2.
            self.assertAlmostEqual(intensity[0], (amplitude*duration/2)**2, places=10)

    def test_fixed_duration_tone_peak_is_sampling_invariant(self):
        peaks = []
        for count in (512, 1024, 2048):
            duration = 64.0
            dt = duration/count
            t = np.arange(count)*dt
            omega = 2*np.pi
            h, intensity = compute_hhg_spectrum(np.cos(3*omega*t), dt, omega)
            index = np.argmax(intensity)
            self.assertAlmostEqual(h[index], 3)
            np.testing.assert_allclose(intensity[index], (duration/4)**2, rtol=0.005)
            peaks.append(intensity[index])
        np.testing.assert_allclose(peaks, peaks[-1], rtol=0.005)

    def test_odd_and_even_lengths_keep_every_nonnegative_bin(self):
        for count in (9, 10):
            dt = 0.2
            t = np.arange(count)*dt
            signal = np.cos(2*np.pi*(count//2)*np.arange(count)/count)
            h, intensity = compute_hhg_spectrum(signal, dt, 1)
            omega = 2*np.pi*np.arange(count//2+1)/(count*dt)
            # Direct quadrature of the complex integral, including the top bin.
            values = np.hanning(count)*signal
            real = dt * (np.cos(np.outer(omega, t)) @ values)
            imag = -dt * (np.sin(np.outer(omega, t)) @ values)
            np.testing.assert_allclose(h, omega)
            np.testing.assert_allclose(intensity, real**2+imag**2, atol=1e-13)
            self.assertGreater(intensity[-1], 0)

    def test_invalid_inputs_fail_explicitly(self):
        for signal in ([], [1, 2], [[1, 2, 3]], [1, np.nan, 3], [1j, 2j, 3j]):
            with self.assertRaises(ValueError):
                compute_hhg_spectrum(signal, 0.1, 1)
        for name in ('dt', 'omega_drive'):
            for value in (0, -1, np.inf, np.nan):
                args = dict(dt=0.1, omega_drive=1)
                args[name] = value
                with self.assertRaisesRegex(ValueError, name):
                    compute_hhg_spectrum([0, 1, 0], **args)


if __name__ == '__main__':
    unittest.main()
