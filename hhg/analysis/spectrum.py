import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_hhg_spectrum(acceleration: np.ndarray, dt: float, omega_drive: float):
    """Return nonnegative harmonic orders and a windowed Fourier intensity.

    For N real, uniformly spaced acceleration samples, use a symmetric Hann
    window and F(Omega) = dt * sum_n W_n * acceleration_n * exp(-i*Omega*n*dt).
    Intensity is abs(F)**2: no N normalization, window-gain correction, or
    one-sided factor of two. This approximates a continuous time integral,
    not a power spectral density; duration/window still affect intensities.
    Compare timesteps at the same physical observation duration and window.

    dt is in atomic time units and omega_drive is an angular frequency in
    atomic units. Output includes DC and, for even N, the Nyquist bin.
    Relative to the former raw FFT convention, shared-bin intensities are
    multiplied by dt**2. Returns two arrays of length N//2 + 1.
    """
    signal = np.asarray(acceleration)
    if signal.ndim != 1 or signal.size < 3:
        raise ValueError("acceleration must be a 1D array with at least 3 samples.")
    if not np.isrealobj(signal) or not np.all(np.isfinite(signal)):
        raise ValueError("acceleration must contain finite real values.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(omega_drive) or omega_drive <= 0:
        raise ValueError("omega_drive must be finite and positive.")
    window = np.hanning(signal.size)
    amplitude = dt * rfft(signal * window)
    harmonics = 2 * np.pi * rfftfreq(signal.size, d=dt) / omega_drive
    return harmonics, np.abs(amplitude) ** 2
