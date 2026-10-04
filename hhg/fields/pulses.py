from abc import ABC, abstractmethod
import numpy as np

class Field(ABC):
    """Abstract base class for time-dependent fields."""
    
    @abstractmethod
    def __call__(self, t: float):
        """Return the field value at time t."""
        pass

class SinSquaredPulse(Field):
    """
    Electric field derived from a finite sin^2 vector-potential envelope.
    E(t) = -A0 * [ d(env)/dt * sin(wt) + w * env * cos(wt) ]
    The field is zero outside 0 <= t <= 2*pi*ncyc/omega.
    """
    
    def __init__(self, A0: float = 0.2, omega: float = 0.0075, ncyc: float = 5):
        """
        Args:
            A0 (float): Vector-potential amplitude in atomic units.
            omega (float): Positive angular frequency in atomic units.
            ncyc (float): Positive number of optical cycles.
        """
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be finite and positive.")
        if not np.isfinite(ncyc) or ncyc <= 0:
            raise ValueError("ncyc must be finite and positive.")
        self.A0 = A0
        self.omega = omega
        self.ncyc = ncyc

    def __call__(self, t: float) -> float:
        """
        Calculate electric field at time t.
        """
        if t <= 0 or t >= (2 * np.pi * self.ncyc / self.omega):
            return 0.0

        omega_t = self.omega * t

        arg = omega_t / (2 * self.ncyc)
        envelope = np.sin(arg) ** 2
        
        # Derivative of envelope wrt t
        d_envelope = (self.omega / self.ncyc) * np.sin(arg) * np.cos(arg)
        
        E = -self.A0 * (d_envelope * np.sin(omega_t) + self.omega * envelope * np.cos(omega_t))
        return E
