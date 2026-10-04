import numpy as np
from scipy.linalg import eigh
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from .base import Hamiltonian

class SSHModel(Hamiltonian):
    """
    Su-Schrieffer-Heeger (SSH) Model with Nearest-Neighbor (NN) hopping.
    """
    def __init__(self, N: int = 100, a: float = 2.0, delta: float = -0.15, 
                 V_A: float = 0.0):
        """
        Initialize the SSH Model.

        Args:
            N (int): Number of sites.
            a (float): Lattice constant.
            delta (float): Dimerization parameter.
            V_A (float): On-site potential strength (staggered V_A, -V_A).
        """
        super().__init__(N)
        
        self.a = a
        self.delta = delta
        self.V_A = V_A
        self.V_B = -V_A

        # Hopping amplitudes
        self.v = -np.exp(-(a - 2 * delta))
        self.w = -np.exp(-(a + 2 * delta))

        # Site positions
        j_indices = np.arange(1, N + 1)
        self.positions = (j_indices - (N + 1) / 2) * a - ((-1)**j_indices * delta)

        # Build static Hamiltonian 
        self.static_H = self.build_static_hamiltonian()

    def build_static_hamiltonian(self) -> csr_matrix:
        H = lil_matrix((self.N, self.N), dtype=float)

        # Intracell hopping (v)
        for i in range(self.N // 2):
            a_idx, b_idx = 2 * i, 2 * i + 1
            H[a_idx, b_idx] = H[b_idx, a_idx] = self.v

        # Intercell hopping (w)
        for i in range(self.N // 2 - 1):
            b_idx, a_next = 2 * i + 1, 2 * (i + 1)
            H[b_idx, a_next] = H[a_next, b_idx] = self.w
        
        return H.tocsr()

    def build_time_dependent_hamiltonian(self, t: float, electric_field_func) -> csr_matrix:
        """
        Construct H(t) = H_static + V(t).
        V(t) includes the staggered potential and the electric field potential.
        
        Args:
            t (float): Time.
            electric_field_func (callable): Function E(t) or E(x, t).
        """
        # Determine potential term
        # If val is scalar (time-only), it's E(t)*x
        # If val is array (space-time), it's integral E(x,t) dx
        
        try:
            val = electric_field_func(t)
        except TypeError:
            val = electric_field_func(self.positions, t)
            
        diag = np.array([self.V_A if i % 2 == 0 else self.V_B for i in range(self.N)], dtype=np.float64)
        
        if np.isscalar(val) or val.shape == ():
            diag += val * self.positions
        else:
            x = self.positions
            dx = np.diff(x)
            dx = np.append(dx, self.a + 2*self.delta)
            
            potential = np.cumsum(val * dx)
            diag += potential
            
        return self.static_H + diags(diag, 0, format='csr')

    def _get_lowest_parity_edge_state(self):
        """Resolve the central pair and select the occupied parity analytically.

        For this inversion-symmetric chain with negative NN hoppings, the
        lower central state is odd for N = 4m and even for N = 4m + 2.
        This avoids sorting numerically unresolved edge energies; see
        Rutkevich, doi:10.1155/2018/9784091, Eq. (35).
        Only call this for an inversion-symmetric (V_A == 0) chain.
        """
        H0 = self.build_time_dependent_hamiltonian(0, lambda t: 0.0)
        
        N_occ = self.N // 2
        if N_occ + 1 >= self.N:
            eigvals, eigvecs = eigh(H0.toarray())
        else:
            eigvals, eigvecs = eigsh(H0, k=N_occ + 1, which='SA')
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        psi1 = eigvecs[:, N_occ - 1]
        psi2 = eigvecs[:, N_occ]

        subspace = np.stack([psi1, psi2], axis=1)
        P_sub = subspace.T.conj() @ subspace[::-1]
        parities, Vp = np.linalg.eigh(P_sub)
        
        parity_states = subspace @ Vp
        parity_states /= np.linalg.norm(parity_states, axis=0)
        target_parity = -1 if self.N % 4 == 0 else 1
        return parity_states[:, np.argmin(np.abs(parities - target_parity))]

    def get_ground_state(self) -> np.ndarray:
        """Return the N/2 lowest-energy occupied orbitals, ordered by energy.

        For inversion-symmetric topological chains, resolve the central pair
        into parity eigenstates and select the lower central branch by chain
        length: odd for N = 4m, even for N = 4m + 2. This remains deterministic
        when the edge-energy splitting is below floating-point resolution.
        """
        # 1. Build H0
        H0 = self.build_time_dependent_hamiltonian(0, lambda t: 0.0)
        N_occ = self.N // 2
        
        vals, vecs = eigsh(H0, k=N_occ, which='SA')
        order = np.argsort(vals)
        psi = vecs[:, order]
        
        # 2. Check Topological Condition
        if self.V_A == 0.0 and self.delta < 0:
            psi[:, -1] = self._get_lowest_parity_edge_state()
            
        return psi
