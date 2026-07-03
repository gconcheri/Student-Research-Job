"""Code implementing the least squares optimization algorithm."""

import numpy as np
import scipy.sparse.linalg as spla

class Engine(object):
    """Least squares optimization algorithm, implemented as class holding the necessary data.

    Attributes:
        As (list of np.ndarray): List of MPS tensors representing the state.
        Ws (list of np.ndarray): List of MPO tensors representing the operator.
        dic_y (dict): Dictionary containing the evaluated theoretical function values
        of the type {site_index: function_value}.
    """

    def __init__(self, As, Ws, dic_y):

        self.L = len(As)
        self.As = As
        self.Ws = Ws
        self.dic_y = dic_y

        # initialize left and right environment for term I
        self.LPs = [None] * self.L
        self.RPs = [None] * self.L

        LP = np.zeros([1,1,1,1], dtype="float")  # vL wL* vL*
        RP = np.zeros([1,1,1,1], dtype="float")  # vR* wR* vR
        LP[0, 0, 0, 0] = 1
        RP[0, 0, 0, 0] = 1

        self.LPs[0] = LP
        self.RPs[-1] = RP


        # --- PARSE dic_y FOR TERM II ---
        self.N = len(dic_y)
        # Convert keys (tuples) into an (N, L) array of integers
        self.idx_grid = np.array(list(dic_y.keys()), dtype=np.int32)
        # Convert values (arrays) into an (N, 11) array of complex targets
        self.y_targets = np.array(list(dic_y.values()), dtype=complex)
        
        # Initialize Term II environments
        # unc = Top Layer (Predictions), conj = Bottom Layer (Pulling back)
        self.L_phi_unc = [None] * self.L
        self.R_phi_unc = [None] * self.L
        self.L_phi_conj = [None] * self.L
        self.R_phi_conj = [None] * self.L
        
        # Leftmost L_phis are just 1s of shape (N, alpha) where alpha=1
        self.L_phi_unc[0] = np.ones((self.N, 1), dtype=complex)
        self.L_phi_conj[0] = np.ones((self.N, 1), dtype=complex)

        # Rightmost R_phis are just 1s of shape (N, beta) where beta=1
        self.R_phi_unc[-1] = np.ones((self.N, 1), dtype=complex)
        self.R_phi_conj[-1] = np.ones((self.N, 1), dtype=complex)

        # initialize necessary RPs for Term I
        for i in range(self.L- 1, 0, -1):
            self.update_R1(i)

        # Initialize necessary R_phis for Term II
        for i in range(self.L - 1, 0, -1):
            self.update_phir(i)

    def _find_new_tensor(self, i, method, lambda_, lr):
        """Finds the optimized tensor using the specified method."""
        if method == 'gd':
            A = self.As[i]
            grad_I = self.A1_m(i)
            b_tensor = self.build_b(i)
            # In GD, gradient of ||Am - b||^2 is (Am - b)
            grad_II = self.A2_m(i) - b_tensor 
            
            # Take the gradient step
            m_new = A - lr * (lambda_ * grad_I + grad_II)
            return m_new
            
        elif method == 'als':
            # Call the GMRES solver we built earlier
            return self.solve_local_system(i, lambda_)
            
        else:
            raise ValueError("Optimization method must be 'gd' or 'als'")


    def solve_local_system(self, i, lambda_):
        """Solves the linear system A_eff * m = b for site i using GMRES."""
        shape = self.As[i].shape
        size = np.prod(shape)
        
        b_tensor = self.build_b(i)
        b_flat = b_tensor.flatten()
        
        def matvec(v):
            m_tensor = v.reshape(shape)
            res_I = self.A1_m(i, m_input=m_tensor)
            res_II = self.A2_m(i, m_input=m_tensor)
            return (lambda_ * res_I + res_II).flatten()
            
        A_eff = spla.LinearOperator((size, size), matvec=matvec, dtype=complex)
        m0 = self.As[i].flatten()
        
        m_new_flat, info = spla.gmres(A_eff, b_flat, x0=m0, rtol=1e-6)
        
        if info > 0:
            print(f"Warning: GMRES at site {i} did not converge. Info: {info}")
            
        return m_new_flat.reshape(shape)


    def sweep(self, method='als', lambda_=1e-10, lr=0.01):
        """
        Sweeps across the tensor network.
        method: 'als' for direct linear solver, 'gd' for gradient descent.
        """
        # Sweep from Left to Right
        for i in range(self.L - 1):
            self.update_bond_LR(i, method, lambda_, lr)
            
        # Sweep from Right to Left
        for i in range(self.L - 1, 0, -1):
            self.update_bond_RL(i, method, lambda_, lr)

    def update_bond_LR(self, i, method, lambda_, lr):
        """Update bond moving Left to Right and gauge fix via SVD."""
        # 1. Get the newly optimized tensor
        m_new = self._find_new_tensor(i, method, lambda_, lr)
        
        # 2. SVD to restore left-canonical form
        if i == 0:
            dim_y, alpha, d, beta = m_new.shape
            M_mat = m_new.reshape(dim_y * alpha * d, beta)
            U, S, Vh = np.linalg.svd(M_mat, full_matrices=False)
            self.As[i] = U.reshape(dim_y, alpha, d, U.shape[1])
        else:
            alpha, d, beta = m_new.shape
            M_mat = m_new.reshape(alpha * d, beta)
            U, S, Vh = np.linalg.svd(M_mat, full_matrices=False)
            self.As[i] = U.reshape(alpha, d, U.shape[1])
            
        # 3. Push the weight into the next site to the right
        rest = np.diag(S) @ Vh
        self.As[i+1] = np.einsum('ab, bcd -> acd', rest, self.As[i+1])
        
        # 4. Update Environments
        self.update_L1(i)
        self.update_phil(i)

    def update_bond_RL(self, i, method, lambda_, lr):
        """Update bond moving Right to Left and gauge fix via SVD."""
        # 1. Get the newly optimized tensor
        m_new = self._find_new_tensor(i, method, lambda_, lr)
        
        # 2. SVD to restore right-canonical form
        alpha, d, beta = m_new.shape
        M_mat = m_new.reshape(alpha, d * beta)
        U, S, Vh = np.linalg.svd(M_mat, full_matrices=False)
        
        self.As[i] = Vh.reshape(Vh.shape[0], d, beta)
        rest = U @ np.diag(S)
        
        # 3. Push the weight into the previous site to the left
        if i - 1 == 0:
            self.As[i-1] = np.einsum('yadb, bc -> yadc', self.As[i-1], rest)
        else:
            self.As[i-1] = np.einsum('adb, bc -> adc', self.As[i-1], rest)
            
        # 4. Update Environments
        self.update_R1(i)
        self.update_phir(i)

    def update_R1(self, i):
        """Calculate RP right of site `i-1` from RP right of site `i`.
        RPs range from 0 to L-1
        The leftmost RP, RPs[0], stops at A[1], W[1]..
        The rightmost RP, RPs[L-1], is dummy tensor
        """
        j = i - 1
        R1 = self.RPs[i]
        A = self.As[i]
        Ac = A.conj()
        W = self.Ws[i]
        Wc = W.conj()
        R1 = np.einsum('abc, cdef -> abdef', A, R1)
        R1 = np.einsum('cdlb, abdef -> aclef', W, R1)
        R1 = np.einsum('bedl, aclef -> acbdf', Wc, R1)
        R1 = np.einsum('edf, acbdf -> acbe', Ac, R1)
        self.RPs[j] = R1


    def update_L1(self, i):
        """Calculate LP left of site `i+1` from LP left of site `i`
        LPs range from 0 to L-1
        The rightmost LP, LPs[L-1], stops at A[L-2], W[L-2]..
        The leftmost LP, LPs[0], is dummy tensor (defined above)
        The second LP, LPs[1], stops at A[0], W[0].. and is calculated differently than the others
        """
        j = i + 1
        L1 = self.LPs[i]
        A = self.As[i]
        Ac = A.conj()
        W = self.Ws[i]
        Wc = W.conj()
        if i==0:
            L1 = np.einsum('abcd, befg -> agefcd', L1, W)
            L1 = np.einsum('agefcd, chif -> agehid', L1, Wc)
            L1 = np.einsum('agehid, oagl -> olehid', L1, A)
            L1 = np.einsum('olehid, odir -> lehr', L1, Ac)
        elif i>0:
            L1 = np.einsum('abcd, aef -> febcd', L1, A)
            L1 = np.einsum('febcd, bghe -> fghcd', L1, W)
            L1 = np.einsum('fghcd, cilh -> fgild', L1, Wc)
            L1 = np.einsum('fgild, dlm -> fgim', L1, Ac)
        self.LPs[j] = L1

    def A1_m(self, i, m_input=None):
        """Calculate the effective operator for site `i`."""
        A = self.As[i] if m_input is None else m_input
        W = self.Ws[i]
        Wc = W.conj()
        R1 = self.RPs[i]
        L1 = self.LPs[i]
        if i>0:
            L1 = np.einsum('abcd, aef -> febcd', L1, A)
            L1 = np.einsum('febcd, bghe -> fghcd', L1, W)
            L1 = np.einsum('fghcd, cilh -> fgild', L1, Wc)
            A1_m = np.einsum('abcde, abcf -> edf', L1, R1)
        else:
            L1 = np.einsum('abcd, befg -> agefcd', L1, W)
            L1 = np.einsum('agefcd, chif -> agehid', L1, Wc)
            L1 = np.einsum('agehid, oagl -> olehid', L1, A)
            A1_m = np.einsum('olehid, lehr -> odir', L1, R1)
            # A1_m = np.squeeze(A1_m)
        return A1_m

    def update_phil(self, i):
        """Calculate phi left of site `i+1` from phi left of site `i`."""
        j = i + 1
        d_indices = self.idx_grid[:, i]
        
        L_unc = self.L_phi_unc[i]
        L_conj = self.L_phi_conj[i]
        A = self.As[i]
        Ac = A.conj()
        
        if i == 0:
            # A has shape (11, alpha, d, beta) -> let's call 11 the 'y' dimension
            A_sliced = A[:, :, d_indices, :]   # Shape: (y, alpha, N, beta)
            Ac_sliced = Ac[:, :, d_indices, :] 
            
            # L_phi goes from shape (N, alpha) to (N, y, beta)
            # We keep the 'y' dimension open!
            self.L_phi_unc[j] = np.einsum('na, yanb -> nyb', L_unc, A_sliced)
            self.L_phi_conj[j] = np.einsum('na, yanb -> nyb', L_conj, Ac_sliced)
            
        else:
            # A has shape (alpha, d, beta)
            A_sliced = A[:, d_indices, :]   # Shape: (alpha, N, beta)
            Ac_sliced = Ac[:, d_indices, :]
            
            # L_phi has shape (N, y, alpha). It contracts with A, keeping 'y' open.
            self.L_phi_unc[j] = np.einsum('nya, anb -> nyb', L_unc, A_sliced)
            self.L_phi_conj[j] = np.einsum('nya, anb -> nyb', L_conj, Ac_sliced)


    def update_phir(self, i):
        """Calculate phi right of site `i-1` from phi right of site `i`."""
        j = i - 1
        d_indices = self.idx_grid[:, i]
        
        A = self.As[i]
        Ac = A.conj()
        
        # 1. Top Layer (Unconjugated)
        R_unc = self.R_phi_unc[i]
        self.R_phi_unc[j] = np.einsum('nb, anb -> na', R_unc, A[:, d_indices, :])
        
        # 2. Bottom Layer (Conjugated)
        R_conj = self.R_phi_conj[i]
        self.R_phi_conj[j] = np.einsum('nb, anb -> na', R_conj, Ac[:, d_indices, :])


    def A2_m(self, i, m_input=None):
            """Calculates A_II * m as shown in the double-layer diagram."""
            # 1. Get the UNCONJUGATED environments and current tensor (Top Layer)
            L_unc = self.L_phi_unc[i]   # Shape: (N, alpha)
            R_unc = self.R_phi_unc[i]   # Shape: (N, beta)
            m = self.As[i] if m_input is None else m_input              # Shape: (alpha, d, beta)
            d_indices = self.idx_grid[:, i]
            
            # 2. Get the CONJUGATED environments (Bottom Layer)
            L_conj = self.L_phi_conj[i]
            R_conj = self.R_phi_conj[i]
            
            if i == 0:
                dim_y, alpha, d, beta = m.shape
                out_tensor = np.zeros((dim_y, alpha, d, beta), dtype=complex)
                
                for n in range(self.N):
                    idx = d_indices[n]
                    # Top Layer: Calculate prediction vector (size 11)
                    pred = np.einsum('a, yab, b -> y', L_unc[n], m[:, :, idx, :], R_unc[n])
                    # Bottom Layer: Multiply prediction with conjugated environments
                    out_tensor[:, :, idx, :] += np.einsum('y, a, b -> yab', pred, L_conj[n], R_conj[n])
                    
            else:
                alpha, d, beta = m.shape
                out_tensor = np.zeros((alpha, d, beta), dtype=complex)
                
                for n in range(self.N):
                    idx = d_indices[n]
                    # Top Layer: Calculate prediction vector using the 'y' dim in L_unc
                    pred = np.einsum('ya, ab, b -> y', L_unc[n], m[:, idx, :], R_unc[n])     

                    # Bottom Layer: Contract prediction with the 'y' dim in L_conj
                    out_tensor[:, idx, :] += np.einsum('y, ya, b -> ab', pred, L_conj[n], R_conj[n])
                    
            return out_tensor / self.N

    def build_b(self, i):
        """Builds b = sum(y_j * Phi_j)"""
        L_conj = self.L_phi_conj[i]
        R_conj = self.R_phi_conj[i]
        d_indices = self.idx_grid[:, i]
        
        if i == 0:
            dim_y, alpha, d, beta = self.As[0].shape
            b_tensor = np.zeros((dim_y, alpha, d, beta), dtype=complex)
            
            for n in range(self.N):
                idx = d_indices[n]
                # Outer product of (target, left, right)
                b_tensor[:, :, idx, :] += np.einsum(
                    'y, a, b -> yab', 
                    self.y_targets[n], L_conj[n], R_conj[n]
                )
        else:
            alpha, d, beta = self.As[i].shape
            b_tensor = np.zeros((alpha, d, beta), dtype=complex)
            
            for n in range(self.N):
                idx = d_indices[n]
                # Contract the target with the 'y' dimension held in the left environment
                b_tensor[:, idx, :] += np.einsum(
                    'y, ya, b -> ab', 
                    self.y_targets[n], L_conj[n], R_conj[n]
                )
                    
        return b_tensor / self.N