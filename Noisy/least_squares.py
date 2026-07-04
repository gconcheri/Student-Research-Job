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
        
        # --- PARSE dic_y FOR TERM II (Store FULL dataset safely) ---
        self.N_total = len(dic_y)
        self.idx_grid_full = np.array(list(dic_y.keys()), dtype=np.int32)
        self.y_targets_full = np.array(list(dic_y.values()), dtype=complex)
        
        # Initialize Term II environment lists (Just empty shells for now!)
        self.L_phi_unc = [None] * self.L
        self.R_phi_unc = [None] * self.L
        self.L_phi_conj = [None] * self.L
        self.R_phi_conj = [None] * self.L

        # --- TERM I INITIALIZATION ---
        self.LPs = [None] * self.L
        self.RPs = [None] * self.L

        LP = np.zeros([1,1,1,1], dtype="float")
        RP = np.zeros([1,1,1,1], dtype="float")
        LP[0, 0, 0, 0] = 1
        RP[0, 0, 0, 0] = 1

        self.LPs[0] = LP
        self.RPs[-1] = RP

        # Build Term I environments
        for i in range(self.L - 1, 0, -1):
            self.update_R1(i)

        # --- THIS DOES ALL THE MISSING WORK ---
        # It defines self.N, builds the np.ones() boundaries, and runs update_phir!
        self._prepare_batch(batch_size=None)

    def optimize(self, method='als', lambda_=1e-10, lr=0.01, batch_size=None, max_sweeps=100, tol=1e-5):
        """
        Runs sweeps until the total loss converges.
        """
        print(f"\n--- Starting Optimization | Method: {method.upper()} | Lambda: {lambda_} ---")

        prev_loss = np.inf
        
        for k in range(max_sweeps):
            # 1. Perform one full forward and backward sweep
            self.sweep(method=method, lambda_=lambda_, lr=lr, batch_size=batch_size)
            
            # 2. Calculate the global loss (Term II + lambda * Term I)
            # (You evaluate this using your left/right environments at the boundary)
            current_loss = self.compute_total_loss(lambda_) 
            
            # 3. Check for convergence
            relative_change = abs(current_loss - prev_loss) / abs(prev_loss)
            
            print(f"Sweep {k+1}/{max_sweeps} | Loss: {current_loss:.6e} | Rel Change: {relative_change:.2e}")
            
            if relative_change < tol:
                print(f"Converged after {k+1} sweeps.")
                return current_loss
                
            prev_loss = current_loss
        
        print(f"-> Warning: Reached max_sweeps ({max_sweeps}) without fully converging.")
        return current_loss

    def compute_total_loss(self, lambda_):
        """Computes the total loss as L = lambda * Term I + Term II + sum_j y_j^2 / N"""
        
        # --- TERM I ---
        # Direct trace of the adjacent left and right environments.
        # (Taking .real prevents Python from logging "+ 0.000j")
        term_I = np.einsum('abcd, abcd ->', self.LPs[1], self.RPs[0]).real  

        # --- TERM II ---
        # Evaluate at site 0 because it is the orthogonality center at the end of a full sweep
        i = 0 
        A_m = self.A2_m(i)
        b = self.build_b(i)
        m = self.As[i]

        # 1. <m | A_eff | m>  -> Equivalent to <psi | psi>
        m_A_m = np.vdot(m, A_m).real
        
        # 2. <m | b> + <b | m> -> Equivalent to <psi | y> + <y | psi>
        # Since <b | m> is the complex conjugate of <m | b>, we just take 2 * real part
        m_b = np.vdot(m, b)
        cross_terms = 2.0 * m_b.real
        
        # 3. <y | y> -> The missing constant target offset!
        y_y = np.sum(np.abs(self.y_targets)**2) / self.N
        
        # Assemble Term II
        term_II = m_A_m - cross_terms + y_y

        return (lambda_ * term_I) + term_II


    def _prepare_batch(self, batch_size):
        """Samples a random mini-batch and builds its initial right environments."""
        # 1. Select the random indices
        if batch_size is None or batch_size >= self.N_total:
            batch_indices = np.arange(self.N_total)
            self.N = self.N_total
        else:
            # Randomly select 'batch_size' indices without replacement
            batch_indices = np.random.choice(self.N_total, batch_size, replace=False)
            self.N = batch_size

        # 2. Set the active grid and targets for the current sweep
        self.idx_grid = self.idx_grid_full[batch_indices]
        self.y_targets = self.y_targets_full[batch_indices]

        # 3. Reset the boundary environments to match the new batch size (N)
        self.L_phi_unc[0] = np.ones((self.N, 1), dtype=complex)
        self.L_phi_conj[0] = np.ones((self.N, 1), dtype=complex)
        self.R_phi_unc[-1] = np.ones((self.N, 1), dtype=complex)
        self.R_phi_conj[-1] = np.ones((self.N, 1), dtype=complex)

        # 4. Build the new right environments right-to-left before the sweep starts
        for i in range(self.L - 1, 0, -1):
            self.update_phir(i)

    def sweep(self, method='als', lambda_=1e-10, lr=0.01, batch_size=None):
        """
        Sweeps across the tensor network.
        method: 'als' for direct linear solver, 'gd' for gradient descent.
        batch_size: Integer to randomly subsample Term II data. If None, uses all data.
        """
        # 1. Prepare the stochastic batch and its environments BEFORE sweeping
        self._prepare_batch(batch_size)

        # 2. Sweep from Left to Right
        for i in range(self.L - 1):
            self.update_bond_LR(i, method, lambda_, lr)
            
        # 3. Sweep from Right to Left
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

    def _find_new_tensor(self, i, method, lambda_, lr):
        """Finds the optimized tensor using the specified method."""
        if method == 'gd':
            A = self.As[i]
            grad_I = self.A1_m(i)
            b_tensor = self.build_b(i)

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
        Wc = np.transpose(Wc, [0,1,3,2]) # Transpose Wc to match the correct contraction order
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
        Wc = np.transpose(Wc, [0,1,3,2]) # Transpose Wc to match the correct contraction order

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
        Wc = np.transpose(Wc, [0,1,3,2]) # Transpose Wc to match the correct contraction order
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


    # def A2_m(self, i, m_input=None):
    #         """Calculates A_II * m as shown in the double-layer diagram."""
    #         # 1. Get the UNCONJUGATED environments and current tensor (Top Layer)
    #         L_unc = self.L_phi_unc[i]   # Shape: (N, alpha)
    #         R_unc = self.R_phi_unc[i]   # Shape: (N, beta)
    #         m = self.As[i] if m_input is None else m_input              # Shape: (alpha, d, beta)
    #         d_indices = self.idx_grid[:, i]
            
    #         # 2. Get the CONJUGATED environments (Bottom Layer)
    #         L_conj = self.L_phi_conj[i]
    #         R_conj = self.R_phi_conj[i]
            
    #         if i == 0:
    #             dim_y, alpha, d, beta = m.shape
    #             out_tensor = np.zeros((dim_y, alpha, d, beta), dtype=complex)
                
    #             for n in range(self.N):
    #                 idx = d_indices[n]
    #                 # Top Layer: Calculate prediction vector (size 11)
    #                 pred = np.einsum('a, yab, b -> y', L_unc[n], m[:, :, idx, :], R_unc[n])
    #                 # Bottom Layer: Multiply prediction with conjugated environments
    #                 out_tensor[:, :, idx, :] += np.einsum('y, a, b -> yab', pred, L_conj[n], R_conj[n])
                    
    #         else:
    #             alpha, d, beta = m.shape
    #             out_tensor = np.zeros((alpha, d, beta), dtype=complex)
                
    #             for n in range(self.N):
    #                 idx = d_indices[n]
    #                 # Top Layer: Calculate prediction vector using the 'y' dim in L_unc
    #                 pred = np.einsum('ya, ab, b -> y', L_unc[n], m[:, idx, :], R_unc[n])     

    #                 # Bottom Layer: Contract prediction with the 'y' dim in L_conj
    #                 out_tensor[:, idx, :] += np.einsum('y, ya, b -> ab', pred, L_conj[n], R_conj[n])
                    
    #         return out_tensor / self.N

    # def build_b(self, i):
    #     """Builds b = sum(y_j * Phi_j)"""
    #     L_conj = self.L_phi_conj[i]
    #     R_conj = self.R_phi_conj[i]
    #     d_indices = self.idx_grid[:, i]
        
    #     if i == 0:
    #         dim_y, alpha, d, beta = self.As[0].shape
    #         b_tensor = np.zeros((dim_y, alpha, d, beta), dtype=complex)
            
    #         for n in range(self.N):
    #             idx = d_indices[n]
    #             # Outer product of (target, left, right)
    #             b_tensor[:, :, idx, :] += np.einsum(
    #                 'y, a, b -> yab', 
    #                 self.y_targets[n], L_conj[n], R_conj[n]
    #             )
    #     else:
    #         alpha, d, beta = self.As[i].shape
    #         b_tensor = np.zeros((alpha, d, beta), dtype=complex)
            
    #         for n in range(self.N):
    #             idx = d_indices[n]
    #             # Contract the target with the 'y' dimension held in the left environment
    #             b_tensor[:, idx, :] += np.einsum(
    #                 'y, ya, b -> ab', 
    #                 self.y_targets[n], L_conj[n], R_conj[n]
    #             )
                    
    #     return b_tensor / self.N

        """
        To understand code below:
        Writing (slice(None), d_indices, slice(None)) is the programmatic way of
        writing [:, d_indices, :]. It tells NumPy: Leave the $a$ and $b$ dimensions
        completely alone. Pick up the entire $a \times b$ matrix intact, and only route
        it based on the middle index."""

    def A2_m(self, i, m_input=None):
        """Calculates A_II * m vectorized and optimally factored over N."""
        L_unc = self.L_phi_unc[i]   
        R_unc = self.R_phi_unc[i]   
        L_conj = self.L_phi_conj[i]
        R_conj = self.R_phi_conj[i]
        
        m = self.As[i] if m_input is None else m_input
        d_indices = self.idx_grid[:, i]
        
        if i == 0:
            dim_y, alpha, d, beta = m.shape
            
            # --- TOP LAYER ---
            # Slice active dimensions -> m_sliced shape: (y, a, N, b)
            m_sliced = m[:, :, d_indices, :]
            
            # SMART SPLIT: Contract b first, then a
            # m_R shape: (y, a, N)
            m_R = np.einsum('yanb, nb -> yan', m_sliced, R_unc)
            # pred shape: (N, y)
            pred = np.einsum('na, yan -> ny', L_unc, m_R)
            
            # --- BOTTOM LAYER ---
            # out_n shape: (N, y, a, b)
            out_n = np.einsum('ny, na, nb -> nyab', pred, L_conj, R_conj)
            
            # Scatter-add back into d dimension
            out_tensor = np.zeros((dim_y, alpha, d, beta), dtype=complex)
            np.add.at(
                out_tensor, 
                (slice(None), slice(None), d_indices, slice(None)), 
                np.transpose(out_n, (1, 2, 0, 3))
            )
                
        else:
            alpha, d, beta = m.shape
            
            # --- TOP LAYER ---
            # Slice active dimensions -> m_sliced shape: (a, N, b)
            m_sliced = m[:, d_indices, :]
            
            # SMART SPLIT: Contract b first, then a
            # m_R shape: (a, N)
            m_R = np.einsum('anb, nb -> an', m_sliced, R_unc)
            # pred shape: (N, y)
            pred = np.einsum('nya, an -> ny', L_unc, m_R)     

            # --- BOTTOM LAYER ---
            # SMART SPLIT: Contract y first, then form outer product with R
            # L_pred shape: (N, a)
            L_pred = np.einsum('ny, nya -> na', pred, L_conj)
            # out_n shape: (N, a, b)
            out_n = np.einsum('na, nb -> nab', L_pred, R_conj)
            
            # Scatter-add back into d dimension
            out_tensor = np.zeros((alpha, d, beta), dtype=complex)
            np.add.at(
                out_tensor, 
                (slice(None), d_indices, slice(None)), 
                np.transpose(out_n, (1, 0, 2))
            )
                
        return out_tensor / self.N
    
    def build_b(self, i):
        """Builds b = sum(y_j * Phi_j) vectorized over N."""
        L_conj = self.L_phi_conj[i]
        R_conj = self.R_phi_conj[i]
        d_indices = self.idx_grid[:, i]
        
        if i == 0:
            dim_y, alpha, d, beta = self.As[0].shape
            
            # Pure outer product for all N. No intermediate contractions possible.
            # Shape: (N, y, a, b)
            b_n = np.einsum('ny, na, nb -> nyab', self.y_targets, L_conj, R_conj)
            
            # Scatter-add into the active d_indices
            b_tensor = np.zeros((dim_y, alpha, d, beta), dtype=complex)
            np.add.at(
                b_tensor, 
                (slice(None), slice(None), d_indices, slice(None)), 
                np.transpose(b_n, (1, 2, 0, 3)) # Align N with the d_indices axis -> (y, a, N, b)
            )
            
        else:
            alpha, d, beta = self.As[i].shape
            
            # SMART SPLIT: Contract y first, then form the outer product with R
            # 1. L_target shape: (N, a)
            L_target = np.einsum('ny, nya -> na', self.y_targets, L_conj)
            # 2. b_n shape: (N, a, b)
            b_n = np.einsum('na, nb -> nab', L_target, R_conj)
            
            # Scatter-add into the active d_indices
            b_tensor = np.zeros((alpha, d, beta), dtype=complex)
            np.add.at(
                b_tensor, 
                (slice(None), d_indices, slice(None)), 
                np.transpose(b_n, (1, 0, 2)) # Align N with the d_indices axis -> (a, N, b)
            )
                    
        return b_tensor / self.N