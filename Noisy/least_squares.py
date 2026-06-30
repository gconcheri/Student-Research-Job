"""Code implementing the least squares optimization algorithm."""


import numpy as np

class Engine(object):
    """Least squares optimization algorithm, implemented as class holding the necessary data.

    Attributes:
        As (list of np.ndarray): List of MPS tensors representing the state.
        Ws (list of np.ndarray): List of MPO tensors representing the operator.
        dic_y (dict): Dictionary containing the evaluated theoretical function values
        of the type {site_index: function_value}.
    """

    def __init__(self, As, Ws, dic_y, ):

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

        # initialize necessary RPs
        for i in range(self.L- 1, 1, -1):
            self.update_R1(i)

        # --- PARSE dic_y FOR TERM II ---
        self.N = len(dic_y)
        # Convert keys (tuples) into an (N, L) array of integers
        self.idx_grid = np.array(list(dic_y.keys()), dtype=np.int32)
        # Convert values (arrays) into an (N, 11) array of complex targets
        self.y_targets = np.array(list(dic_y.values()), dtype=complex)
        
        # Initialize Term II environments (phis)
        self.L_phis = [None] * self.L
        self.R_phis = [None] * self.L
        
        # Leftmost L_phi is just 1s of shape (N, alpha) where alpha=1
        self.L_phis[0] = np.ones((self.N, 1), dtype=complex)
        # Rightmost R_phi is just 1s of shape (N, beta) where beta=1
        self.R_phis[-1] = np.ones((self.N, 1), dtype=complex)
        
        # Initialize necessary R_phis for Term II
        for i in range(self.L - 1, 1, -1):
            self.update_phir(i)


    def sweep(self):
        # sweep from left to right
        for i in range(self.L - 2):
            self.update_bond(i)
        # sweep from right to left
        for i in range(self.L - 2, 0, -1):
            self.update_bond(i)

    def update_bond(self, i, lambda_=1e-10):
        j = i + 1
        A1_m = self.A1_m(i)
        A = self.As[i]
        Anew = A - lambda_ * A1_m
        self.As[i] = Anew
        self.update_L1(i)
        self.update_R1(j)


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
        if i==1:
            L1 = np.einsum('abcd, befg -> agefcd', L1, W)
            L1 = np.einsum('agefcd, chif -> agehid', L1, Wc)
            L1 = np.einsum('agehid, oagl -> olehid', L1, A)
            L1 = np.einsum('olehid, idor -> lehr', L1, Ac)
        elif i>1:
            L1 = np.einsum('abcd, aef -> febcd', L1, A)
            L1 = np.einsum('febcd, bghe -> fghcd', L1, W)
            L1 = np.einsum('fghcd, cilh -> fgild', L1, Wc)
            L1 = np.einsum('fgild, dlm -> fgim', L1, Ac)
        self.LPs[j] = L1

    def A1_m(self, i):
        """Calculate the effective operator for site `i`."""
        j = i + 1
        A = self.As[i]
        W = self.Ws[i]
        Wc = W.conj()
        R1 = self.RPs[j]
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
            A1_m = np.einsum('olehid, lehr -> oidr', L1, R1)
            A1_m = np.squeeze(A1_m)
        return A1_m

    def update_phir(self, i):
        """Calculate phi right of site `i` from phi right of site `i+1`."""
        j = i + 1
        ##MISSING
    
    def update_phil(self, i):
        """Calculate phi left of site `i` from phi left of site `i-1`."""
        j = i - 1
        ##MISSING

    def build_b(self, i):
        """Build the effective operator for the bond between sites `i` and `i+1`."""
        ##MISSING