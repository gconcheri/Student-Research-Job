import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh, expm_multiply, expm

### Correlator generalized for L sites
def gen_spin_operators(L):
    """Returns the spin-1/2 operators sigma_x and sigma_z for L sites."""
    X = sparse.csr_array(np.array([[0.,1.],[1.,0.]]))
    Z = sparse.csr_array(np.diag([1.,-1.]))
    
    d = 2
    Sx_list = []
    Sz_list = []
    
    for i_site in range(L):
        # ops on first site
        if i_site == 0: 
            Sx = X
            Sz = Z 
        else: 
            Sx = sparse.csr_array(np.eye(d))
            Sz = sparse.csr_array(np.eye(d))
        # ops on remaining sites
        for j_site in range(1, L):
            if j_site == i_site: 
                Sx = sparse.kron(Sx, X, 'csr')
                Sz = sparse.kron(Sz, Z, 'csr')
            else:
                Sx = sparse.kron(Sx, np.eye(d), 'csr')
                Sz = sparse.kron(Sz, np.eye(d), 'csr')
        Sx_list.append(Sx)
        Sz_list.append(Sz)
    
    return Sx_list, Sz_list

def gen_hamiltonian_terms(L, Sx_list, Sz_list):
    """Generates the XX, ZZ, X and Z terms of the Hamiltonian."""
    D = Sx_list[0].shape[0]
    #print(f'System with {L:d} sites, Hilbert space dimension is {D:d}.')

    # Ising interaction
    Hxx = Sx_list[0] @ Sx_list[1]
    for i in range(1, L-1):
        Hxx += Sx_list[i] @ Sx_list[i+1]

    Hzz = Sz_list[0] @ Sz_list[1]
    for i in range(1, L-1):
        Hzz += Sz_list[i] @ Sz_list[i+1]

    Hx = Sx_list[0]
    for Sx in Sx_list[1:L]:
        Hx += Sx

    # onsite field terms
    Hz = Sz_list[0]
    for Sz in Sz_list[1:L]:
        Hz += Sz
    
    return Hxx, Hzz, Hx, Hz
# define Hamiltonian terms

def gen_Ham(L = 11, model = 1, h = 10**(-2.5), k = 1.):

    Sx_list, Sz_list = gen_spin_operators(L)
    Hxx, Hzz, Hx, Hz = gen_hamiltonian_terms(L, Sx_list, Sz_list)

    g = 2.
    J = 1.
    H = -J * Hxx -g * Hz

    if model == 2:
            H = H - k * Hzz
    elif model == 3 or model ==4:
        g = 0.5
        H = H + h*Hx
        if model ==4:
            k = 0.5
            H = H - k *Hzz

    return H


def correlator(H, L = 11., n = 10, dt = 1e-2):

    Sx_list, Sz_list = gen_spin_operators(L)

        
    # compute correlator
    N = 2**n

    # get ground state
    E, psi = eigsh(H, k=1, which='SA')
    E0, psi = np.squeeze(E), np.squeeze(psi)
    # print('Ground state energy:', E0)
    psi_0 = psi.copy()

    # put in excitation
    psi = Sx_list[L//2] @ psi
    E1 = np.dot(psi.conj(), H @ psi).real
    #print('Excited state energy:', E1)

    psil = np.array([Sx_list[l] @ psi_0 for l in range(L)])


    # evolve states in time
    psis = expm_multiply(-1j * H,
                        psi,
                        start=0,
                        stop=N*dt,
                        num=N,
                        endpoint=False)

    # calculate correlators C = <psi| e^iHt X_ell e^-iHt X_L/2 |psi>
    Cs = np.einsum('lj, ij -> li', psil.conj(), psis) * np.exp(1j * E0 * np.arange(N) * dt)
    #form is (L,2**n) <-> (X,T)
    # plt.figure(figsize=(8, 4))
    # plt.imshow(Cs.real, aspect = 'auto', 
    #         interpolation = 'none'
    #         )

    # plt.colorbar()
    # plt.title("correlation function evaluated on " + f"{L}" + " sites")
    
    return Cs


def correlator_Chebyshev(D_list, t_matrix, H, dt= 1e-2, n=10):
    N = 2**n
    D = len(D_list)
    Sx_list, Sz_list = gen_spin_operators(D)


    # get ground state
    E, psi = eigsh(H, k=1, which='SA')
    E0, psi = np.squeeze(E), np.squeeze(psi)
    psi_0 = psi.copy()

    # put in excitation
    psi = Sx_list[D//2] @ psi
    E1 = np.dot(psi.conj(), H @ psi).real

    psil = np.array([Sx_list[l] @ psi_0 for l in range(D)])

    a,b = t_matrix.shape

    #print(a, b)

    t = t_matrix.reshape(-1)*N*dt
    psis = np.zeros((a*b,psi.shape[0]), dtype=np.complex128)
    for i,tt in enumerate(t):
        psis[i,:] = expm_multiply(-1j * H * tt, psi)
    corr = np.einsum('lj, ij -> li', psil.conj(), psis) * np.exp(1j * E0 * t)
    corr = corr.reshape(D, a, b)

    return corr