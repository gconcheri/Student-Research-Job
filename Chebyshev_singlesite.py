import numpy as np
import matplotlib.pyplot as plt

# Chebyshev Lobatto grid points
c_a_N = lambda alpha, N: 0.5*(np.cos(np.pi*alpha/N)+1)

# Here we construct the Chebyshev polynomials from trigonometric functions
# See 'Cardinal functions' in https://deepblue.lib.umich.edu/handle/2027.42/29694
def C_Fourier_0_N(x, N):
    result = np.zeros(x.shape)
    x = np.mod(x + np.pi, 2*np.pi) - np.pi # comment out this line to see anomalous behavior
    result[np.abs(x)> 1e-16] = np.sin(N*x[np.abs(x)>1e-16])/np.tan(x[np.abs(x)>1e-16]/2)/(2*N)
    result[np.abs(x)<=1e-16] = 1.
    return result
C_Fourier_j_N = lambda x, j, N: C_Fourier_0_N(x-j*np.pi/N, N) #this takes values from -pi to 2pi
C_Chebysh_j_N = lambda x, j, N: (C_Fourier_j_N(np.arccos(x), j, N) + C_Fourier_j_N(np.arccos(x), -j, N))/(1 + (j==0 or j==N)) #this takes values from 0 to pi
# The following P_a_N are the Chebyshev polynomials used in arxiv:2311.12554
P_a_N = lambda x, alpha, N: C_Chebysh_j_N(2*x-1, alpha, N) #this takes values from -1 to 1


# define function that constructs MPS tensors of Chebyshev interpolation
def Chebyshev_interpolation(func, L, chi):
    N = chi-1
    args = 0.5 * (np.arange(2)[:, None] + c_a_N(np.arange(N+1), N)[None, :]) # see Eqs. (4.1) and (4.2)
    As = []
    # construct (data-dependent) left tensor
    A = func(args)[None] # (1, 2, chi)
    As.append(A)
    # construct (data-independent) bulk tensors from Chebyshev polynomials
    for i in range(1, L-1):
        A = np.array([P_a_N(args, alpha, N) for alpha in range(N+1)]) # (chi, 2, chi)
        As.append(A)
    # construct (data-independent) final tensor from Chebyshev polynomials
    A = np.array([P_a_N(np.arange(2)/2, alpha, N) for alpha in range(N+1)])[:, :, None] # (chi, 2, 1)
    As.append(A)
    return As

def interpolate(As):
    func_interp = np.squeeze(As[0])
    for A in As[1:]:
        func_interp = np.einsum('ia, ajb -> ijb', func_interp, A)
        func_interp = func_interp.reshape(-1, A.shape[-1])
    func_interp = np.squeeze(func_interp)
    return func_interp