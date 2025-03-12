import numpy as np
from scipy.linalg.interpolative import interp_decomp
import matplotlib.pyplot as plt

# Use scipy's implementation of the interpolative decomposition
# Instead of the matrix cross interpolation M = C @ P^-1 @ R
# it factorizes as M = A @ P with A = M[:, idx]
def interpolative_decomposition(M, eps_or_k=1e-5, k_min=2):
    r = min(M.shape)
    if r <= k_min:
        k = r
        idx, proj = interp_decomp(M, eps_or_k=k) #eps_or_k = precision of decomposition
    elif isinstance(eps_or_k, int): #checks if eps is an integer
        k = min(r, eps_or_k)
        idx, proj = interp_decomp(M, eps_or_k=k)
    else:
        k, idx,  proj = interp_decomp(M, eps_or_k=eps_or_k)
        if k <= k_min:
            k = min(r, k_min) #is it not enough to put k = k_min? 
                              #r>k_min otherwise first condition would have been true
            idx, proj = interp_decomp(M, eps_or_k=k)
    A = M[:, idx[:k]]
    P = np.concatenate([np.eye(k), proj], axis=1)[:, np.argsort(idx)]
    return A, P, k, idx[:k]

# k is the 'compressed' rank = number of pivot columns
# idx is the array with entries the indeces of the pivot columns
# proj = matrix R s.t. M[:,idx[:k]]*R = M[:,idx[k:]] 
# P = matrix s.t.  M[:,idx[:k]]*P = M (approximated)


class function:  # certain function f(x) with x given as binary

    def __init__(self, f):
        self.cache = {}
        self.f = f #store function passed during instantiation
        self.numcacheused = 0
        self.numvals = 0
        self.unique = 0


    def __call__(self, *args, **kwds): #here args is a tuple (0,1,1,1,...)
        self.numvals+=1
        if args in self.cache:
            self.numcacheused+=1
            return self.cache[*args]
        else:
            self.unique+=1
            val = self.f(*args)
            self.cache[*args] = val
            return val
    
    def cache_size(self): #size of cache = number of current evaluations
        return len(self.cache) #return the number of entries in the cache

"""Now val defined in this class will be an array of shape (D,) with D number of points
in which correlation function is evaluated in space"""


def tensor_cross_interpolation(tensor, func_vals, D, L, d=2, eps_or_chi=1e-6, iters=6):
    #tensor must be function s.t. f(*il,σj,σj+1,*jr).shape = (D,) with D number of points in space
    # random initial choice for index sets
    idxs = np.random.choice(d, size=(L)) #array of L random numbers from 0 to d-1 - index sigma
    As = [None] * L
    I = [idxs[:j].reshape(1, -1) for j in range(L)] # creates list of I_l arrays
    J = [idxs[j:].reshape(1, -1) for j in range(1, L+1)] # list of J_l
    # sweep
    for i in range(iters):
        #print(f'Sweep: {i+1:d}.')
        As, I = left_to_right_sweep(tensor, As, I, J, L, d, D, eps_or_chi)
        As, J = right_to_left_sweep(tensor, As, I, J, L, d, D, eps_or_chi)
    #in theory, at the end of these sweeps the first tensor of As should be the only one with the additional leg of dim D

    #func_interp = np.squeeze(As[0]) removes any singleton dimensions (dimensions of size 1).
    # we now have func_interp.shape = (d,chir,D) - I think this is wrong now

    func_interp = np.squeeze(As[-1]) #now has two legs: ab
    for A in As[-2:0:-1]:
        func_interp = np.einsum('idk, kj -> idj', A, func_interp)
        func_interp = func_interp.reshape(A.shape[0], -1)
    if As[0].shape[0] == D:
        #print(As[0].shape)
        func_interp = np.einsum('dak, kj -> daj', np.squeeze(As[0]), func_interp)
        #func_interp = np.einsum('ak, kj -> aj', np.squeeze(As[0]), func_interp) #uncomment this in case D = 1
        func_interp = np.transpose(func_interp, [1,2,0])
        #print("func interp shape = ", func_interp.shape)
        
        func_interp = func_interp.reshape(-1, D)
        #func_interp = func_interp.reshape(-1) #uncomment this in case D = 1
    else: 
        return print("Error")

    difference = func_vals-func_interp #should be difference between 2 matrices
    err_max = np.max(np.abs(difference))/np.max(np.abs(func_vals))
    err_2 = np.linalg.norm(difference)/np.linalg.norm(func_vals)
    evals = tensor.cache_size() * D

    print('err_max: ', err_max)
    print('err_2: ', err_2)
    print("eval/D: ", tensor.cache_size())
    print()
    print('repeated evaluations: ', tensor.numcacheused)
    print('unique evaluations', tensor.unique)
    print('unique + repeated: ', tensor.numcacheused + tensor.unique)
    print('total evaluations: ', tensor.numvals)
    print()

    return As, J, evals, err_2, err_max, func_interp

def left_to_right_sweep(tensor, As, I, J, L, d, D, eps_or_chi):
    # sweep left to right
    for bond in range(L-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape #chil = number of rows in array I_l: number of combinations (σ1,..,σl)
        chir, _ = J[bond+1].shape #which corresponds to number of "points" on which I am evaluating function
        
        Pi = np.zeros((D, chil, d, d, chir), dtype=np.complex128)
        for il in range(chil):
            for s1 in range(d):
                for s2 in range(d):
                    for jr in range(chir):
                        val = tensor(*I[bond][il,:],s1,s2,*J[bond+1][jr,:])
                        Pi[:,il,s1,s2,jr] = val
                        #print(val.shape)
        #print("Pi.shape = ", Pi.shape)
        Pi = np.transpose(Pi, [1,2,3,4,0])
        Pi = Pi.reshape(chil * d, d * chir * D)
        # decompose using interpolative decomposition:
        # Pi = P^T @ A^T, A^T = Pi[idx,:]
        A, P, k, idx = interpolative_decomposition(Pi.T, eps_or_k=eps_or_chi)
        #print("A.shape " ,A.shape)
        #print("P.shape ", P.shape)
        # update indices using idxs c I[bond] x {0, 1, ..., d-1}
        I[bond+1] = np.array([np.append(I[bond][i//d], [i%d]) for i in idx])
        # update tensors
        As[bond] = P.T.reshape(chil, d, k)
        As[bond+1] = A.T.reshape(k, d, chir, D)
    return As, I

def right_to_left_sweep(tensor, As, I, J, L, d, D, eps_or_chi):
    # sweep right to left
    for bond in range(L-2,-1,-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape
        chir, _ = J[bond+1].shape

        Pi = np.zeros((D, chil, d, d, chir), dtype=np.complex128)
        for il in range(chil):
            for s1 in range(d):
                for s2 in range(d):
                    for jr in range(chir):
                        val = tensor(*I[bond][il,:],s1,s2,*J[bond+1][jr,:])
                        #print(val.shape)
                        Pi[:,il,s1,s2,jr] = val
        Pi = Pi.reshape(D * chil * d, d * chir)
        # decompose using interpolative decomposition:
        # Pi = A @ P, A = Pi[:,idx]
        A, P, k, idx = interpolative_decomposition(Pi, eps_or_k=eps_or_chi)
        # update indices using idxs c {0, 1, ..., d-1} x J[bond+1]
        J[bond] = np.array([np.append([i//chir], J[bond+1][i%chir]) for i in idx])
        # update tensors
        As[bond] = A.reshape(D, chil, d, k)
        As[bond+1] = P.reshape(k, d, chir)
    return As, J