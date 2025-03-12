import numpy as np
import matplotlib.pyplot as plt

# %%
def find_ID_coeffs_via_iterated_leastsq(A, J, notJ=None):
    """
    Given a set of columns A[:, J], finds a matrix Z such that
    approximately A = A[:, J] @ Z by iteratively solving for
    each column A[:, J] @ Z[:, j] = A[:, j] with least squares.
    """
    # number of columns
    _, N = A.shape
    # fill empty matrix with solutions
    Z = np.zeros((len(J), N), dtype=A.dtype)
    # set columns J to identity in Z
    Z[np.arange(len(J)), J] = 1.
    # for the rest solve A[:, J] @ x = A[:, j] for x, then Z[:, j] = x
    if notJ is None:
        notJ = list(set(range(N)) - set(J))
    x = np.linalg.lstsq(A[:, J], A[:, notJ], rcond=None)[0] # x.shape = (len(J), N)
    Z[:, notJ] = x
    return Z

#%%
class function:  # certain function f(x) with x given as binary

    def __init__(self, f):
        self.cache = {}
        self.f = f #store function passed during instantiation
        self.numcacheused = 0
        self.numvals = 0
        self.unique = 0


    def __call__(self, *args, **kwds):
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

#%%
def accumulative_tensor_cross_interpolation(tensor, func_vals, D, L, threshold = 10**(-14), d=2, euclidean = True):
    #tensor must be function s.t. f(*il,σj,σj+1,*jr).shape = (D,) with D number of points in space
    
    # initial choice is all zeros
    idxs = np.zeros((L,), dtype=np.int32)

    #idxs = np.random.choice(d, size=(L)) #array of L random numbers from 0 to d-1 - index sigma

    dtype = np.complex128

    As = [None] * L

    J = [idxs[j:].reshape(1, -1) for j in range(1, L+1)]
    I = [idxs[:j].reshape(1, -1) for j in range(L)]

    error = 1
    ii=0
    # sweep
    while error > threshold:
        #print(f'Sweep: {i+1:d}.')
        ii+=1
        As, I = left_to_right_sweep(tensor, As, I, J, L, d, D, dtype, euclidean)
        func_interp = interpolation(As, D, righttoleft = False)
        difference = func_vals-func_interp #should be difference between 2 matrices
        error = np.linalg.norm(difference)/np.linalg.norm(func_vals)
        #print("iter = ", ii, " error = ", error)
        if error > threshold:
            ii+=1
            As, J = right_to_left_sweep(tensor, As, I, J, L, d, D, dtype, euclidean)
            func_interp = interpolation(As, D)
            difference = func_vals-func_interp #should be difference between 2 matrices
            error = np.linalg.norm(difference)/np.linalg.norm(func_vals)
            #print("iter = ", ii, " error = ", error)



    #in theory, at the end of these sweeps the first tensor of As should be the only one with the additional leg of dim D

    err_max = np.max(np.abs(difference))/np.max(np.abs(func_vals))

    print('err_max: ', err_max)
    print('err_2: ', error)
    print('unique evaluations', tensor.unique)


    return tensor.unique, error, err_max

def interpolation(As, D, righttoleft = True):
    if righttoleft == True:
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
            return func_interp
        else: 
            return print("Error")
    else: 
        func_interp = np.squeeze(As[0]) #now has two legs: ab
        for A in As[1:-1]:
            func_interp = np.einsum('di, ikj -> dkj', func_interp, A)
            func_interp = func_interp.reshape(-1, A.shape[-1])
        if As[-1].shape[-1] == D:
            func_interp = np.einsum('dk, kjl -> djl', func_interp, np.squeeze(As[-1])) #func_interp dim: a, chir, D
            #here we probably have to insert a transpose
            func_interp = func_interp.reshape(-1, D)
            return func_interp
        else:
            return print("Error")

def left_to_right_sweep(tensor, As, I, J, L, d, D, dtype, euclidean):
    # sweep left to right
    for bond in range(L-1):
        #print(bond)
        # construct local two-site tensor
        chil, _ = I[bond].shape #chil = number of rows in array I_l: number of combinations (σ1,..,σl)
        chir, _ = J[bond+1].shape #which corresponds to number of "points" on which I am evaluating function
        
        Pi = np.zeros((D, chil, d, d, chir), dtype=dtype)
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

        # all rows of Pi = I[bond] x {0, ..., d-1}
        all_idxs = np.concatenate(
                       [np.append(I[bond],
                                  s1 * np.ones((chil, 1), dtype=np.int32),
                                  axis=1)
                        for s1 in range(d)],
                       axis=0)
        # turn multi-indices I[bond+1] c I[bond] x {0, ..., d-1} into numbered indexes
        I2 = list(np.argmax((I[bond+1][:, None] == all_idxs[None]).all(axis=2),
                            axis=1))
        # update index set I2
        N = Pi.shape[0]
        notI2 = list(set(range(N)) - set(I2)) # get indices not in I2
        
        if len(I[bond+1]) == N:
            pass # don't add a row if set of rows is already maximal
        else:
            Z = find_ID_coeffs_via_iterated_leastsq(Pi.T, J=I2, notJ=notI2)
            Pi_tilde = Z.T @ Pi[I2, :]
            # compute errors and update index set
            if euclidean == True:
                errs = np.linalg.norm(Pi[notI2, :] - Pi_tilde[notI2, :], axis=1) # row errors
            else:
                errs = np.sum(np.abs(Pi[notI2, :] - Pi_tilde[notI2, :]), axis=1)
            ell = notI2[np.argmax(errs)]
            I2.append(ell)
            notI2.remove(ell)
            I[bond+1] = np.append(I[bond+1],
                                  all_idxs[None, ell],
                                  axis=0)
        # shift canonicality center by ID with updated index set
        #find Z s.t. Pi = Z^T @ Pi[I2,:]
        Z = find_ID_coeffs_via_iterated_leastsq(Pi.T, J=I2, notJ=notI2)
        chi = len(I2)
        # update tensors
        As[bond] = Z.T.reshape(chil, d, chi)
        As[bond+1] = Pi[I2, :].reshape(chi, d, chir, D)
        # # decompose using interpolative decomposition:
        # # Pi = P^T @ A^T, A^T = Pi[idx,:]
        # A, P, k, idx = interpolative_decomposition(Pi.T, eps_or_k=eps_or_chi)
        # #print("A.shape " ,A.shape)
        # #print("P.shape ", P.shape)
        # # update indices using idxs c I[bond] x {0, 1, ..., d-1}
        # I[bond+1] = np.array([np.append(I[bond][i//d], [i%d]) for i in idx])
        # # update tensors
        # As[bond] = P.T.reshape(chil, d, k)
        # As[bond+1] = A.T.reshape(k, d, chir, D)

    return As, I

def right_to_left_sweep(tensor, As, I, J, L, d, D, dtype, euclidean):
    # sweep right to left
    for bond in range(L-2,-1,-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape
        chir, _ = J[bond+1].shape

        Pi = np.zeros((D, chil, d, d, chir), dtype=dtype)
        for il in range(chil):
            for s1 in range(d):
                for s2 in range(d):
                    for jr in range(chir):
                        val = tensor(*I[bond][il,:],s1,s2,*J[bond+1][jr,:])
                        #print(val.shape)
                        Pi[:,il,s1,s2,jr] = val
        Pi = Pi.reshape(D * chil * d, d * chir)

        # all columns of Pi = {0, ..., d-1} x J[bond+1]
        all_idxs = np.concatenate(
                       [np.append(s2 * np.ones((chir, 1), dtype=np.int32),
                                  J[bond+1],
                                  axis=1)
                        for s2 in range(d)],
                       axis=0)
        # turn multi-indices J[bond] c {0, ..., d-1} x J[bond+1] into numbered indexes
        J1 = list(np.argmax((J[bond][:, None] == all_idxs[None]).all(axis=2),
                            axis=1))
        # update index set J1
        N = Pi.shape[1]
        notJ1 = list(set(range(N)) - set(J1)) # get indices not in J1
        if len(J[bond]) == N:
            pass # don't add a column if set of columns is already maximal
        else:
            # get current iteration for Pi_tilde from ID with columns in J1
            Z = find_ID_coeffs_via_iterated_leastsq(Pi, J=J1, notJ=notJ1)
            Pi_tilde = Pi[:, J1] @ Z            
            # compute errors and update index set
            if euclidean == True:
                errs = np.linalg.norm(Pi[:, notJ1] - Pi_tilde[:, notJ1], axis=0) # column errors
            else:
                errs = np.sum(np.abs(Pi[:, notJ1] - Pi_tilde[:, notJ1]), axis=0)
            ell = notJ1[np.argmax(errs)]
            J1.append(ell)
            notJ1.remove(ell)
            J[bond] = np.append(J[bond],
                                all_idxs[None, ell],
                                axis=0)
        # shift canonicality center by ID with updated index set
        Z = find_ID_coeffs_via_iterated_leastsq(Pi, J=J1, notJ=notJ1)
        chi = len(J1)
        # update tensors
        As[bond] = Pi[:, J1].reshape(D, chil, d, chi)
        As[bond+1] = Z.reshape(chi, d, chir)

        # # decompose using interpolative decomposition:
        # # Pi = A @ P, A = Pi[:,idx]
        # A, P, k, idx = interpolative_decomposition(Pi, eps_or_k=eps_or_chi)
        # # update indices using idxs c {0, 1, ..., d-1} x J[bond+1]
        # J[bond] = np.array([np.append([i//chir], J[bond+1][i%chir]) for i in idx])
        # # update tensors
        # As[bond] = A.reshape(D, chil, d, k)
        # As[bond+1] = P.reshape(k, d, chir)
    return As, J