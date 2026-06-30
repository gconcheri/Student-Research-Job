import numpy as np

def binary_addition_MPO(L, l=None, modular=False):
    r"""
    this adds two binary numbers k + l = j

    L = number of bits
    l =  An optional specific integer to add. If left as None, the function builds a general 3-leg MPO (where j, k, and l are all open variables)

    Construct an MPO for an operation of the form
        O_jkl = \sum_m <j|m + l><m|k>.
    If l is given this will be an MPO for this specific l.
    If l is None this will be a three-leg MPO for general l.
    Can be modular, i.e., j = 0, 1, ..., 2^n-1 computing (j + l) mod 2^n,
    or else j = 0, ..., 2^n-1-l computing j + l < 2^n.
    """
    
    # right edge
    Wright = np.zeros((2, 1, 2, 2, 2), dtype=np.float64) #left right, 3 physical legs
    Wright[0, 0, :, :, 0] = np.eye(2)
    Wright[0, 0, 1, 0, 1] = Wright[1, 0, 0, 1, 1] = 1.
    
    # bulk
    Wbulk = np.zeros((2, 2, 2, 2, 2), dtype=np.float64)
    Wbulk[0, 0, :, :, 0] = np.eye(2)
    Wbulk[0, 1, 1, 0, 0] = Wbulk[1, 1, 0, 1, 0] = 1.
    Wbulk[1, 1, :, :, 1] = np.eye(2)
    Wbulk[0, 0, 1, 0, 1] = Wbulk[1, 0, 0, 1, 1] = 1.
    
    # left edge
    Wleft = np.zeros((1, 2, 2, 2, 2), dtype=np.float64)
    if modular:
        Wleft[0, 0, :, :, 0] = np.eye(2)
        Wleft[0, 1, :, :, 0] = np.fliplr(np.eye(2))
        Wleft[0, 0, :, :, 1] = np.fliplr(np.eye(2))
        Wleft[0, 1, :, :, 1] = np.eye(2)
    else:
        Wleft[0, 0, :, :, 0] = np.eye(2)
        Wleft[0, 1, 1, 0, 0] = 1.
        Wleft[0, 0, 1, 0, 1] = 1.

    # combine to MPO
    if l is not None:
        if isinstance(l, int):
            assert l < 2**L
            l = np.array([(l//2**bit) % 2 for bit in range(L-1, -1, -1)])
        Ws = [Wleft[..., l[0]]]
        for bit in l[1:-1]:
            Ws.append(Wbulk[..., bit])
        Ws.append(Wright[..., l[-1]])
    else:
        Ws = [Wleft] + [Wbulk]*(L-2) + [Wright]
    return Ws

def construct_convolution_MPO(L, conv_MPS=None):
    """
    mathematically, convolution is given by (f * g)(l) = \sum_k f(k) g(l-k).
    We can construct an MPO for this operation by modifying the binary addition MPO 
    such that it computes j = (2^n - 1) - k + l instead of j = k + l, and post-selecting on the first bit of j being 0. This way, we have j = l - k mod 2^n, and by feeding in f(k) and g(l-k) we can compute the convolution.
    """
    # binary addition MPO with open legs jkl
    Ws = binary_addition_MPO(L+1, modular=True)
    Wleft = Ws.pop(0)
    # shift input k -> (2^n -1) - k
    X = np.fliplr(np.eye(2)) # bit flip matrix, i.e., X|0> = |1>, X|1> = |0>
    Ws = list(map(lambda W: np.einsum('abjkl, km -> abjml', W, X), Ws)) # apply X to the k leg of each W, i.e., shift input k -> (2^n-1) - k
    # shift output l -> l + 2^n/2 and post-select on l's first bit being 0
    X1_CX10 = np.zeros((2,)*4)
    X1_CX10[:, 1, :, 0] = np.eye(2)
    X1_CX10[:, 0, :, 1] = X
    Ws[0] = np.einsum('nlm, abl, bcjkm -> acjkn', X1_CX10[0], Wleft[:, :, 0, 0, :], Ws[0])
    # transpose such that physical legs follow numpy convention:
    # output, function, convolution function
    Ws = list(map(lambda x: x.transpose(0, 1, 4, 2, 3), Ws))
    # check if convolution function is already given
    if conv_MPS is not None:
        Ws = list(map(lambda args: np.einsum('abjkl, cld -> acbdjk', *args)\
                                      .reshape(args[0].shape[0] * args[1].shape[0], -1, 2, 2),
                      zip(Ws, conv_MPS)))
    return Ws


def diag_MPO_from_MPS(Ms):
    Ws = []
    for M in Ms:
        chil, d, chir = M.shape
        W = np.zeros((chil, chir, d, d), dtype=M.dtype)
        W[:, :, np.arange(d), np.arange(d)] = M.transpose(0, 2, 1)
        Ws.append(W)
    return Ws

def contract(Ws):
    op = Ws[0]
    for W in Ws[1:]:
        op = np.tensordot(op, W, axes=(-3, 0))
    op = op.squeeze()
    idxs = [*range(0, op.ndim, 2), *range(1, op.ndim, 2)]
    op = op.transpose(idxs)
    dims = np.prod(op.shape[:op.ndim//2]), np.prod(op.shape[op.ndim//2:])
    op = op.reshape(dims)
    return op

def add_MPO(As, Bs, weights=None):
    if weights is None:
        weights = [1, 1]
    Cs = []
    # left boundary
    A = As[0]; B = Bs[0]
    _, chirA, d1, d2 = A.shape
    _, chirB,  _,  _ = B.shape
    C = np.zeros((1, chirA + chirB, d1, d2), dtype=A.dtype)
    C[:, :chirA, :, :] = weights[0] * A
    C[:, chirA:, :, :] = weights[1] * B
    Cs.append(C)
    # bulk
    for A, B in zip(As[1:-1], Bs[1:-1]):
        chilA, chirA, d1, d2 = A.shape
        chilB, chirB,  _,  _ = B.shape
        C = np.zeros((chilA + chilB, chirA + chirB, d1, d2), dtype=A.dtype)
        C[:chilA, :chirA, :, :] = A
        C[chilA:, chirA:, :, :] = B
        Cs.append(C)
    # right boundary
    A = As[-1]; B = Bs[-1]
    chilA, _, d1, d2 = A.shape
    chilB, _,  _,  _ = B.shape
    C = np.zeros((chilA + chilB, 1, d1, d2), dtype=A.dtype)
    C[:chilA, :, :, :] = A
    C[chilA:, :, :, :] = B
    Cs.append(C)
    return Cs