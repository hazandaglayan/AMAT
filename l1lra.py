import numpy as np

# It solves $\min\|M-UV\|_1$
def L1LRAcd(M, r=1, maxiter=100, U0=None, V0=None, rel_error=[]):
    m, n = M.shape
    if isinstance(U0, np.ndarray) or isinstance(V0, np.ndarray):
        U = U0
        V = V0
    else:
        U_, S_, VT_ = np.linalg.svd(M,full_matrices=0)
        V = VT_[:r]
        U = np.dot(U_[:, :r],np.diag(S_[:r]))


    for i in range(maxiter):

        R = M-np.dot(U,V)
        for k in range(r):
            # Current residue
            R = R + U[:,k].reshape(-1,1)*V[k,:].reshape(1,-1)
            # Weighted median subproblems
            U[:,k] = wmedian(R,V[k,:].T)
            V[k,:] = wmedian(R.T, U[:,k])

            # Update total residue
            R = R - U[:,k].reshape(-1,1)*V[k,:].reshape(1,-1)
        
        rel_error.append(np.sum(sum(abs(R))))

    return U, V, rel_error



def wmedian(A,y):

    ''' WMEDIAN computes an optimal solution of
    min_x  || A - xy^T ||_1

    where A has dimension (m x n), x (m) and y (n),
    in O(mn log(n)) operations. Note that it can be done in O(mn).

    This code comes from the paper
    "Dimensionality Reduction, Classification, and Spectral Mixture Analysis
    using Nonnegative Underapproximation", N. Gillis and R.J. Plemmons,
    Optical Engineering 50, 027001, February 2011.
    Available on http://sites.google.com/site/nicolasgillis/code'''

    # Reduce the problem for nonzero entries of y
    A = np.array(A)
    shape = y.shape
    if len(shape)==1:
        y = y.reshape(-1,1)
    indi = np.absolute(y) > 1e-16;
    m,n = A.shape
    y = y[indi]

    B = np.zeros((m, len(y.T)), dtype=type(A[0][0]))
    for i in range(m):
        B[i] = A[i].reshape(-1,1)[indi]
    A = B
    m,n = A.shape
    A = A/y
    y = np.absolute(y)/np.sum(np.absolute(y))


    # Sort rows of A, m*O(n log(n)) operations
    Inds = np.argsort(A)
    As = np.take_along_axis(A, Inds, axis=1)

    Y = np.array(np.matrix(y).getH()[Inds])
    # Extract the median
    actind = np.arange(m)
    i = 0;
    sumY = np.zeros((m,1));
    x = np.zeros((m,1), dtype=As.dtype);


    while len(actind):
        #sum of the weights
        sumY[actind] = sumY[actind] + np.reshape(Y[actind,i], (len(Y[actind,i]),1))
        # check which weight >=0
        supind = sumY[actind,:].reshape(-1) >=0.5
        # update corresponding x
        if len(As[actind[supind],i])>0:
            x[actind[supind],0] = As[actind[supind],i].reshape(-1)
        # only look reminding x to update
        actind = actind[~supind];
        i = i+1;

    return x.reshape(-1)
