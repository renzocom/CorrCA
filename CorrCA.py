#
# Renzo Comolatti (renzo.com@gmail.com)
#
# Class with Correlated Component Analysis (CorrCA) method based on
# original matlab code from Parra's lab (https://www.parralab.org/corrca/).
#
# started 18/10/2019

import numpy as np
import scipy as sp

def fit(X, version=2, gamma=0, k=None):
    '''
    Correlated Component Analysis (CorrCA).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal to calculate CorrCA.
    k : int,
        Truncates eigenvalues on the Kth component.
    gamma : float,
        Truncates eigenvalues using SVD.

    Returns
    -------
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).
    ISC : list of floats
        Inter-subject Correlation values.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).
    '''

    # TODO: implement case 3, tsvd truncation

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    if k is not None: # truncate eigenvalues using SVD
        gamma = 0
    else:
        k = D

    # Compute within- (Rw) and between-subject (Rb) covariances
    if False: # Intuitive but innefficient way to calculate Rb and Rw
        Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
        Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
        Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
        Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
        Rb = (Rt - Rw) / (N-1)

    Rw = sum(np.cov(X[n,...]) for n in range(N))
    Rt = N**2 * np.cov(X.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    k = min(k, np.linalg.matrix_rank(Rw)) # handle rank deficient data.
    if k < D:
        def regInv(R, k):
            '''PCA regularized inverse of square symmetric positive definite matrix R.'''

            U, S, Vh = np.linalg.svd(R)
            invR = U[:, :k].dot(sp.diag(1 / S[:k])).dot(Vh[:k, :])
            return invR

        invR = regInv(Rw, k)
        ISC, W = sp.linalg.eig(invR.dot(Rb))
        ISC, W = ISC[:k], W[:, :k]

    else:
        Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
        ISC, W = sp.linalg.eig(Rb, Rw_reg) # W is already sorted by eigenvalue and normalized

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))

    ISC, W = np.real(ISC), np.real(W)

    if k==D:
        A = Rw.dot(W).dot(sp.linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))

    return W, ISC, A

def transform(X, W):
    '''
    Get CorrCA components from signal(X), e.g. epochs or evoked, using backward model (W).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times) or (n_dim, n_times)
        Signal  to transform.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    '''

    flag = False
    if X.ndim == 2:
        flag = True
        X = X[np.newaxis, ...]
    N, _, T = X.shape
    K = W.shape[1]
    Y = np.zeros((N, K, T))
    for n in range(N):
        Y[n, ...] = W.T.dot(X[n, ...])
    if flag:
        Y = np.squeeze(Y, axis=0)
    return Y

def get_ISC(X, W):
    '''
    Get ISC values from signal (X) and backward model (W)

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal to calculate CorrCA.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    ISC : list of floats
        Inter-subject Correlation values.
    '''
    N, D, T = X.shape

    Rw = sum(np.cov(X[n,...]) for n in range(N))
    Rt = N**2 * np.cov(X.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))
    return np.real(ISC)

def get_forwardmodel(X, W):
    '''
    Get forward model from signal(X) and backward model (W).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal  to transform.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).
    '''

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    Rw = sum(np.cov(X[n,...]) for n in range(N))
    Rt = N**2 * np.cov(X.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    k = np.linalg.matrix_rank(Rw)
    if k==D:
        A = Rw.dot(W).dot(sp.linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))
    return A

def reconstruct(Y, A):
    '''
    Reconstruct signal(X) from components (Y) and forward model (A).

    Parameters
    ----------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).

    Returns
    -------
    X : ndarray of shape = (n_subj, n_dim, n_times) or (n_dim, n_times)
        Signal.
    '''

    flag = False
    if Y.ndim == 2:
        flag = True
        Y = Y[np.newaxis, ...]
    N, _, T = Y.shape
    D = A.shape[0]
    X = np.zeros((N, D, T))
    for n in range(N):
        X[n, ...] = A.dot(Y[n, ...])

    if flag:
        X = np.squeeze(X, axis=0)
    return X

def stats(X, gamma=0, k=None, n_surrogates=200, alpha=0.05):
    '''
    Compute ISC statistical threshold using circular shift surrogates.
    Parameters
    ----------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).

    Returns
    -------
    '''
    ISC_null = []
    for n in range(n_surrogates):
        if n%10==0:
            print('#', end='')
        surrogate = circular_shift(X)
        W, ISC, A = fit(surrogate, gamma=gamma, k=k)
        ISC_null.append(ISC[0]) # get max ISC
    ISC_null = np.array(ISC_null)
    thr = np.percentile(ISC_null, (1 - alpha) * 100)
    print('')
    return thr, ISC_null

def circular_shift(X):
    n_reps, n_dims, n_times = X.shape
    shifts = np.random.choice(range(n_times), n_reps, replace=True)
    surrogate = np.zeros_like(X)
    for i in range(n_reps):
        surrogate[i, ...] = np.roll(X[i, ...], shifts[i], axis=1)
    return surrogate

def calc_CCA(epochs, times, **par):
    ini_ix = time2ix(times, par['response_window'][0])
    end_ix = time2ix(times, par['response_window'][1])
    X = np.array(epochs)[:, :, ini_ix : end_ix]

    W, ISC, A = fit(X, gamma=par['gamma'], k=par['K'])

    n_components = W.shape[1]
    if stats:
        print('Calculating statistics...')
        ISC_thr, ISC_null = stats(X, par['gamma'], par['K'], par['n_surrogates'], par['alpha'])
        n_components = sum(ISC > ISC_thr)
        W, ISC, A = W[:, :n_components], ISC[:n_components], A[:, :n_components]
    # else:
    #     ISC_cum = np.sumcum(100 * ISC/np.sum(ISC))
    #     n_components = len(ISC_cum < par['max_var']) + 1
    Y = transform(X, W)
    Yfull = transform(epochs, W)
    return W, ISC, A, Y, Yfull, ISC_thr

def time2ix(times, t):
    return np.abs(times - t).argmin()

def get_CCA_id(**par):
    CCA_id = 'CCA_{}_{}'.format(par['response_window'][0], par['response_window'][1])
    if par['stats']:
        CCA_id += '_stats_K_{}_surr_{}_alpha_{}_gamma_{}'.format(par['K'], par['n_surrogates'], par['alpha'], par['gamma'])
    return CCA_id

################################################################################

def CorrCA_matlab(X, W=None, version=2, gamma=0, k=None):
    '''
    Correlated Component Analysis.

    Parameters
    ----------
    X : array, shape (n_subj, n_dim, n_times)
    k : int,
        Truncates eigenvalues on the Kth component.

    Returns
    -------
    W
    ISC
    Y
    A
    '''

    # TODO: implement case 3, tsvd truncation

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    if k is not None: # truncate eigenvalues using SVD
        gamma = 0
    else:
        k = D

    # Compute within- and between-subject covariances
    if version == 1:
        Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
        Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
        Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
        Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
        Rb = (Rt - Rw) / (N-1)

    elif version == 2:
        Rw = sum(np.cov(X[n,...]) for n in range(N))
        Rt = N**2 * np.cov(X.mean(axis=0))
        Rb = (Rt - Rw) / (N-1)

    elif version == 3:
        pass

    if W is None:
        k = min(k, np.linalg.matrix_rank(Rw)) # handle rank deficient data.
        if k < D:
            def regInv(R, k):
                '''PCA regularized inverse of square symmetric positive definite matrix R.'''

                U, S, Vh = np.linalg.svd(R)
                invR = U[:, :k].dot(sp.diag(1 / S[:k])).dot(Vh[:k, :])
                return invR

            invR = regInv(Rw, k)
            ISC, W = sp.linalg.eig(invR.dot(Rb))
            ISC, W = ISC[:k], W[:, :k]

        else:
            Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
            ISC, W = sp.linalg.eig(Rb, Rw_reg) # W is already sorted by eigenvalue and normalized

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))

    ISC, W = np.real(ISC), np.real(W)

    Y = np.zeros((N, k, T))
    for n in range(N):
        Y[n, ...] = W.T.dot(X[n, ...])

    if k==D:
        A = Rw.dot(W).dot(sp.linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))

    return W, ISC, Y, A
