import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags


def Laplacian_1D_eig(a, b, N, eps=lambda x: np.ones_like(x), k=3):
    n = N - 2
    h = (b - a) / (N - 1)

    L = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    L = -L / h**2

    x = np.linspace(a, b, num=N)
    M = diags([eps(x[1:-1])], [0])

    eigvals, eigvecs = eigsh(L, k=k, M=M, which="SM")
    eigvecs = np.vstack((np.zeros((1, k)), eigvecs, np.zeros((1, k))))
    norm_eigvecs = np.linalg.norm(eigvecs, axis=0)
    eigvecs /= norm_eigvecs
    return eigvals.astype(np.float32), eigvecs.astype(np.float32), x.astype(np.float32)
