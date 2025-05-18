import numpy as np
MACHINE_EPSILON = np.finfo(np.double).eps

def weight_fusion_matrices(P_list):
    NITER = 50
    num = len(P_list)
    alpha = np.array([1 / num for _ in range(num)])
    Obj = []
    for iter in range(NITER):
        # Fix alpha, update S
        S = np.zeros_like(P_list[0])
        # Compute the weighted sum
        alpha = alpha / np.sum(alpha)
        for m, w in zip(P_list, alpha):
            S += w * m

        # Fix S, update alpha
        for v in range(num):
            alpha[v] = 0.5 / np.linalg.norm(S - P_list[v], 'fro')

        # Calculate obj
        obj = 0
        for v in range(num):
            obj += np.linalg.norm(S - P_list[v], 'fro')
        Obj.append(obj)

        if iter > 1 and abs(Obj[iter - 1] - Obj[iter]) < 1e-8:
            print('iteration stops at', iter)
            break
    return S, alpha

