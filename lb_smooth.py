# This code is an adaptation of the matlabcode presented in:
# [3] Seo, S., Chung, M.K., Vorperian, H. K. 2010. Heat kernel smoothing 
#     using Laplace-Beltrami eigenfunctions,
#     Medical Image Computing and Computer-Assisted Intervention (MICCAI) 
#     2010, Lecture Notes in Computer Science (LNCS). 6363:505-512.
#     http://www.stat.wisc.edu/~mchung/papers/miccai.2010.seo.pdf

import math
import numpy as np
# The function performs heat kernel smoothing using the eigenfunctions 
# of the Laplace-Beltrami operator on a triangle mesh. 
def lb_smoothing(mesh, sigma, k, evecs, evals):

    p = mesh.points()
    print(p)
    #sortear evals?
    Psi = evecs[:, 0:k]

    W = np.exp(-1*evals*sigma)   

    W = np.tile(W.transpose(), (len(Psi), 1))

    PsiT = Psi.transpose()

    A = np.matmul(PsiT, Psi)

    beta = np.matmul(np.linalg.inv(A), PsiT)

    beta = np.matmul(beta, p)

    phat = np.multiply(W, Psi)

    phat = phat.dot(beta)
    print(phat)
    return phat
