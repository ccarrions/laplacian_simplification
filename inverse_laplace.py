from scipy.sparse import *
from scipy.sparse.linalg import eigs, spsolve
import numpy as np

#Esta función calcula el ángulo entre dos aristas u y v
def myangle(u,v):
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)

    du = max(du, 1e-8)
    dv = max(dv, 1e-8)

    return np.arccos(np.dot(u,v)/(du*dv))

#Esta función computa la matriz Laplaciana de una malla
#Como la matriz Laplaciana tiene muchos elementos ceros, se utiliza una matriz dispersa (lil_matrix)
def laplacian(mesh):
    n = mesh.n_vertices() #Num. vértices de la malla
    print(f"Num. vertices {n}")
    W = lil_matrix((n,n), dtype=np.float) #Se crea una matriz dispersa de n x n
    print(W.shape)

    points = mesh.points() #Se obtienen las coordenadas de los vértices de la malla

    #Para cada vertice de la malla
    for i,v in enumerate(mesh.vertices()):
        f_it = openmesh.VertexFaceIter(mesh, v) #Se obtienen las caras que comparten el vértice v
        for f in f_it: #Para cada cara f
            v_it = openmesh.FaceVertexIter(mesh,f) #Se obtienen los vértices de la cara f
            L = [] 
            for vv in v_it: # Se obtienen los vértices compartidos para esa cara
                if vv.idx()!=i:
                    L.append(vv.idx())
            j = L[0]
            k = L[1]

            #Se obtienen las coordenadas de los vértices en una cara
            vi = points[i,:]
            vj = points[j,:]
            vk = points[k,:]

            #Se calculan los ángulos alpha y beta
            alpha = myangle(vi-vk, vj-vk)
            beta = myangle(vi-vj,vk-vj)

            #Se acumulan los cotangentes para las aristas ij e ik
            W[i,j] = W[i,j] + 1.0/np.tan(alpha)
            W[i,k] = W[i,k] + 1.0/np.tan(beta)
    
    #Se suman todas las filas de la matriz W y se calcula la inversa de cada suma
    S = 1.0/W.sum(axis=1)
    print(S.shape)

    #Se calcula la matriz Laplaciana normalizada. La diagonal es 1 y los demás elementos son -1/n. La idea es que
    #la suma de cada fila sea 0
    W = eye(n,n)-spdiags(np.squeeze(S),0,n,n)*W
    return W