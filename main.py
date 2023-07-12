import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs, spsolve
import openmesh
import argparse
import os
import polyscope as ps
import trimesh

import signature
import laplace
from lb_smooth import *

def compute_eigenvalues(filename, args):
    name = os.path.splitext(filename)[0]
    if os.path.exists(name + '.npz'):
        extractor = signature.SignatureExtractor(path=name+'.npz')
    else:
        mesh = trimesh.load(filename)
        extractor = signature.SignatureExtractor(mesh, 10, args.approx)
        np.savez_compressed(name+'.npz', evals=extractor.evals, evecs=extractor.evecs)
    
    return extractor.get_values()


parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('--n_basis', default='2000', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--kernel', type=str, default='heat', help='Feature type to extract. Must be in [heat, wave]')

args = parser.parse_args()

file = 'cat0.off'

mesh = openmesh.read_trimesh(file)

evals , evecs = compute_eigenvalues(file, args)

new_points = lb_smoothing(mesh, 0, 2000, evecs, evals)

#mesh2 = openmesh.read_trimesh(file)

ps.init()
ps_mesh = ps.register_surface_mesh("mesh", new_points, mesh.face_vertex_indices())
ps.show()

# obtener mesh

# caclular valores y vectores propios

# smoothing




