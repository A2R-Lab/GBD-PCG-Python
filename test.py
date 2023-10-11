import pcg
import numpy as np
import utils

# num_get_pos
nq = 2 

# num_get_vel
nv = 1

# knot points
n_blocks = 4 

nx = nq+nv
n = nx * n_blocks

# PCG expects a positive semidefinite matrix
A = utils.psd_block_diagonal(nx, n_blocks)
# Generate a random vector b
b = np.random.rand(n, 1)


x_numpy = np.linalg.solve(A, b)
x_pcg = pcg.solve(A, b, "SS", nx)
if(np.allclose(x_numpy,  x_pcg)):
	print("Test passed")
else:
	print("Test failed")
	print("Numpy answer")
	print(x_numpy)
	print("PCG answer")
	print(x_pcg)
