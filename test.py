from PCG import PCG
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

# create PCG object
options = {'preconditioner_type' : 'SS'}
pcg = PCG(A, b, nx, n_blocks, options = options)
x_pcg = pcg.solve()

# compare
x_numpy = np.linalg.solve(A, b)
if(np.allclose(x_numpy,  x_pcg)):
	print("Test passed")
else:
	print("Test failed")
	print("Numpy answer")
	print(x_numpy)
	print("PCG answer")
	print(x_pcg)

# print trace
pcg.update_RETURN_TRACE(True)
x_pcg, traces = pcg.solve()
print("Printing Traces")
print(traces)