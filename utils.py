import numpy as np

def check_positive_semidefinite(A):
	eigenvalues = np.linalg.eigvals(A)

	# Check if all eigenvalues are non-negative
	is_positive_semidefinite = np.all(eigenvalues >= 0)

	if is_positive_semidefinite:
		return True
	else:
		return False

def generate_A(nx, n_blocks):
	
	# Define the size of the overall matrix
	n = nx * n_blocks

	# Initialize the block tridiagonal matrix
	A = np.zeros((n, n))

	# Create positive semi-definite diagonal blocks (nx x nx) with rounding
	# Create positive semi-definite diagonal blocks (nx x nx) with rounding
	for i in range(n_blocks):
		# Generate a random nx x nx matrix with values between 0 and 1
		diag_block = np.random.rand(nx, nx)
		# Make it symmetric
		diag_block = 0.5 * (diag_block + diag_block.T)
		# Ensure it's positive semidefinite by adding a multiple of the identity matrix
		diag_block += 0.1 * np.identity(nx)
		# Round the elements to one decimal place
		diag_block = np.round(diag_block, 1)
		# Assign it to the diagonal block
		A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = diag_block

	# Create sub-diagonal block (nx x nx) with non-zero values and rounding
	subdiagonal_block = np.random.rand(nx, nx)
	subdiagonal_block = np.round(subdiagonal_block, 1)

	# Create super-diagonal block (nx x nx) with non-zero values and rounding
	superdiagonal_block = np.random.rand(nx, nx)
	superdiagonal_block = np.round(superdiagonal_block, 1)

	# Fill the super-diagonal and sub-diagonal blocks in the main block tridiagonal matrix
	for i in range(n_blocks - 1):
		A[i * nx:(i + 1) * nx, (i + 1) * nx:(i + 2) * nx] = superdiagonal_block
		A[(i + 1) * nx:(i + 2) * nx, i * nx:(i + 1) * nx] = subdiagonal_block
	
	return A

def blockdiagonal(nx, n_blocks):
	
	# Define the size of the overall matrix
	n = nx * n_blocks

	# Initialize the block tridiagonal matrix
	A = np.zeros((n, n))

	# Create positive semi-definite diagonal blocks (nx x nx) with rounding
	# Create positive semi-definite diagonal blocks (nx x nx) with rounding
	for i in range(n_blocks):
		# Generate a random nx x nx matrix with values between 0 and 1
		diag_block = np.random.rand(nx, nx)
		# Make it symmetric
		diag_block = 0.5 * (diag_block + diag_block.T)
		# Ensure it's positive semidefinite by adding a multiple of the identity matrix
		diag_block += 0.1 * np.identity(nx)
		# Round the elements to one decimal place
		diag_block = np.round(diag_block, 1)
		# Assign it to the diagonal block
		A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = diag_block
	return A


def psd_block_diagonal(nx, n_blocks):
	A = blockdiagonal(nx, n_blocks)
	return np.dot(A, A.T)