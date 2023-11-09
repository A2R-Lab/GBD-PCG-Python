# GBD-PCG-Python

Python proof of concept for [GBD-PCG](https://github.com/A2R-Lab/GBD-PCG/tree/main). 

Solves for x in Ax = b, using the Preconditioned Conjugate Gradient algorithm. It requires A to be a positive semi-definite matrix to guarantee good results.

## Requirements

Only python and numpy are required

## Usage

You can look at ```test.py``` for an example:
```
from PCG import PCG
options = {'preconditioner_type' : 'SS'}
pcg = PCG(A, b, nx, n_blocks, options = options)
x_pcg = pcg.solve()
```

## Preconditoners

This provides the following preconditoners

1. Identity: "0"
2. Jacobi (Diagonal): "J"
3. Block Jacobi (Block-diagonal): "BJ"
4. Symmetric Stair: "SS" 

While (1) and (2) only require the matrix A, (3) and (4) additional require the block_size (nx) to be passed in.

### Citing
To cite this work in your research, please use the following bibtex:
```
@misc{adabag2023mpcgpu,
      title={MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU}, 
      author={Emre Adabag and Miloni Atal and William Gerard and Brian Plancher},
      year={2023},
      eprint={2309.08079},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```