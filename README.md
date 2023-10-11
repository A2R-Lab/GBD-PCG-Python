# GBD-PCG-Python

Python proof of concept for [GBD-PCG](https://github.com/A2R-Lab/GBD-PCG/tree/main). 

Solves for x in Ax = b, using the Preconditioned Conjugate Gradient algorithm. It requires A to be a positive semi-definite matrix.

## Requirements

Only python and numpy are required

## Usage

You can look at ```test.py``` for an example:
```
import pcg

x_pcg = pcg.solve(A, b, "SS", block_size)
```

## Preconditoners

This provides the following preconditoners

1. Identity: "0"
2. Jacobi (Diagonal): "J"
3. Block Jacobi (Block-diagonal): "BJ"
4. Symmetric Stair: "SS" 

While (1) and (2) only require the matrix A, (3) and (4) additional require the block_size (nx) to be passed in.