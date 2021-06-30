CSR format and the code of SpMV based on CSR:

![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/csr_format.png)

``` Java
for(i = 0; i < N; i++)
{
	a = rowptr[i];
	b = rowptr[i + 1];
	for(k = a; k < b; k++)
	{
		y += val[k] * x[colidx[k]];
	}
}
```

## ELLPACK
ELLPACK was introduced as a format to compress a sparse matrix with the purpose of solving large sparse linear systems with ITPACKV subroutines on vector computers.

This format stores the sparse matrix on two arrays, one float (val[ ]), to save the entries, and one integer (J[ ]), to save the column index of every entry. Both arrays are, at least, of dimension N × MaxEntriesByRows, where N is the number of rows and MaxEntriesByRows is the maximum number of nonzeros per row in the matrix, with the maximum being taken over all rows. Note that the size of all rows in these compressed arrays val[ ] and J[ ] is the same, because every row is padded with zeros.

ELLPACK can be considered as an approach to fit a sparse matrix in a regular data structure like a dense matrix. Consequently, this format is appropriate to compute operations with sparse matrices on vector architectures.

Recall SpMV is y = A * x. If every element of vector y is computed by a thread identified by index = i and the arrays store their elements in column-major order, then the SpMV based on ELLPACK can improve the performance due to:

1. The coalesced global memory access, thanks to the column-major ordering used to store the matrix elements into the data structures. Then, the thread identified by index i accesses to the elements in the i row: val[i + k ∗ N] with 0 ≤ k < MaxEntriesByRows where k is the column index into the new data structures val[ ] and J[ ]. Consequently, two threads i and i + 1 access to consecutive memory address, thereby full filling the conditions of coalesced global memory access.

2. Non-synchronized execution between different blocks of threads. Every block of threads can complete its computation without synchronization with others blocks, because every thread computes one element of the vector y, and there are no data dependencies in the computation of different elements of y.

However, if the percentage of zeros is high in the ELLPACK data structure and there is a relevant amount of padding zeros, then the performance decreases. This penalty even remains when conditional branches are included to avoid the memory access and arithmetic operations with padding zeros, as described in the figure(right). This is because to compute every y[i], with 0 ≤ i ≤ N, the k-loop must iterate until k = MaxEntriesByRows and the conditional branch is executed in every iteration; hence in order to reduce the memory access and activity of arithmetic units, the computation is penalized with N × MaxEntriesByRows executions of the conditional branch.

![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/ELLPACK_SpMV_algo.png)

The code of SpMV based on ELLPACK:
``` Java
b = MaxEntriesByRows;
for(i = 0; i < N; i++)
{
	for(k = a; k < b; k++)
	{
		if(val[i + N * k] != 0)
		{
			y[i] += val[i + N * k] * x[J[i + N * k]];
		}
	}
}
```

## Sliced ELLPACK

![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/Sliced-ELLPACK_format.png)

It can be represented by:

An array val storing the nonzero values and padded zeros

An array colidx storing the column indices

An array rlen storing the number of nonzeros in each row

An array storing the beginning position (index in val) of each slice. The rows are padded with zero values so that the lengths of the rows in each slice are equal.

Slicing is the most important modification to the original ELLPACK format. It not only reduces zero-padding significant but also promotes data locality in accessing the input vector. The lower the slice height, the less zero padding would be required. In order to effectively utilize vector registers in CPUs, the slice height should be a multiple of
the vector length, which is equal to the cache line length of SIMD registers. 

The code of SpMV based on Sliced ELLPACK:

![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/Sliced-ELLPACK_SpMV_algo.png)

## Concepts
### ITPACK
https://web.ma.utexas.edu/CNA/ITPACK/
