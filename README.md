# Scientific-Computing-Individual-Research-Project
## Background
The Sparse matrix-vector multiplication (SpMV) operation (y = A ∗ x) is widely used in scientific and engineering calculations. But CSR-based SpMV has poor performance on processors with vector units [1].

CSR format stores nonzero elements discretely; thus, each multiplication needs memory access to fetch the nonzero elements in the matrix as well as the corresponding elements in the dense vector. Hence, a new pattern is needed to ameliorate this drawback [2].

## Goals
In order to learn the advantages of SIMD acceleration technology and sparse matrix storge format, the goals of my project are:
1.	Implement Sliced ELLPACK format and the SpMV algorithm based on it.
2.	Parallelize the Sliced ELLPACK format SpMV algorithm (At the first stage use Python and Numba, then use OpenCL).
3.	Design a format converter to convert CSR format to Sliced ELLPACK format.
4.	Compare SpMV performance between Sliced ELLPACK format and CSR format.

## Steps
1.	Understand current sparse matrix storage format and SpMV algorithms.
2.	Understand SIMD acceleration technology in SpMV.
3.	Find the reason why current sparse matrix storage format and current SpMV algorithms are not good.
e.g.: Loop remainder, Data locality, Inefficient use of CPU, SIMD-unfriendly, …
4.	Implement Sliced ELLPACK format and the SpMV algorithm based on it (Use Python and Numba).
5.	Design a format converter to convert CSR format to Sliced ELLPACK format.
6.	Understand how to design an OpenCL software and use OpenCL to implement Sliced ELLPACK format SpMV.
7.	Compare SpMV performance between Sliced ELLPACK format and CSR format.

## Basic concepts
### CSR:
As its name implies, this scheme compresses sparse matrix and reduces the storage requirements of the matrix and executes suitably by performing only the necessary computations on cache-based traditional microprocessors [1].
CSR format are efficient arithmetic operations CSR + CSR, CSR * CSR, etc. and efficient row slicing and fast matrix-vector products. But CSR format are slow column slicing operations and changes to the sparsity structure are expensive.

### ELLPACK:
ELLPACK was introduced as a format to compress a sparse matrix with the purpose of solving large sparse linear systems with ITPACKV subroutines on vector computers.
This format stores the sparse matrix on two arrays, one float (val[ ]), to save the entries, and one integer (J[ ]), to save the column index of every entry. Both arrays are, at least, of dimension N × MaxEntriesByRows, where N is the number of rows and MaxEntriesByRows is the maximum number of nonzeros per row in the matrix, with the maximum being taken over all rows. Note that the size of all rows in these compressed arrays val[ ] and J[ ] is the same, because every row is padded with zeros.
ELLPACK can be considered as an approach to fit a sparse matrix in a regular data structure like a dense matrix. Consequently, this format is appropriate to compute operations with sparse matrices on vector architectures.

### Vector processor:
A vector processor is a CPU that implements an instruction set containing instructions that operate on one-dimensional arrays of data called vectors, compared to the scalar processors, whose instructions operate on single data items.

### SIMD:
SIMD = Single instruction multiple data.
As its name indicates, instead of performing a single instruction on every single data, it provides the capability of using wider data-width for similar computational operations [3].

### OpenCL:
Open Computing Language (OpenCL) is one of the programming languages. It is an industry standard framework for programming computers composed of a combination of CPUs, GPUs and other processors. These so-called heterogeneous systems have become an important class of platforms, and OpenCL is the first industry standard that directly addresses their needs [4].

### Temporal locality:
If at one point a particular memory location is referenced, then it is likely that the same location will be referenced again soon. There is temporal proximity between adjacent references to the same memory location. In this case it is common to make efforts to store a copy of the referenced data in faster memory storage, to reduce the latency of subsequent references. Temporal locality is a special case of spatial locality, namely when the prospective location is identical to the present location.

### Spatial locality:
If a particular storage location is referenced at a particular time, then it is likely that nearby memory locations will be referenced soon. In this case it is common to attempt to guess the size and shape of the area around the current reference for which it is worthwhile to prepare faster access for subsequent reference.

### Memory locality (or data locality):
Spatial locality explicitly relating to memory.

## References
1.	Chen X, Xie P, Chi L, et al. An efficient SIMD compression format for sparse matrix‐vector multiplication[J]. Concurrency and Computation: Practice and Experience, 2018, 30(23): e4800.
2.	Li, Y., Xie, P., Chen, X. et al. VBSF: a new storage format for SIMD sparse matrix–vector multiplication on modern processors. J Supercomput 76, 2063–2081 (2020). https://doi.org/10.1007/s11227-019-02835-4
3.	Hossein Amiri, Asadollah Shahbahrami, SIMD programming using Intel vector extensions, Journal of Parallel and Distributed Computing, Volume 135, 2020, Pages 83-100, ISSN 0743-7315, https://doi.org/10.1016/j.jpdc.2019.09.012.
4.	Munshi A, Gaster B, Mattson T G, et al. OpenCL programming guide[M]. Pearson Education, 2011.
