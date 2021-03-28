# Scientific-Computing-Individual-Research-Project
## Background
The Sparse matrix-vector multiplication (SpMV) operation (y = A ∗ x) is widely used in scientific and engineering calculations. But CSR-based SpMV has poor performance on processors with vector units [1].

CSR format stores nonzero elements discretely; thus, each multiplication needs memory access to fetch the nonzero elements in the matrix as well as the corresponding elements in the dense vector. Hence, a new pattern is needed to ameliorate this drawback [2].

## Goals
In order to take full advantage of SIMD acceleration technology, the goals of my project are:
1.	Design a new sparse matrix storage format to adapt the SIMD units.
2.	Design a new sparse matrix storage format‑based SpMV algorithm using SIMD vectorization.

## Steps
1.	Understand current sparse matrix storage format and SpMV algorithms.
2.	Understand SIMD acceleration technology in SpMV.
3.	Find the reason why current sparse matrix storage format and current SpMV algorithms are not good.
e.g.: Loop remainder, Data locality, Inefficient use of CPU, SIMD-unfriendly, …
4.	Design a new sparse matrix storage format.
5.	Understand how to design an OpenCL software.
6.	Design a new sparse matrix storage format‑based SpMV algorithm.

## Schedule
1.	Read and understand the source code of current sparse matrix storage format (CSR, ELLPACK, CSRL and others from research paper). (2-3 weeks)
2.	Learn SIMD acceleration technology in SpMV. (1-2 weeks)
3.	Read and understand the source code of SpMV multiplication functions from Scipy and other sparse matrix storage format‑based SpMV algorithms from research paper. (3-4 weeks)
4.	Learn OpenCL programming language. (2-3 weeks)
5.	Design a new sparse matrix storage format and Design a SpMV algorithm to fit the new data structure. (3-4 weeks)

## Basic concepts
### CSR:
As its name implies, this scheme compresses sparse matrix and reduces the storage requirements of the matrix and executes suitably by performing only the necessary computations on cache-based traditional microprocessors [1].
CSR format are efficient arithmetic operations CSR + CSR, CSR * CSR, etc. and efficient row slicing and fast matrix-vector products. But CSR format are slow column slicing operations and changes to the sparsity structure are expensive.
### Vector processor:
A vector processor is a CPU that implements an instruction set containing instructions that operate on one-dimensional arrays of data called vectors, compared to the scalar processors, whose instructions operate on single data items.
### SIMD:
SIMD = Single instruction multiple data.
As its name indicates, instead of performing a single instruction on every single data, it provides the capability of using wider data-width for similar computational operations [3].
### OpenCL:
Open Computing Language (OpenCL) is one of the programming languages. It is an industry standard framework for programming computers composed of a combination of CPUs, GPUs and other processors. These so-called heterogeneous systems have become an important class of platforms, and OpenCL is the first industry standard that directly addresses their needs [4].

## References
1.	Chen X, Xie P, Chi L, et al. An efficient SIMD compression format for sparse matrix‐vector multiplication[J]. Concurrency and Computation: Practice and Experience, 2018, 30(23): e4800.
2.	Li, Y., Xie, P., Chen, X. et al. VBSF: a new storage format for SIMD sparse matrix–vector multiplication on modern processors. J Supercomput 76, 2063–2081 (2020). https://doi.org/10.1007/s11227-019-02835-4
3.	Hossein Amiri, Asadollah Shahbahrami, SIMD programming using Intel vector extensions, Journal of Parallel and Distributed Computing, Volume 135, 2020, Pages 83-100, ISSN 0743-7315, https://doi.org/10.1016/j.jpdc.2019.09.012.
4.	Munshi A, Gaster B, Mattson T G, et al. OpenCL programming guide[M]. Pearson Education, 2011.
