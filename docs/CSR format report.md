The data structure of sparse matrix must have two components:
(i) An array that stores all the floating-point number of the matrix.
(ii) The pointers to the locations of the floating-point number in the matrix.

To exploit the sparsity of the matrix the use of pointers is unavoidable but often limits the memory system performance.
One reason for this is that pointers usually lead to poor cache utilization, since they lack spatial locality. The number of cache misses for the right- and/or left hand-side vectors can dramatically increase if the sparse matrix has an irregular sparsity structure.
Another important factor is that memory indirections (pointers) require extra load operations. In sparse matrix operations, the number of floating-point operations per load operation is lower than that of dense matrix operations, limiting overall performance.

## Basic CSR format:
Three arrays are employed: val, colidx and rowptr.
The nonzeros of the sparse matrix A are compressed into an array val in a row-wise manner. 
The column index of each nonzero entry is stored in the array colind.
The rowptr stores the index of the first nonzero of each row.
![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/csr_format.png)

Scipy example:
>>> indptr = np.array([0, 2, 3, 6]) # rowptr
>>> indices = np.array([0, 2, 2, 0, 1, 2]) # colidx
>>> data = np.array([1, 2, 3, 4, 5, 6]) # val
>>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
>>> array([[1, 0, 2],
>>>          [0, 0, 3],
>>>          [4, 5, 6]])

![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/csr_SpMV_algo.png)

## Alternative CSR format:
1.	Fixed-Size Blocked CSR:
![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/fixed-size_bcsr_format.png)

Each block length in A_12 is 2. Each block length in A_11 is 1.
It can reduce the number of load operations and the memory requirement, because only one index per block is required.

2.	Variable-Size Blocked CSR:
![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/variable-size_bcsr_format.png)

This storage scheme requires an array Nzptr (of length the number of blocks) in addition to the other three arrays used in CRS:
A_f (of length the number of nonzeros) to store the nonzero values.
Colind (of length the number of blocks) to store the column number of the first nonzero for each block.
Rowptr (of length the number of rows) to point to the position where the blocks of each row start. Nzptr stores the location of the first nonzero of each block in array A_f.

![Alt text](https://github.com/YYCHEN-299/Scientific-Computing-Individual-Research-Project/blob/main/docs/img/bcsr_SpMV_algo.png)

This storage scheme reduces extra load operations but requires an extra loop during the SpMV operation and thus suffers additional loop overhead. If the sizes of the blocks are small, the overhead due to the extra loop will dominate the gain due to decreased load operations. Thus, the effectiveness of this storage scheme depends directly on the sizes of the blocks in the matrix.

## Concepts:
### Temporal locality: If at one point a particular memory location is referenced, then it is likely that the same location will be referenced again soon. There is temporal proximity between adjacent references to the same memory location. In this case it is common to make efforts to store a copy of the referenced data in faster memory storage, to reduce the latency of subsequent references. Temporal locality is a special case of spatial locality, namely when the prospective location is identical to the present location.

### Spatial locality: If a particular storage location is referenced at a particular time, then it is likely that nearby memory locations will be referenced soon. In this case it is common to attempt to guess the size and shape of the area around the current reference for which it is worthwhile to prepare faster access for subsequent reference.

### Memory locality (or data locality): Spatial locality explicitly relating to memory.