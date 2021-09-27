#define FP_FAST_FMAF
__kernel void csr_spmv(__global const int * restrict rowptr,
                       __global const int * restrict colidx,
                       __global const float * restrict val,
                       __global const float * restrict x,
                       __global float * restrict y)
{
    int i, k;
    i = get_global_id(0);
    float row_data = 0.0f;
    for (k = rowptr[i]; k < rowptr[i + 1]; k++) {
        row_data = mad(x[colidx[k]], val[k], row_data);
    }
    y[i] = row_data;
}


