__kernel void bsell_spmv(__global const int * restrict slice_ptr,
                         __global const int * restrict slice_col,
                         __global const int * restrict colidx,
                         __global const float * restrict val,
                         __global const float * restrict x,
                         const int slice_height,
                         __global float * restrict y)
{
    int z, i, j, k, idx;
    float row_data = 0.0f;
    i = get_group_id(0);
    j = get_local_id(0);
    for (k = 0; k < slice_col[i]; k++) {
        idx = slice_ptr[i] + k * slice_height + j;
        row_data += x[colidx[idx]] * val[idx];
    }
    z = get_global_id(0);
    y[z] = row_data;
}