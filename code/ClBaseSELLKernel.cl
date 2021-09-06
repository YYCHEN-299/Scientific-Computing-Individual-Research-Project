__kernel void bsell_spmv(__global const int * restrict slice_ptr,
                         __global const int * restrict colidx,
                         __global const float * restrict val,
                         __global const float * restrict x,
                         const int slice_height,
                         __global float * restrict y)
{
    int z = get_global_id(0);
    int i, j, k;
    i = get_group_id(0);
    j = get_local_id(0);
    float row_data = 0.0f;
    for (k = slice_ptr[i] + j; k < slice_ptr[i + 1]; k += slice_height) {
        row_data += x[colidx[k]] * val[k];
    }
    y[z] = row_data;
}