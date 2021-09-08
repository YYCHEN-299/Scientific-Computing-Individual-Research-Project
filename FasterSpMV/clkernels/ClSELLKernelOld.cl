__kernel void sell_spmv(__global const int * restrict slice_ptr,
                        __global const int * restrict colidx,
                        __global const float * restrict val,
                        __global const float * restrict x,
                        const int slice_height,
                        const int slice_count,
                        __global float * restrict y,
                        __local float * lm)
{
    int z = get_group_id(0);
    int j, k;
    float row_data;
    for (j = 0; j < slice_height; j++) {
        row_data = 0.0f;
        for (k = slice_ptr[z] + j; k < slice_ptr[z + 1]; k += slice_height) {
            row_data += x[colidx[k]] * val[k];
        }
        y[z * slice_height + j]  = row_data;
    }
}