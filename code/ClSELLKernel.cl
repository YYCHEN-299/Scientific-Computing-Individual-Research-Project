__kernel void sell_spmv(__global const int * slice_ptr,
                        __global const int * colidx,
                        __global const float * val,
                        __global const float * x,
                        const int slice_height,
                        const int slice_count,
                        __global float * y,
                        __local float * lm)
{
    int z = get_global_id(0);
    int j, k;
    for (k = slice_ptr[z]; k < slice_ptr[z + 1]; k += slice_height) {
        for (j = 0; j < slice_height; j++) {
            y[z * slice_height + j] += x[colidx[k + j]] * val[k + j];
        }
    }
}