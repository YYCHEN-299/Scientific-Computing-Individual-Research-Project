#define FP_FAST_FMAF
__kernel void sell2_spmv(__global const int * restrict slice_col,
                         __global const int2 * restrict colidx,
                         __global const float2 * restrict val,
                         __global const float * restrict x,
                         __global float2 * restrict y)
{
    int i, k;
    i = get_global_id(0);
    int2 lm_colidx;
    float2 lm_x, lm_val;
    float2 row_data = (float2)(0.0f, 0.0f);
    int ptr1 = slice_col[i];
    int ptr2 = slice_col[i + 1];
    for (k = ptr1; k < ptr2; k++) {
        lm_colidx = vload2(k, (__global int *)colidx);
        lm_x = (float2)(x[lm_colidx.x], x[lm_colidx.y]);
        lm_val = vload2(k, (__global float *)val);
        row_data = mad(lm_x, lm_val, row_data);
    }
    vstore2(row_data, i, (__global float *)y);
}
