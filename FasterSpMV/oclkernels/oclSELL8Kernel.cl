#define FP_FAST_FMAF
__kernel void sell8_spmv(__global const int * restrict slice_col,
                         __global const int8 * restrict colidx,
                         __global const float8 * restrict val,
                         __global const float * restrict x,
                         __global float8 * restrict y)
{
    int i, k;
    i = get_global_id(0);
    int8 lm_colidx;
    float8 lm_x, lm_val;
    float8 row_data = (float8)(0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f);
    int ptr1 = slice_col[i];
    int ptr2 = slice_col[i + 1];
    for (k = ptr1; k < ptr2; k++) {
        lm_colidx = vload8(k, (__global int *)colidx);
        lm_x = (float8)(x[lm_colidx.s0], x[lm_colidx.s1],
                        x[lm_colidx.s2], x[lm_colidx.s3],
                        x[lm_colidx.s4], x[lm_colidx.s5],
                        x[lm_colidx.s6], x[lm_colidx.s7]);
        lm_val = vload8(k, (__global float *)val);
        row_data = mad(lm_x, lm_val, row_data);
    }
    vstore8(row_data, i, (__global float *)y);
}
