#define FP_FAST_FMAF
__kernel void sell16_spmv(__global const int * restrict slice_col,
                          __global const int16 * restrict colidx,
                          __global const float16 * restrict val,
                          __global const float * restrict x,
                          __global float16 * restrict y)
{
    int i, k;
    i = get_global_id(0);
    int16 lm_colidx;
    float16 lm_x, lm_val;
    float16 row_data = (float16)(0.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f);
    int ptr1 = slice_col[i];
    int ptr2 = slice_col[i + 1];
    for (k = ptr1; k < ptr2; k++) {
        lm_colidx = vload16(k, (__global int *)colidx);
        lm_x = (float16)(x[lm_colidx.s0], x[lm_colidx.s1],
                         x[lm_colidx.s2], x[lm_colidx.s3],
                         x[lm_colidx.s4], x[lm_colidx.s5],
                         x[lm_colidx.s6], x[lm_colidx.s7],
                         x[lm_colidx.s8], x[lm_colidx.s9],
                         x[lm_colidx.sa], x[lm_colidx.sb],
                         x[lm_colidx.sc], x[lm_colidx.sd],
                         x[lm_colidx.se], x[lm_colidx.sf]);
        lm_val = vload16(k, (__global float *)val);
        row_data = mad(lm_x, lm_val, row_data);
    }
    vstore16(row_data, i, (__global float *)y);
}
