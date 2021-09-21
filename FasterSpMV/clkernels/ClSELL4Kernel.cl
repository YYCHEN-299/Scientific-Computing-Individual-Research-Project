#define FP_FAST_FMAF
__kernel void sell4_spmv(__global const int * restrict slice_col,
                         __global const int4 * restrict colidx,
                         __global const float4 * restrict val,
                         __global const float * restrict x,
                         __global float4 * restrict y)
{
    __private int i, j;
    i = get_global_id(0);
    __private int4 lm_colidx;
    __private float4 lm_x, lm_val;
    __private float4 row_data = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    __private int ptr1 = slice_col[i];
    __private int ptr2 = slice_col[i + 1];
    for (j = ptr1; j < ptr2; j++) {
        lm_colidx = vload4(j, (__global int *)colidx);
        lm_x = (float4)(x[lm_colidx.x], x[lm_colidx.y],
                        x[lm_colidx.z], x[lm_colidx.w]);
        lm_val = vload4(j, (__global float *)val);
        row_data = fma(lm_x, lm_val, row_data);
    }
    vstore4(row_data, i, (__global float *)y);
}