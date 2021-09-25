#define FP_FAST_FMAF
__kernel void sell4_spmv(__global const int * restrict slice_col,
                         __global const int4 * restrict colidx,
                         __global const float4 * restrict val,
                         __global const float * restrict x,
                         __global float4 * restrict y)
{
    int i, j;
    i = get_global_id(0);
    int4 lm_colidx;
    float4 lm_x, lm_val;
    float4 row_data = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (j = slice_col[i]; j < slice_col[i + 1]; j++) {
        lm_colidx = vload4(j, (__global int *)colidx);
        lm_x = (float4)(x[lm_colidx.x], x[lm_colidx.y],
                        x[lm_colidx.z], x[lm_colidx.w]);
        lm_val = vload4(j, (__global float *)val);
        row_data = mad(lm_x, lm_val, row_data);
    }
    vstore4(row_data, i, (__global float *)y);
}