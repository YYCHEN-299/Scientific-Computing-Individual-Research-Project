#define FP_FAST_FMAF
__kernel void sell_rd_spmv(__global const int * slice_start,
                           __global const int * slice_col,
                           __global const int * colidx,
                           __global const float * val,
                           __global const float * x,
                           const int slice_height,
                           const int slice_count,
                           const int row_th,
                           __local float * lm_y,
                           __global float * y)
{
    int local_id = get_local_id(0);
    int local_th = local_id % row_th;
    int slice_id = get_group_id(0);
    int global_th = slice_id * slice_height * row_th + local_id;
    int row_id = global_th / row_th;
    float sub_data = 0.0;
    int row_len = (slice_col[row_id] + row_th - 1) / row_th;
    int idx = 0;
    for (int i = 0; i < row_len; i++) {
        idx = i * slice_height * row_th + slice_start[slice_id] + local_id;
        sub_data = mad(val[idx], x[colidx[idx]], sub_data);
    }
    lm_y[local_id] = sub_data;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = row_th / 2; s > 0; s >>= 1) {
        if (local_th < s) {
            lm_y[local_id] += lm_y[local_id + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    if (local_th == 0) {
        y[row_id] = lm_y[local_id];
    }
}
