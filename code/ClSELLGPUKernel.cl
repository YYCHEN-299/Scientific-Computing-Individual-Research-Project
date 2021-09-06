__kernel void sell_spmv(__global const int * slice_start,
                        __global const int * column_indices,
                        __global const float * elements,
                        __global const float * x,
                        const int slice_height,
                        const int slice_count,
                        __global float * result)
{
    int slices_per_workgroup = get_local_size(0) / slice_height;
    int id_in_slice = get_local_id(0) % slice_height;
    int global_warp_count = slices_per_workgroup * get_num_groups(0);
    int global_warp_id = slices_per_workgroup * get_group_id(0) + get_local_id(0) / slice_height;

    for (int slice_idx = global_warp_id; slice_idx < slice_count; slice_idx += global_warp_count) {
        float sum = 0.0f;

        int row = slice_idx * slice_height + id_in_slice;
        int offset = slice_start[slice_idx];
        int next_offset = slice_start[slice_idx + 1];
        int num_columns = (next_offset - offset) / slice_height;

        for (int item_id = 0; item_id < num_columns; item_id++) {
            int index = offset + item_id * slice_height + id_in_slice;
            sum += x[column_indices[index]] * elements[index];
        }

        result[row] = sum;
    }
}