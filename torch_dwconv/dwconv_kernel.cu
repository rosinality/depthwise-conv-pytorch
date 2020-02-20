#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cub/cub.cuh"

#define FULL_WARP_MASK 0xFFFFFFFF

#define CREATE_SHFL_MASK(mask, predicate)                                      \
  unsigned mask = __ballot_sync(FULL_WARP_MASK, (predicate))

static __host__ __device__ __forceinline__ int floor_div(int a, int b) {
  int c = a / b;

  if (c * b > a) {
    c--;
  }

  return c;
}

__device__ inline unsigned get_lane_id() {
  unsigned int lane_id;

#if __clang__
  return __nvvm_read_ptx_sreg_laneid();
#else
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
#endif

  return lane_id;
}

enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

struct DWConv2dKernelParams {
  int batch;
  int in_h;
  int in_w;
  int in_channel;

  int kernel_h;
  int kernel_w;

  int up_x;
  int up_y;
  int down_x;
  int down_y;

  int pad_x0;
  int pad_x1;
  int pad_y0;
  int pad_y1;

  int out_h;
  int out_w;
  int out_channel;

  int loop_major;
  int n_out;
};

template <typename scalar_t, DepthwiseConv2dDirection direction, int up_x,
          int up_y, int down_x, int down_y, int kernel_h, int kernel_w,
          int tile_out_h, int tile_out_w>
__global__ void dwconv2d_kernel(scalar_t *out, const scalar_t *input,
                                const scalar_t *kernel,
                                const DWConv2dKernelParams p) {
  const int tile_in_h = ((tile_out_h - 1) * down_y + kernel_h - 1) / up_y + 1;
  const int tile_in_w = ((tile_out_w - 1) * down_x + kernel_w - 1) / up_x + 1;

  __shared__ scalar_t sk[kernel_h][kernel_w];
  __shared__ scalar_t sx[tile_in_h][tile_in_w];

  int minor_idx = blockIdx.x;
  int tile_out_y = minor_idx;
  minor_idx -= tile_out_y;
  tile_out_y *= tile_out_h;
  int tile_out_x_base = blockIdx.y * tile_out_w;
  int major_idx_base = blockIdx.z * p.loop_major;

  const int major_dim = p.batch * p.in_channel;

  if (tile_out_x_base >= p.out_w | tile_out_y >= p.out_h |
      major_idx_base >= major_dim) {
    return;
  }

  for (int loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major & major_idx < major_dim;
       loop_major++, major_idx++) {
    int channel_idx = major_idx % p.in_channel;

    for (int tap_idx = threadIdx.x; tap_idx < kernel_h * kernel_w;
         tap_idx += blockDim.x) {
      int ky = tap_idx / kernel_w;
      int kx = tap_idx - ky * kernel_w;
      scalar_t v = 0.0;

      if (kx < p.kernel_w & ky < p.kernel_h) {
        if (direction == DIRECTION_FORWARD) {
          v = kernel[channel_idx * p.kernel_w * p.kernel_h + ky * p.kernel_w +
                     kx];
        } else {
          v = kernel[channel_idx * p.kernel_w * p.kernel_h +
                     (p.kernel_h - 1 - ky) * p.kernel_w +
                     (p.kernel_w - 1 - kx)];
        }
      }

      sk[ky][kx] = v;
    }

    __syncthreads();

    for (int loop_x = 0, tile_out_x = tile_out_x_base;
         loop_x < 1 & tile_out_x < p.out_w;
         loop_x++, tile_out_x += tile_out_w) {
      int tile_mid_x = tile_out_x * down_x + up_x - 1 - p.pad_x0;
      int tile_mid_y = tile_out_y * down_y + up_y - 1 - p.pad_y0;
      int tile_in_x = floor_div(tile_mid_x, up_x);
      int tile_in_y = floor_div(tile_mid_y, up_y);

      for (int in_idx = threadIdx.x; in_idx < tile_in_h * tile_in_w;
           in_idx += blockDim.x) {
        int rel_in_y = in_idx / tile_in_w;
        int rel_in_x = in_idx - rel_in_y * tile_in_w;
        int in_x = rel_in_x + tile_in_x;
        int in_y = rel_in_y + tile_in_y;

        scalar_t v = 0.0;

        if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
          v = input[((major_idx * p.in_h + in_y) * p.in_w + in_x) + minor_idx];
        }

        sx[rel_in_y][rel_in_x] = v;
      }

      __syncthreads();

      for (int out_idx = threadIdx.x; out_idx < tile_out_h * tile_out_w;
           out_idx += blockDim.x) {
        int rel_out_y = out_idx / tile_out_w;
        int rel_out_x = out_idx - rel_out_y * tile_out_w;
        int out_x = rel_out_x + tile_out_x;
        int out_y = rel_out_y + tile_out_y;

        int mid_x = tile_mid_x + rel_out_x * down_x;
        int mid_y = tile_mid_y + rel_out_y * down_y;
        int in_x = floor_div(mid_x, up_x);
        int in_y = floor_div(mid_y, up_y);
        int rel_in_x = in_x - tile_in_x;
        int rel_in_y = in_y - tile_in_y;
        int kernel_x = (in_x + 1) * up_x - mid_x - 1;
        int kernel_y = (in_y + 1) * up_y - mid_y - 1;

        scalar_t v = 0.0;

#pragma unroll
        for (int y = 0; y < kernel_h / up_y; y++)
#pragma unroll
          for (int x = 0; x < kernel_w / up_x; x++) {
            v += sx[rel_in_y + y][rel_in_x + x] *
                 sk[kernel_y + y * up_y][kernel_x + x * up_x];
          }

        if (out_x < p.out_w & out_y < p.out_h) {
          out[((major_idx * p.out_h + out_y) * p.out_w + out_x) + minor_idx] =
              v;
        }
      }
    }
  }
}

/*template <typename scalar_t, int kFilterHeight, int kFilterWidth>
__global__ void dwconv2d_backward_kernel_kernel(const scalar_t *out,
                                                const scalar_t *input,
                                                scalar_t *kernel,
                                                const DWConv2dKernelParams p) {
  const int in_channel = p.in_channel;
  const int in_height = p.in_h;
  const int in_width = p.in_w;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : p.kernel_h;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : p.kernel_w;
  const int stride_height = p.down_y;
  const int stride_width = p.down_x;
  const int pad_height = p.pad_y0;
  const int pad_width = p.pad_x0;
  const int out_channel = p.in_channel;
  const int out_height = p.out_h;
  const int out_width = p.out_w;

  for (int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < p.n_out; thread_id += blockDim.x * gridDim.x) {
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % out_channel;
    const int out_b = thread_id / out_width / out_height / out_channel;
    const int in_c = out_c;

    const int in_row_start = out_h * stride_height - pad_height;
    const int in_col_start = out_w * stride_width - pad_width;
    const int in_row_end = in_row_start + filter_height;
    const int in_col_end = in_col_start + filter_width;

    const int out_backprop_offset =
        (out_b * out_channel * out_height * out_width) +
        (out_c * out_height * out_width) + (out_h * out_width) + (out_w);

    const scalar_t out_bp = __ldg(out + out_backprop_offset);

    int f_h = threadIdx.y;
    int f_w = threadIdx.z;

    if (in_row_start >= 0 && in_col_start >= 0 && in_row_end < in_height &&
        in_col_end < in_width) {
      const int in_row = in_row_start + f_h;
      const int input_offset_temp =
          (out_b * in_channel * in_height * in_width) +
          (in_c * in_height * in_width) + (in_row * in_width);
      const int filter_backprop_temp =
          (in_c * filter_width * filter_height) + (filter_width * f_h);

      const int in_col = in_col_start + f_w;
      const int input_offset = input_offset_temp + in_col;
      scalar_t partial_sum = __ldg(input + input_offset) * out_bp;
      scalar_t *addr = kernel + (filter_backprop_temp + f_w);
      atomicAdd(addr, partial_sum);

    } else {

      const int in_row = in_row_start + f_h;
      const int input_offset_temp =
          (out_b * in_channel * in_height * in_width) +
          (in_c * in_height * in_width) + (in_row * in_width);
      const int filter_backprop_temp =
          (in_c * filter_width * filter_height) + (filter_width * f_h);

      const int in_col = in_col_start + f_w;

      if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
          in_col < in_width) {
        const int input_offset = input_offset_temp + in_col;
        scalar_t partial_sum = __ldg(input + input_offset) * out_bp;
        scalar_t *addr = kernel + (filter_backprop_temp + f_w);
        atomicAdd(addr, partial_sum);
      }
    }
  }
}*/

template <typename scalar_t, int kFilterHeight, int kFilterWidth>
__global__ void dwconv2d_backward_kernel_kernel(const scalar_t *out,
                                                const scalar_t *input,
                                                scalar_t *kernel,
                                                const DWConv2dKernelParams p) {
  scalar_t s = 0;

  int gbid = ((blockIdx.z * gridDim.y) + blockIdx.y) * gridDim.x + blockIdx.x;

  for (int image_w = threadIdx.x; image_w < p.out_w; image_w += blockDim.x) {
    for (int bid = 0; bid < p.batch; ++bid) {
      for (int image_h = threadIdx.y; image_h < p.out_h;
           image_h += blockDim.y) {
        int kernel_id = blockIdx.z;
        int kernel_h = blockIdx.y - p.pad_y0;
        int kernel_w = blockIdx.x - p.pad_x0;

        int image_hk = image_h * p.down_y + kernel_h;
        int image_wk = image_w * p.down_x + kernel_w;

        if (image_hk < 0 || image_hk >= p.in_h) {
          continue;
        }

        if (image_wk < 0 || image_wk >= p.in_w) {
          continue;
        }

        int input_id =
            ((bid * gridDim.z + kernel_id) * p.in_h + image_hk) * p.in_w +
            image_wk;

        s += out[((bid * gridDim.z + kernel_id) * p.out_h + image_h) * p.out_w +
                 image_w] *
             input[input_id];
      }
    }
  }

  typedef cub::WarpReduce<scalar_t> WarpReduce;
  typename WarpReduce::TempStorage temp_storage;

  scalar_t val = WarpReduce(temp_storage).Sum(s);
  if (cub::LaneId() == 0) {
    atomicAdd(&kernel[gbid], val);
  }
}

template <typename scalar_t, DepthwiseConv2dDirection kDirection,
          int kBlockSlices, bool kEvenHeight, int kFilterHeight,
          int kFilterWidth>
__global__ void __launch_bounds__(1024, 2)
    dwconv2d_small_kernel(scalar_t *out, const scalar_t *input,
                          const scalar_t *kernel,
                          const DWConv2dKernelParams p) {
  extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_memory[];
  scalar_t *const shared_data = reinterpret_cast<scalar_t *>(shared_memory);
  // extern __shared__ __align__(sizeof(scalar_t)) scalar_t shared_data[];

  const int in_height = p.in_h;
  const int in_width = p.in_w;
  const int in_channel = p.in_channel;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : p.kernel_h;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : p.kernel_w;
  const int pad_height = p.pad_y0;
  const int pad_width = p.pad_x0;

  const int block_height = blockDim.y;

  const int block_pixels = in_width * block_height;
  const int block_size = block_pixels * kBlockSlices;
  const int in_pixels = in_width * in_height;
  const int in_increment = in_width - 1;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int even_height = kEvenHeight || (1 & ~in_height);
  const int tile_height = in_height + filter_height - even_height;
  const int tile_pixels = tile_width * tile_height;
  const int tile_size = tile_pixels * kBlockSlices;
  const int tile_offset = block_height * tile_width;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int in_slices = in_channel * p.batch;
  const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;

  const int thread_width = threadIdx.x;
  const int thread_height = threadIdx.y;
  const int thread_channel = threadIdx.z;

  const int thread_pix = thread_height * in_width + thread_width;
  const int thread_idx = thread_channel * block_pixels + thread_pix;

  for (int i = thread_idx; i < tile_size; i += block_size) {
    shared_data[i] = scalar_t(0);
  }

  __syncthreads();

  const int tensor_idx = thread_channel * in_pixels + thread_pix;

  const int data_pix = thread_height * tile_width + thread_width;
  const int data_idx = thread_channel * tile_pixels + data_pix;

  const int tile_idx = data_idx + pad_offset;

  const int filter_pix = thread_pix;
  const int filter_channel = thread_channel;
  const int filter_idx = filter_pixels * filter_channel + filter_pix;

  const int max_slice = in_slices - thread_channel;
  const int filter_write_offset =
      filter_pix < filter_pixels ? tile_size + filter_idx : 0;
  const int filter_read_offset =
      tile_size + (kDirection == DIRECTION_FORWARD
                       ? filter_pixels * filter_channel
                       : filter_pixels * (filter_channel + 1));
  const bool skip_second =
      !kEvenHeight && thread_height + (in_height & 1) == block_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int slice = b * kBlockSlices;

    const int inout_offset = slice * in_pixels + tensor_idx;
    const bool slice_in_range = slice < max_slice;

    if (slice_in_range) {
      const scalar_t *const in_ptr = inout_offset + input;
      scalar_t *const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = __ldg(in_ptr);

      if (!skip_second) {
        tile_ptr[tile_offset] = __ldg(block_pixels + in_ptr);
      }
    }

    if (filter_write_offset != 0) {
      const int filter_offset =
          ((slice + filter_channel) % in_channel) * filter_pixels + filter_pix;
      shared_data[filter_write_offset] = __ldg(filter_offset + kernel);
    }

    __syncthreads();

    if (slice_in_range) {
      scalar_t sum1 = 0;
      scalar_t sum2 = 0;
      int shared_offset = data_idx;
      const scalar_t *filter_ptr = filter_read_offset + shared_data;

#pragma unroll
      for (int r = 0; r < filter_height; ++r) {
#pragma unroll
        for (int c = 0; c < filter_width; ++c) {
          if (kDirection == DIRECTION_BACKWARD) {
            filter_ptr--;
          }

          const scalar_t filter_value = *filter_ptr;
          const scalar_t *const tile_ptr = shared_offset + shared_data;

          sum1 += filter_value * tile_ptr[0];
          sum2 += filter_value * tile_ptr[tile_offset];
          ++shared_offset;

          if (kDirection == DIRECTION_FORWARD) {
            filter_ptr++;
          }
        }

        shared_offset += in_increment;
      }

      scalar_t *const out_ptr = inout_offset + out;

      out_ptr[0] = sum1;

      if (!skip_second) {
        out_ptr[block_pixels] = sum2;
      }
    }

    __syncthreads();
  }
}

template <typename scalar_t, int kBlockSlices, int kAccumPixels,
          int kFilterHeight, int kFilterWidth>
__global__ void __launch_bounds__(1024, 2)
    dwconv2d_backward_kernel_small_kernel(const scalar_t *out,
                                          const scalar_t *input,
                                          scalar_t *kernel,
                                          const DWConv2dKernelParams p) {
  extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_memory[];
  scalar_t *const shared_data = reinterpret_cast<scalar_t *>(shared_memory);

  const int in_height = p.in_h;
  const int in_width = blockDim.x;
  const int in_channel = p.in_channel;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : p.kernel_h;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : p.kernel_w;
  const int pad_height = p.pad_y0;
  const int pad_width = p.pad_x0;

  const int block_height = blockDim.y;

  const int block_pixels = in_width * block_height;
  const int block_size = block_pixels * kBlockSlices;
  const int in_pixels = in_width * in_height;
  const int in_increment = in_width - 1;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int tile_height = 2 * block_height + filter_height - 1;
  const int tile_pixels = tile_width * tile_height;
  const int tile_size = tile_pixels * kBlockSlices;
  const int tile_offset = block_height * tile_width;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int in_slices = in_channel * p.batch;
  const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;
  const int accum_increment = kAccumPixels * kBlockSlices;
  const int accum_size = filter_pixels * accum_increment;

  const int thread_width = threadIdx.x;
  const int thread_height = threadIdx.y;
  const int thread_channel = threadIdx.z;

  const int thread_pix = thread_height * in_width + thread_width;
  const int thread_idx = thread_channel * block_pixels + thread_pix;

  for (int i = thread_idx; i < tile_size + accum_size; i += block_size) {
    shared_data[i] = scalar_t(0);
  }

  __syncthreads();

  const int tensor_idx = thread_channel * in_pixels + thread_pix;
  const int data_pix = thread_height * tile_width + thread_width;
  const int data_idx = thread_channel * tile_pixels + data_pix;

  const int tile_idx = data_idx + pad_offset;

  const int accum_pix = thread_pix / (32 / kBlockSlices);
  const int accum_idx = thread_channel * kAccumPixels + accum_pix;

  const int max_slice = in_slices - thread_channel;
  const int accum_offset = tile_size + accum_idx;
  const bool skip_second = block_height + thread_height >= in_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int slice = b * kBlockSlices;

    const int inout_offset = slice * in_pixels + tensor_idx;
    const bool slice_in_range = slice < max_slice;

    if (slice_in_range) {
      const scalar_t *const in_ptr = inout_offset + input;
      scalar_t *const tile_ptr = tile_idx + shared_data;

      tile_ptr[0] = __ldg(in_ptr);

      if (!skip_second) {
        tile_ptr[tile_offset] = __ldg(block_pixels + in_ptr);
      }
    }

    __syncthreads();

    CREATE_SHFL_MASK(active_threads, slice_in_range);

    if (slice_in_range) {
      const scalar_t *const out_ptr = inout_offset + out;
      const scalar_t out1 = __ldg(out_ptr);
      const scalar_t out2 =
          skip_second ? scalar_t(0) : __ldg(block_pixels + out_ptr);
      int shared_offset = data_idx;
      scalar_t *accum_ptr = accum_offset + shared_data;

#pragma unroll
      for (int r = 0; r < filter_height; ++r) {
#pragma unroll
        for (int c = 0; c < filter_width; ++c) {
          const scalar_t *const tile_ptr = shared_offset + shared_data;
          scalar_t val = out1 * tile_ptr[0] + out2 * tile_ptr[tile_offset];

          for (int delta = 16 / kBlockSlices; delta > 0; delta /= 2) {
            val += __shfl_xor_sync(active_threads, val, delta);
          }

          if (!(thread_idx & 32 / kBlockSlices - 1)) {
            *accum_ptr = val;
          }

          ++shared_offset;
          accum_ptr += accum_increment;
        }

        shared_offset += in_increment;
      }
    }

    __syncthreads();

    const scalar_t *const accum_data = tile_size + shared_data;

    for (int i = thread_idx; i < accum_size; i += block_size) {
      const int filter_idx = i / kAccumPixels;
      const int filter_pix = filter_idx / kBlockSlices;
      const int filter_channel =
          (slice + filter_idx % kBlockSlices) % in_channel;
      const int filter_offset = filter_channel * filter_pixels +
                                (filter_pix / filter_width) * filter_height +
                                filter_pix % filter_width;

      if (filter_channel < in_channel) {
        scalar_t val = accum_data[i];
        unsigned int lane_id = get_lane_id();
        int sub_warp = lane_id / kAccumPixels;
        int zeros = sub_warp * kAccumPixels;
        unsigned mask = ((1UL << kAccumPixels) - 1) << zeros;

        for (int delta = kAccumPixels / 2; delta > 0; delta /= 2) {
          val += __shfl_xor_sync(mask, val, delta);
        }

        if (!(thread_idx & kAccumPixels - 1)) {
          atomicAdd(filter_offset + kernel, val);
        }
      }
    }
  }
}

DWConv2dKernelParams make_conv2d_params(const torch::Tensor &input,
                                        const torch::Tensor &kernel, int up_h,
                                        int up_w, int down_h, int down_w,
                                        int pad_h0, int pad_h1, int pad_w0,
                                        int pad_w1) {
  DWConv2dKernelParams p;

  p.batch = input.size(0);
  p.in_channel = input.size(1);
  p.in_h = input.size(2);
  p.in_w = input.size(3);
  p.kernel_h = kernel.size(2);
  p.kernel_w = kernel.size(3);
  p.up_x = up_w;
  p.up_y = up_h;
  p.down_x = down_w;
  p.down_y = down_h;
  p.pad_x0 = pad_w0;
  p.pad_x1 = pad_w1;
  p.pad_y0 = pad_h0;
  p.pad_y1 = pad_h1;

  p.out_h = (p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - p.kernel_h + p.down_y) /
            p.down_y;
  p.out_w = (p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - p.kernel_w + p.down_x) /
            p.down_x;
  p.out_channel = p.in_channel;
  p.n_out = p.batch * p.in_channel * p.out_h * p.out_w;

  return p;
}

DWConv2dKernelParams make_conv2d_kernel_backward_params(
    const torch::Tensor &input, const torch::Tensor &out_grad,
    const torch::Tensor &kernel, int up_h, int up_w, int down_h, int down_w,
    int pad_h, int pad_w) {
  DWConv2dKernelParams p;

  p.batch = input.size(0);
  p.in_channel = input.size(1);
  p.in_h = input.size(2);
  p.in_w = input.size(3);
  p.kernel_h = kernel.size(2);
  p.kernel_w = kernel.size(3);
  p.up_x = up_w;
  p.up_y = up_h;
  p.down_x = down_w;
  p.down_y = down_h;
  p.pad_x0 = pad_w;
  p.pad_x1 = pad_w;
  p.pad_y0 = pad_h;
  p.pad_y1 = pad_h;

  p.out_h = out_grad.size(2);
  p.out_w = out_grad.size(3);
  p.out_channel = p.in_channel;

  p.n_out = p.batch * p.out_channel * p.out_h * p.out_w;

  return p;
}

bool use_dwconv2d_small(const torch::Tensor &input, const torch::Tensor &kernel,
                        int up_h, int up_w, int down_h, int down_w, int pad_h,
                        int pad_w) {
  DWConv2dKernelParams p = make_conv2d_params(
      input, kernel, up_h, up_w, down_h, down_w, pad_h, pad_h, pad_w, pad_w);

  return p.down_y == 1 && p.down_x == 1 && p.in_h <= 32 && p.in_w <= 32 &&
         p.in_h == p.out_h && p.in_w == p.out_w && p.pad_y0 >= 0 &&
         p.pad_y0 < p.kernel_h && p.pad_x0 >= 0 && p.pad_x0 < p.kernel_w &&
         p.kernel_h * p.kernel_w <= (p.in_h + 1) / 2 * p.in_w;
}

bool use_dwconv2d_backward_kernel_small(const DWConv2dKernelParams p,
                                        const int block_height) {
  return p.down_x == 1 && p.down_y == 1 && p.in_h <= 32 && p.in_w <= 32 &&
         p.in_h == p.in_w && p.in_w == p.out_w && p.in_h == p.out_h &&
         p.pad_y0 >= 0 && p.pad_y0 < p.kernel_h && p.pad_x0 >= 0 &&
         p.pad_x0 < p.kernel_w && block_height <= p.in_h &&
         p.kernel_h * p.kernel_w <= block_height * p.in_w;
}

template <typename scalar_t, DepthwiseConv2dDirection kDirection,
          int kBlockSlices, bool kEvenHeight>
torch::Tensor dwconv2d_small_op(const torch::Tensor &input,
                                const torch::Tensor &kernel,
                                const DWConv2dKernelParams p) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

  const int block_height = (p.in_h + 1) / 2;
  dim3 block_dim = dim3(p.in_w, block_height, kBlockSlices);

  const int tile_width = p.in_w + p.kernel_w - 1;
  const int tile_height = block_height * 2 + p.kernel_h - 1;
  const int tile_pixels = tile_height * tile_width;
  const int filter_pixels = p.kernel_h * p.kernel_w;
  const int num_outputs = p.batch * p.out_h * p.out_w * p.out_channel;
  int block_count =
      std::min(num_outputs / (block_dim.x * block_dim.y * block_dim.z),
               static_cast<unsigned int>(65535));

  auto out =
      at::empty({p.batch, p.in_channel, p.out_h, p.out_w}, input.options());

  const int shared_memory_size =
      kBlockSlices * (tile_pixels + filter_pixels) * sizeof(scalar_t);

  if (p.kernel_h == 3 && p.kernel_w == 3) {
    dwconv2d_small_kernel<scalar_t, kDirection, kBlockSlices, kEvenHeight, 3, 3>
        <<<block_count, block_dim, shared_memory_size, stream>>>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(), p);
  } else {
    dwconv2d_small_kernel<scalar_t, kDirection, kBlockSlices, kEvenHeight, -1,
                          -1>
        <<<block_count, block_dim, shared_memory_size, stream>>>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(), p);
  }

  return out;
}

template <typename scalar_t, DepthwiseConv2dDirection kDirection,
          int kBlockSlices>
torch::Tensor dwconv2d_small_op(const torch::Tensor &input,
                                const torch::Tensor &kernel,
                                const DWConv2dKernelParams p) {
  torch::Tensor out;

  if (p.in_h & 1) {
    out = dwconv2d_small_op<scalar_t, kDirection, kBlockSlices, false>(
        input, kernel, p);
  } else {
    out = dwconv2d_small_op<scalar_t, kDirection, kBlockSlices, true>(
        input, kernel, p);
  }

  return out;
}

torch::Tensor dwconv2d_small_op(const torch::Tensor &input,
                                const torch::Tensor &kernel, int up_h, int up_w,
                                int down_h, int down_w, int pad_h, int pad_w,
                                bool forward) {
  DWConv2dKernelParams p = make_conv2d_params(
      input, kernel, up_h, up_w, down_h, down_w, pad_h, pad_h, pad_w, pad_w);

  auto x = input.contiguous();
  auto k = kernel.contiguous();

  p.n_out = p.batch * p.in_channel * p.out_h * p.out_w;

  const int block_pixels = (p.in_h + 1) / 2 * p.in_w;
  torch::Tensor out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "dwconv2d_small", [&] {
        if (forward) {
          if (block_pixels > 256) {
            out = dwconv2d_small_op<scalar_t, DIRECTION_FORWARD, 2>(x, k, p);
          } else if (block_pixels > 128) {
            out = dwconv2d_small_op<scalar_t, DIRECTION_FORWARD, 4>(x, k, p);
          } else {
            out = dwconv2d_small_op<scalar_t, DIRECTION_FORWARD, 8>(x, k, p);
          }
        } else {
          if (block_pixels > 256) {
            out = dwconv2d_small_op<scalar_t, DIRECTION_BACKWARD, 2>(x, k, p);
          } else if (block_pixels > 128) {
            out = dwconv2d_small_op<scalar_t, DIRECTION_BACKWARD, 4>(x, k, p);
          } else {
            out = dwconv2d_small_op<scalar_t, DIRECTION_BACKWARD, 8>(x, k, p);
          }
        }
      });

  return out;
}

template <typename scalar_t, DepthwiseConv2dDirection direction, int up_x,
          int up_y, int down_x, int down_y, int kernel_h, int kernel_w,
          int tile_out_h, int tile_out_w>
torch::Tensor dwconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, DWConv2dKernelParams p) {
  int cur_device = -1;
  cudaGetDevice(&cur_device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);

  auto out =
      at::empty({p.batch, p.in_channel, p.out_h, p.out_w}, input.options());

  dim3 block_size;
  dim3 grid_size;

  int major_dim = p.batch * p.in_channel;

  if (tile_out_h > 0 && tile_out_w > 0) {
    p.loop_major = (major_dim - 1) / 16384 + 1;
    block_size = dim3(32 * 8, 1, 1);
    grid_size =
        dim3(((p.out_h - 1) / tile_out_h + 1), (p.out_w - 1) / tile_out_w + 1,
             (major_dim - 1) / p.loop_major + 1);
  }

  dwconv2d_kernel<scalar_t, direction, up_x, up_y, down_x, down_y, kernel_h,
                  kernel_w, tile_out_h, tile_out_w>
      <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                             input.data_ptr<scalar_t>(),
                                             kernel.data_ptr<scalar_t>(), p);

  return out;
}

template <typename scalar_t, DepthwiseConv2dDirection direction>
torch::Tensor dwconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, DWConv2dKernelParams p) {
  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1) {
    if (p.kernel_h <= 3 && p.kernel_w <= 3) {
      return dwconv2d_op<scalar_t, direction, 1, 1, 1, 1, 3, 3, 16, 64>(
          input, kernel, p);

    } else if (p.kernel_h <= 5 && p.kernel_w <= 5) {
      return dwconv2d_op<scalar_t, direction, 1, 1, 1, 1, 5, 5, 16, 64>(
          input, kernel, p);
    } else if (p.kernel_h <= 7 && p.kernel_w <= 7) {
      return dwconv2d_op<scalar_t, direction, 1, 1, 1, 1, 7, 7, 16, 64>(
          input, kernel, p);
    }
  } else if (p.up_x == 2 && p.up_y == 2) {
    if (p.kernel_h <= 4 && p.kernel_w <= 4) {
      return dwconv2d_op<scalar_t, direction, 2, 2, 1, 1, 4, 4, 16, 64>(
          input, kernel, p);
    } else if (p.kernel_h <= 6 && p.kernel_w <= 6) {
      return dwconv2d_op<scalar_t, direction, 2, 2, 1, 1, 6, 6, 16, 64>(
          input, kernel, p);
    } else if (p.kernel_h <= 8 && p.kernel_w <= 8) {
      return dwconv2d_op<scalar_t, direction, 2, 2, 1, 1, 8, 8, 16, 64>(
          input, kernel, p);
    }
  } else if (p.down_x == 2 && p.down_y == 2) {
    if (p.kernel_h <= 4 && p.kernel_w <= 4) {
      return dwconv2d_op<scalar_t, direction, 1, 1, 2, 2, 4, 4, 8, 32>(
          input, kernel, p);
    } else if (p.kernel_h <= 6 && p.kernel_w <= 6) {
      return dwconv2d_op<scalar_t, direction, 1, 1, 2, 2, 6, 6, 8, 32>(
          input, kernel, p);
    } else if (p.kernel_h <= 8 && p.kernel_w <= 8) {
      return dwconv2d_op<scalar_t, direction, 1, 1, 2, 2, 8, 8, 8, 32>(
          input, kernel, p);
    }
  }
}

torch::Tensor dwconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, int up_h, int up_w,
                          int down_h, int down_w, int pad_h0, int pad_h1,
                          int pad_w0, int pad_w1, bool forward) {
  DWConv2dKernelParams p =
      make_conv2d_params(input, kernel, up_h, up_w, down_h, down_w, pad_h0,
                         pad_h1, pad_w0, pad_w1);

  auto x = input.contiguous();
  auto k = kernel.contiguous();

  torch::Tensor out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "dwconv2d", [&] {
    if (forward) {
      out = dwconv2d_op<scalar_t, DIRECTION_FORWARD>(x, k, p);
    } else {
      out = dwconv2d_op<scalar_t, DIRECTION_BACKWARD>(x, k, p);
    }
  });

  return out;
}

template <typename scalar_t, int kBlockSlices, int kAccumPixels>
bool dwconv2d_backward_kernel_small_op(const torch::Tensor &input,
                                       const torch::Tensor &out_grad,
                                       torch::Tensor &kernel_grad,
                                       const DWConv2dKernelParams p,
                                       const int block_height) {
  const int tile_width = p.in_w + p.kernel_w - 1;
  const int tile_height = block_height * 2 + p.kernel_h - 1;
  const int tile_pixels = tile_height * tile_width;
  const int filter_pixels = p.kernel_h * p.kernel_w;
  const int shared_memory_size = kBlockSlices *
                                 (tile_pixels + filter_pixels * kAccumPixels) *
                                 sizeof(scalar_t);

  if (shared_memory_size > 46 * 1024) {
    return false;
  }

  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

  dim3 block_dim = dim3(p.in_w, block_height, kBlockSlices);
  int block_count = p.n_out / (block_dim.x * block_dim.y * block_dim.z) + 1;

  if (p.kernel_h == 3 && p.kernel_w == 3) {
    dwconv2d_backward_kernel_small_kernel<scalar_t, kBlockSlices, kAccumPixels,
                                          3, 3>
        <<<block_count, block_dim, shared_memory_size, stream>>>(
            out_grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            kernel_grad.data_ptr<scalar_t>(), p);
  } else {
    dwconv2d_backward_kernel_small_kernel<scalar_t, kBlockSlices, kAccumPixels,
                                          -1, -1>
        <<<block_count, block_dim, shared_memory_size, stream>>>(
            out_grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            kernel_grad.data_ptr<scalar_t>(), p);
  }

  return true;
}

template <typename scalar_t, int kBlockSlices>
bool dwconv2d_backward_kernel_small_op(const torch::Tensor &input,
                                       const torch::Tensor &out_grad,
                                       torch::Tensor &kernel_grad,
                                       const DWConv2dKernelParams p,
                                       const int block_height) {
  const int block_pixels = block_height * p.in_w * kBlockSlices;

  if (block_pixels > 512) {
    return dwconv2d_backward_kernel_small_op<scalar_t, kBlockSlices, 32>(
        input, out_grad, kernel_grad, p, block_height);
  } else if (block_pixels > 256) {
    return dwconv2d_backward_kernel_small_op<scalar_t, kBlockSlices, 16>(
        input, out_grad, kernel_grad, p, block_height);
  } else {
    return dwconv2d_backward_kernel_small_op<scalar_t, kBlockSlices, 8>(
        input, out_grad, kernel_grad, p, block_height);
  }
}

bool dwconv2d_backward_kernel_small_op(const torch::Tensor &input,
                                       const torch::Tensor &out_grad,
                                       torch::Tensor &kernel_grad,
                                       const DWConv2dKernelParams p) {
  int block_slices = 8;
  int block_height = (p.in_h + 1) / 2;
  int round_mask = 1;

  for (; block_slices > 1; block_slices /= 2) {
    for (; block_height * p.in_w * block_slices & 31;
         round_mask = round_mask * 2 + 1) {
      block_height = block_height + round_mask & ~round_mask;
    }

    int block_size = block_height * p.in_w * block_slices;

    if (block_size <= 1024) {
      break;
    }
  }

  if (!use_dwconv2d_backward_kernel_small(p, block_height)) {
    return false;
  }

  bool success;

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dwconv2d_backward_kernel_small", [&] {
        switch (block_slices) {
        case 8:
          success = dwconv2d_backward_kernel_small_op<scalar_t, 8>(
              input, out_grad, kernel_grad, p, block_height);
          break;

        case 4:
          success = dwconv2d_backward_kernel_small_op<scalar_t, 4>(
              input, out_grad, kernel_grad, p, block_height);
          break;

        case 2:
          success = dwconv2d_backward_kernel_small_op<scalar_t, 2>(
              input, out_grad, kernel_grad, p, block_height);
          break;

        default:
          success = false;
        }
      });

  return success;
}

void dwconv2d_backward_kernel_op(const torch::Tensor &input,
                                 const torch::Tensor &out_grad,
                                 torch::Tensor &kernel_grad,
                                 DWConv2dKernelParams p) {

  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

  dim3 block_num = dim3(p.kernel_w, p.kernel_h, p.out_channel);
  dim3 thread_num = dim3(std::min(p.out_w, 512),
                         std::min(std::max(512 / p.out_w, 1), p.out_h), 1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "dwconv2d_backward_kernel", [&] {
        if (p.kernel_h == 3 && p.kernel_w == 3) {
          dwconv2d_backward_kernel_kernel<scalar_t, 3, 3>
              <<<block_num, thread_num, 0, stream>>>(
                  out_grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  kernel_grad.data_ptr<scalar_t>(), p);
        } else {
          dwconv2d_backward_kernel_kernel<scalar_t, -1, -1>
              <<<block_num, thread_num, 0, stream>>>(
                  out_grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  kernel_grad.data_ptr<scalar_t>(), p);
        }
      });
}

torch::Tensor dwconv2d_backward_kernel_op(const torch::Tensor &input,
                                          const torch::Tensor &out_grad,
                                          const torch::Tensor &kernel, int up_h,
                                          int up_w, int down_h, int down_w,
                                          int pad_h, int pad_w) {
  DWConv2dKernelParams p = make_conv2d_kernel_backward_params(
      input, out_grad, kernel, up_h, up_w, down_h, down_w, pad_h, pad_w);

  using namespace std::chrono;

  auto x = input.contiguous();
  auto grad = out_grad.contiguous();

  auto kernel_grad =
      at::zeros({p.in_channel, 1, p.kernel_h, p.kernel_w}, input.options());

  if (dwconv2d_backward_kernel_small_op(x, grad, kernel_grad, p)) {
    return kernel_grad;
  }

  dwconv2d_backward_kernel_op(x, grad, kernel_grad, p);

  return kernel_grad;
}