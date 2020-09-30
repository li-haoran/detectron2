/*!
 *****************
 * COPYRIGHT
 * 
 * LICENSE
 * 
 * author: haoran li
 
 */

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#include <ATen/ATen.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <THC/THCAtomics.cuh>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int h,
    const int w,
    const int height,
    const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int height,
    const int width,
    const scalar_t* im_data,
    const int data_width,
    const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void scatter_img2inst_gpu_kernel(
    const int n,
    const scalar_t* data_feature_,
    const scalar_t* data_sample_offsets_,
    const int* data_batch_index_,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    scalar_t* data_col_,
    scalar_t* data_count_) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w = index % width_out;
    const int h = index / width_out % height_out;
    const int c = index / width_out / height_out % channel_out;
    const int n = index / width_out / height_out / channel_out;

    const int b = data_batch_index_[n];

    for (p = 0; p < num_points; p++) {
      const int data_offset_h_ptr = (n * num_points + p) * 2;
      const int data_offset_w_ptr = (n * num_points + p) * 2 + 1;
      const scalar_t h_im = data_sample_offsets_[data_offset_h_ptr];
      const scalar_t w_im = data_sample_offsets_[data_offset_w_ptr];
      scalar_t avgval = 0;
      scalar_t count = 0;

      if (abs(h_im - h) < 1 && abs(w_im - w) < 1) {
        scalar_t weight =
            get_gradient_weight(h_im, w_im, h, w, height_out, width_out);
        const int data_feature_index = b * channel_out * height_out * width +
            c * height_out * width_out + h * width_out + w;
        avgval += (weight * data_feature_[data_batch_index]);
        count += 1;
      }
      if (count > 0) {
        const int data_col_index = n * channel_out * height_out * width +
            c * height_out * width_out + h * width_out + w;
        data_col_[data_col_index] = avgval / count;
        data_count_[data_col_index] = count;
      }
    }
  }
}

void scatter_img2inst(
    const at::Tensor data_feature,
    const at::Tensor data_sample_offsets,
    const at::Tensor data_batch_index,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    at::Tensor data_col,
    at::Tensor output_count) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int num_kernels = num_instance * channel_out * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "scatter_img2inst_gpu", ([&] {
        const scalar_t* data_feature_ = data_feature.data<scalar_t>();
        const scalar_t* data_sample_offsets_ =
            data_sample_offsets.data<scalar_t>();
        const int* data_batch_index_ =
            data_batch_index.data<int>() scalar_t* data_col_ =
                data_col.data<scalar_t>();
        scalar_t* data_count_ = output_count.data<scalar_t>();

        scatter_img2inst_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            data_feature_,
            data_sample_offsets_,
            data_batch_index_,
            num_instance,
            num_points,
            channel_out,
            height_out,
            width_out,
            data_col_,
            data_count_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in scatter_img2inst: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void scatter_inst2img_gpu_kernel(
    const int n,
    const scalar_t* data_feature_,
    const scalar_t* data_sample_offsets_,
    const int* data_batch_index_,
    const scalar_t* data_count_,
    const int num_instance,
    const int num_points,
    const int channel_out,
    const int height_out,
    const int width_out,
    scalar_t* grad_feature_,
    scalar_t* grad_sample_offsets_,
    const scalar_t* grad_output_) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w = index % width_out;
    const int h = index / width_out % height_out;
    const int c = index / width_out / height_out % channel_out;
    const int n = index / width_out / height_out / channel_out;

    const int b = data_batch_index_[n];

    const int grad_output_index = n * channel_out * height_out * width +
        c * height_out * width_out + h * width_out + w;
    const scalar_t grad_val = grad_output_[grad_output_index];
    const scalar_t num_count = data_count_[grad_output_index];

    const int grad_feature_index = b * channel_out * height_out * width +
        c * height_out * width_out + h * width_out + w;

    const scalar_t* data_feature_ptr = data_feature_ +
        (b * channel_out * height_out * width + c * height_out * width_out);

    if (num_count > 0) {
      for (p = 0; p < num_points; p++) {
        const int data_offset_h_ptr = (n * num_points + p) * 2;
        const int data_offset_w_ptr = (n * num_points + p) * 2 + 1;
        const scalar_t h_im = data_sample_offsets_[data_offset_h_ptr];
        const scalar_t w_im = data_sample_offsets_[data_offset_w_ptr];

        if (abs(h_im - h) < 1 && abs(w_im - w) < 1) {
          scalar_t weight =
              get_gradient_weight(h_im, w_im, h, w, height_out, width_out);
          atomicAdd(
              grad_feature_ + grad_feature_index,
              weight * grad_val / num_count);

          scalar_t h_weight = get_coordinate_weight(
              h_im,
              w_im,
              height_out,
              width_out,
              data_feature_ptr,
              width_out,
              0);
          scalar_t w_weight = get_coordinate_weight(
              h_im,
              w_im,
              height_out,
              width_out,
              data_feature_ptr,
              width_out,
              1);
          atomicAdd(
              grad_sample_offsets_ + data_offset_h_ptr,
              h_weight * grad_val / num_count);
          atomicAdd(
              grad_sample_offsets_ + data_offset_w_ptr,
              w_weight * grad_val / num_count);
        }
      }
    }
  }
}

void scatter_inst2img(
    const at::Tensor data_feature,
    const at::Tensor data_sample_offsets,
    const at::Tensor data_batch_index,
    const at::Tensor data_output_count,
    const int num_instance,
    const int num_points,
    const int channel_out const int height_out,
    const int width_out,
    at::Tensor grad_feature,
    at::Tensor grad_sample_offsets,
    const at::Tensor grad_output) {
  int num_kernels = num_instance * channel_out * height_out * width_out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "scatter_inst2img_gpu", ([&] {
        const scalar_t* grad_output_ = grad_output.data<scalar_t>();
        const scalar_t* data_feature_ = data_feature.data<scalar_t>();
        const scalar_t* data_sample_offsets_ =
            data_sample_offsets.data<scalar_t>();
        const int* data_batch_index_ = data_batch_index.data<int>();
        const scalar_t* data_count_ = data_output_count.data<scalar_t>();
        scalar_t* grad_feature_ = grad_feature.data<scalar_t>();
        scalar_t* grad_sample_offsets_ = grad_sample_offsets<scalar_t>();

        scatter_inst2img_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS>>>(
            num_kernels,
            data_feature_,
            data_sample_offsets_,
            data_batch_index_,
            data_count_,
            num_instance,
            num_points,
            channel_out,
            height_out,
            width_out,
            grad_feature_,
            grad_sample_offsets_,
            grad_output_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in scatter_inst2img: %s\n", cudaGetErrorString(err));
  }
}
