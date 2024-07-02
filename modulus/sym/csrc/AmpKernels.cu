#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

using torch::Tensor;

// Modified from https://github.com/pytorch/pytorch/blob/release/1.11/aten/src/ATen/native/cuda/AmpKernels.cu#L181

// amp_update_scale_cuda_kernel is launched with a single thread to compute the new scale.
// The scale factor is maintained and updated on the GPU to avoid synchronization.
__global__ void amp_update_scale_cuda_kernel(float* current_scale,
                                             int* growth_tracker,
                                             float* found_inf,
                                             float growth_factor,
                                             float backoff_factor,
                                             int growth_interval,
                                             float max_scale,
                                             float recover_threshold,
                                             int recover_growth_interval) {
  if (*found_inf) {
    *current_scale = (*current_scale)*backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    // decide whether to use the recover_growth_interval
    growth_interval = (*current_scale) <= recover_threshold ? recover_growth_interval : growth_interval;
    if (successful == growth_interval) {
      // grow the scale then clamp with max_scale
      *current_scale = min((*current_scale) * growth_factor, max_scale);
      *growth_tracker = 0;
    } else {
      *growth_tracker = successful;
    }
  }
}

// _amp_update_scale_cuda asynchronously updates the scale tensor in place.
//
// Args:
// current_scale:  A one-element cuda float tensor containing the scale value.
// growth_tracker:  A one-element torch.cuda.IntTensor containing the number of recent consecutive unskipped steps.
// found_inf:  A one-element cuda float tensor. If > 0, indicates that infs/nans were found by the relevant
//             prior _amp_non_finite_check_and_unscale_cuda call, and 0 if no infs/nans were found.
// growth_factor:  Multiplier if no infs/NaNs were found (typically slightly > 1).
// backoff_factor:  Multiplier if infs/NaNs were found (typically 0.5).
// growth_interval:  Number of consecutive unskipped steps that must occur for current_scale to be multiplied by
//                   growth_factor.
// max_scale:  The maximum value the scale could grow.
// recover_threshold:  Allowing quickly recover the scaling factor when it is less or equal than this threshold.
// recover_growth_interval:  The growth_interval that will be used when the scaling factor is less or equal than
//                           the recover_threshold.
//
// Returns:
// current_scale
Tensor& _amp_update_scale_cuda_(Tensor& current_scale,
                                Tensor& growth_tracker,
                                const Tensor& found_inf,
                                double growth_factor,
                                double backoff_factor,
                                int64_t growth_interval,
                                double max_scale,
                                double recover_threshold,
                                int64_t recover_growth_interval)
{
  TORCH_CHECK(growth_tracker.is_cuda(), "growth_tracker must be a CUDA tensor.");
  TORCH_CHECK(current_scale.is_cuda(), "current_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(growth_tracker.numel() == 1, "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int, "growth_tracker must be an int tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  amp_update_scale_cuda_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
    current_scale.data_ptr<float>(),
    growth_tracker.data_ptr<int>(),
    found_inf.data_ptr<float>(),
    growth_factor,
    backoff_factor,
    growth_interval,
    max_scale,
    recover_threshold,
    recover_growth_interval);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return current_scale;
}

TORCH_LIBRARY(modulus_ext, m) {
  m.def("_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval, float max_scale, float recover_threshold, int recover_growth_interval) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(modulus_ext, CUDA, m) {
  m.impl("_amp_update_scale_", _amp_update_scale_cuda_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
