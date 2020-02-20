#include <torch/extension.h>

bool use_dwconv2d_small(const torch::Tensor &input, const torch::Tensor &kernel,
                        int up_h, int up_w, int down_h, int down_w, int pad_h,
                        int pad_w);

torch::Tensor dwconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, int up_h, int up_w,
                          int down_h, int down_w, int pad_h0, int pad_h1,
                          int pad_w0, int pad_w1, bool forward);

torch::Tensor dwconv2d_small_op(const torch::Tensor &input,
                                const torch::Tensor &kernel, int up_h, int up_w,
                                int down_h, int down_w, int pad_h, int pad_w,
                                bool forward);
;

torch::Tensor dwconv2d_backward_kernel_op(const torch::Tensor &input,
                                          const torch::Tensor &out_grad,
                                          const torch::Tensor &kernel, int up_h,
                                          int up_w, int down_h, int down_w,
                                          int pad_h, int pad_w);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

torch::Tensor dwconv2d(const torch::Tensor &input, const torch::Tensor &kernel,
                       int up_h, int up_w, int down_h, int down_w, int pad_h0,
                       int pad_h1, int pad_w0, int pad_w1, bool forward) {
  CHECK_CUDA(input);
  CHECK_CUDA(kernel);

  return dwconv2d_op(input, kernel, up_h, up_w, down_h, down_w, pad_h0, pad_h1,
                     pad_w0, pad_w1, forward);
}

torch::Tensor dwconv2d_small(const torch::Tensor &input,
                             const torch::Tensor &kernel, int up_h, int up_w,
                             int down_h, int down_w, int pad_h, int pad_w,
                             bool forward) {
  CHECK_CUDA(input);
  CHECK_CUDA(kernel);

  return dwconv2d_small_op(input, kernel, up_h, up_w, down_h, down_w, pad_h,
                           pad_w, forward);
}

torch::Tensor dwconv2d_backward_kernel(const torch::Tensor &input,
                                       const torch::Tensor &out_grad,
                                       const torch::Tensor &kernel, int up_h,
                                       int up_w, int down_h, int down_w,
                                       int pad_h, int pad_w) {
  CHECK_CUDA(input);
  CHECK_CUDA(out_grad);

  return dwconv2d_backward_kernel_op(input, out_grad, kernel, up_h, up_w,
                                     down_h, down_w, pad_h, pad_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("use_dwconv2d_small", &use_dwconv2d_small,
        "check availability of dwconv2d small");
  m.def("dwconv2d", &dwconv2d, "dwconv2d (CUDA)");
  m.def("dwconv2d_small", &dwconv2d_small, "dwconv2d small (CUDA)");
  m.def("dwconv2d_backward_kernel", &dwconv2d_backward_kernel,
        "dwconv2d backward kernel (CUDA)");
}