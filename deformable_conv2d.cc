#include <torch/extension.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
extern THCState *state;
#include <vector>
#include <stdio.h>
typedef std::vector<int> TShape;
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline int ProdShape(const TShape &shape, int start, int end) {
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape[i];
    }
    return res;
}

inline TShape SubVector(const TShape &shape, int start, int end) {
    TShape res;
    for(int i=start;i<end;i++){
        res.push_back(shape[i]);
    }
    return res;
}

#ifndef FUNCTION_DECLARE
#define FUNCTION_DECLARE

    void deformable_im2col(cudaStream_t stream,
         const float* data_im, const float* data_offset, const float* data_mask,
         const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
         const TShape& pad, const TShape& stride, const TShape& dilation,
         const int deformable_group, float* data_col);

    void deformable_col2im(cudaStream_t stream,
            const float* data_col, const float* data_offset, const float* data_mask,
            const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
            const TShape& pad, const TShape& stride,
            const TShape& dilation, const int deformable_group,
            float* grad_im);

    void deformable_col2im_coord(cudaStream_t stream,
            const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
            const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
            const TShape& pad, const TShape& stride,
            const TShape& dilation, const int deformable_group,
            float* grad_offset, float* grad_mask);

    void setZero(cudaStream_t stream, int n, float* result_data);

    void setOne(cudaStream_t stream, int n, float* result_data);

    void pureAddTo(cudaStream_t stream, const int n, float* result_data, const float* right_data);

    void setNumAtIndex(cudaStream_t stream,  float num, int index, float* data);

    void SwapAxis(cudaStream_t stream, float* input_data, const TShape& origin_shape, const int axis_x, const int axis_y);

#endif

at::Tensor deformable_conv2d_forward(
    at::Tensor input,
    at::Tensor filter,
    at::Tensor offset,
    at::Tensor mask,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int num_groups,
    int deformable_groups,
    int im2col_step,
    bool no_bias
){
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(filter.type().is_cuda(), "filter must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");

    const int height = input.size(2);
    const int width = input.size(3);
    int kernel_h = filter.size(2);
    int kernel_w = filter.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int num_axes = 4;
    bool is_1x1_ = true;
    for (int i = 2; i < num_axes; ++i) {
            is_1x1_ &= filter.size(i) == 1; // only judge by the filter's shape
            if (!is_1x1_) break;
    }
    int num_ = input.size(0);// batch size
    int channels_ = input.size(1);// number of input channels
    int group_ = num_groups;//
    int conv_out_channels_ = filter.size(0); // output channel nums
    int conv_in_channels_ = channels_; // input channel nums

    int kernel_dim_ = conv_in_channels_ / group_ * filter.size(2) * filter.size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    int conv_out_spatial_dim_ = height_out * width_out;
    int im2col_step_ = std::min(im2col_step, num_);

    int input_dim_ = input.size(1) * input.size(2) * input.size(3);// input image size (#channels * height * width)
    int input_offset_dim_ = offset.size(1) * offset.size(2) * offset.size(3); // 18 * H * W
    int input_mask_dim_ =  mask.size(1) * mask.size(2) * mask.size(3); // 9 * H * W

    int M = conv_out_channels_ / group_; // filter的数量
    int N = im2col_step_ * conv_out_spatial_dim_;
    int K = kernel_dim_;

    auto col_buffer = at::empty({conv_in_channels_ * filter.size(2) * filter.size(3), im2col_step_, conv_out_spatial_dim_}, input.options());
    auto output = at::empty({num_, conv_out_channels_, height_out, width_out}, input.options());

    auto input_ptr = input.data<float>();
    auto weight_ptr = filter.data<float>();
    auto offset_ptr = offset.data<float>();
    auto mask_ptr = mask.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();
    auto output_ptr = output.data<float>();

    TShape input_shape;
    TShape filter_shape;
    TShape col_buffer_shape;
    TShape stride_shape;
    TShape dilation_shape;
    TShape padding_shape;

    input_shape.push_back(input.size(0));
    input_shape.push_back(input.size(1));
    input_shape.push_back(input.size(2));
    input_shape.push_back(input.size(3));
    filter_shape.push_back(filter.size(2));
    filter_shape.push_back(filter.size(3));
    col_buffer_shape.push_back(conv_in_channels_ * filter.size(2) * filter.size(3));
    col_buffer_shape.push_back(im2col_step_);
    col_buffer_shape.push_back(height_out);
    col_buffer_shape.push_back(width_out);
    stride_shape.push_back(stride_h);
    stride_shape.push_back(stride_w);
    dilation_shape.push_back(dilation_h);
    dilation_shape.push_back(dilation_w);
    padding_shape.push_back(pad_h);
    padding_shape.push_back(pad_w);

    for (int n = 0; n < num_ / im2col_step_; ++n) {
            deformable_im2col(
            THCState_getCurrentStream(state),
            input_ptr + n * im2col_step_ * input_dim_,
            offset_ptr + n * im2col_step_ * input_offset_dim_,
            mask_ptr + n * im2col_step_ * input_mask_dim_,
            input_shape,
            col_buffer_shape,
            filter_shape,
            padding_shape,
            stride_shape,
            dilation_shape,
            deformable_groups,
            col_buffer_ptr
            );
            for(int g = 0; g < group_;g++){
                auto output_instance_ptr = output_ptr + (n * group_ * M  * N) + g * M * N; //{num_ / im2col_step_, group_, M, N}
                auto weight_tmp_ptr = weight_ptr + g * M * K;
                auto col_buffer_tmp_ptr = col_buffer_ptr + g * K * N;
//      这里0.0f我一开始设置为1.0f, 这个其实叫做主机指针或者设备指针, 只需要计算矩阵乘法时命 β = 0.0f
//    两个“是否需要对输入矩阵 A、B 进行转置”的参数，这是 cuBLAS 库难点之一。简单地说，cuBLAS 中关于矩阵的存储方式与 fortran、MATLAB类似，采用的是列优先，而非 C / C++ 中的行优先。
//     所以，当我们将 C / C++ 中行优先形式保存的数组 A 输入到cuBLAS中时，会被cuBLAS理解为列优先存储。
//   这时如果保持 A 的行、列数不变，则矩阵 A 会发生重排（过程类似 MATLAB 中的 reshape(A, [size(A,2), size(A, 1)])），
//   除非同时交换 A 的行、列数，此时结果才恰好等于 A 的转置，在一般的调用过程中正是利用了这一条性质。
//   yu是 cuBLAS 事先利用这个参数询问，是否需要将矩阵 A、B 进行转置。在这里，我尝试了大量的例子，结合图形来说明cuBLAS中对数组的操作。
//（正确过程）这一般教程上的调用过程。利用了性质 A B = (BT AT)T 来计算矩阵乘积。如前文所述，我们把一个行优先的矩阵看作列优先的同时，
//交换其行、列数，其结果等价于得到该矩阵的转置，反之列优先转行优先的原理相同。所以我们可以在调用该函数的时候，先放入 B 并交换参数 k 和 n 的位置，
//再放入 A 并交换参数 m 和 k 的位置，这样就顺理成章得到了结果的 C （所有转换由cuBLAS完成，不需要手工调整数组），注意以下调用语句中红色的部分。
// 具体见https://www.cnblogs.com/cuancuancuanhao/p/7763256.html
                THCudaBlas_Sgemm(state, 'n', 'n', N, M, K, 1.0f, col_buffer_tmp_ptr, N, weight_tmp_ptr, K, 0.0f, output_instance_ptr, N);
            }
//          SwapAxis<Device, T>(d, output_temp_4d_ptr, ToVector(TensorShape({num_ / im2col_step_, conv_out_channels_, im2col_step_, conv_out_spatial_dim_})), 1, 2);
    }
    return output;
}

std::vector<at::Tensor> deformable_conv2d_backward(
    at::Tensor input,
    at::Tensor filter,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor out_grad,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int num_groups,
    int deformable_groups,
    int im2col_step,
    bool no_bias
){
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(filter.type().is_cuda(), "filter must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");
    AT_ASSERTM(out_grad.type().is_cuda(), "mask must be a CUDA tensor");


    const int height = input.size(2);
    const int width = input.size(3);
    int kernel_h = filter.size(2);
    int kernel_w = filter.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(height_out==out_grad.size(2) && width_out == out_grad.size(3),
        "the calculated out shape won't match the out_grad_shape:(%d x %d vs %d x %d)",
            height_out, width_out, out_grad.size(2), out_grad.size(3));
    const int num_axes = 4;
    bool is_1x1_ = true;
    for (int i = 2; i < num_axes; ++i) {
            is_1x1_ &= filter.size(i) == 1; // only judge by the filter's shape
            if (!is_1x1_) break;
    }
    int num_ = input.size(0);// batch size
    int channels_ = input.size(1);// number of input channels
    int group_ = num_groups;//
    int conv_out_channels_ = filter.size(0); // output channel nums
    int conv_in_channels_ = channels_; // input channel nums

    int kernel_dim_ = conv_in_channels_ / group_ * filter.size(2) * filter.size(3); //Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    int conv_out_spatial_dim_ = height_out * width_out;
    int im2col_step_ = std::min(im2col_step, num_);

    int input_dim_ = input.size(1) * input.size(2) * input.size(3);// input image size (#channels * height * width)
    int input_offset_dim_ = offset.size(1) * offset.size(2) * offset.size(3); // 18 * H * W
    int input_mask_dim_ =  mask.size(1) * mask.size(2) * mask.size(3); // 9 * H * W

    auto col_buffer = at::empty({conv_in_channels_ * filter.size(2) * filter.size(3), im2col_step_, conv_out_spatial_dim_}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(filter);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);
    auto grad_weight_temp = at::zeros_like(filter);

    auto input_ptr = input.data<float>();
    auto weight_ptr = filter.data<float>();
    auto offset_ptr = offset.data<float>();
    auto mask_ptr = mask.data<float>();
    auto out_grad_ptr = out_grad.data<float>();
    auto grad_input_ptr = grad_input.data<float>();
    auto grad_weight_ptr = grad_weight.data<float>();

    auto grad_offset_ptr = grad_offset.data<float>();
    auto grad_mask_ptr = grad_mask.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();

    int M = kernel_dim_;
    int N = im2col_step_ * conv_out_spatial_dim_;
    int K = conv_out_channels_ / group_;

    TShape input_shape;
    TShape filter_shape;
    TShape col_buffer_shape;
    TShape stride_shape;
    TShape dilation_shape;
    TShape padding_shape;

    input_shape.push_back(input.size(0));
    input_shape.push_back(input.size(1));
    input_shape.push_back(input.size(2));
    input_shape.push_back(input.size(3));
    filter_shape.push_back(filter.size(2));
    filter_shape.push_back(filter.size(3));
    col_buffer_shape.push_back(conv_in_channels_ * filter.size(2) * filter.size(3));
    col_buffer_shape.push_back(im2col_step_);
    col_buffer_shape.push_back(height_out);
    col_buffer_shape.push_back(width_out);
    stride_shape.push_back(stride_h);
    stride_shape.push_back(stride_w);
    dilation_shape.push_back(dilation_h);
    dilation_shape.push_back(dilation_w);
    padding_shape.push_back(pad_h);
    padding_shape.push_back(pad_w);

    for(int n = 0;n < num_ / im2col_step_ ;++n){
        auto out_grad_instance_ptr = out_grad_ptr + n * group_ * K * N;
        for(int g = 0;g < group_;g++){
            auto weight_tmp_ptr = weight_ptr + g * M * K;
            auto out_grad_instance_tmp_ptr = out_grad_instance_ptr + g * K * N;
            auto col_buffer_tmp_ptr = col_buffer_ptr + g * M * N;
            THCudaBlas_Sgemm(state,
            'n', 't',
            N, M, K,
            1.0f,
            out_grad_instance_tmp_ptr, N,
            weight_tmp_ptr, M,
            0.0f,
            col_buffer_tmp_ptr, N);
        }
        deformable_col2im_coord(
                THCState_getCurrentStream(state),
                col_buffer_ptr,
                input_ptr + n * im2col_step_ * input_dim_,
                offset_ptr + n * im2col_step_ * input_offset_dim_,
                mask_ptr + n * im2col_step_ * input_mask_dim_,
                input_shape,
                col_buffer_shape,
                filter_shape,
                padding_shape,
                stride_shape,
                dilation_shape,
                deformable_groups,
                grad_offset_ptr + n * im2col_step_ * input_offset_dim_,
                grad_mask_ptr + n * im2col_step_ * input_mask_dim_);

        deformable_col2im(
                THCState_getCurrentStream(state),
                col_buffer_ptr,
                offset_ptr + n * im2col_step_ * input_offset_dim_,
                mask_ptr + n * im2col_step_ * input_mask_dim_,
                input_shape,
                col_buffer_shape,
                filter_shape,
                padding_shape,
                stride_shape,
                dilation_shape,
                deformable_groups,
                grad_input_ptr + n * im2col_step_ * input_dim_);

        deformable_im2col(
                THCState_getCurrentStream(state),
                input_ptr + n * im2col_step_ * input_dim_,
                offset_ptr + n * im2col_step_ * input_offset_dim_,
                mask_ptr + n * im2col_step_ * input_mask_dim_,
                input_shape,
                col_buffer_shape,
                filter_shape,
                padding_shape,
                stride_shape,
                dilation_shape,
                deformable_groups,
                col_buffer_ptr);
            for(int g = 0;g < group_;g++){
                auto grad_weight_tmp_ptr = grad_weight_ptr + g * M * K;
                auto out_grad_instance_tmp_ptr = out_grad_instance_ptr + g * K * N;
                auto col_buffer_tmp_ptr = col_buffer_ptr + g * M * N;
                THCudaBlas_Sgemm(state,
                't', 'n',
                M, K, N,
                1.0f,
                col_buffer_tmp_ptr, N,
                out_grad_instance_tmp_ptr, N,
                1.0f,
                grad_weight_tmp_ptr, M);
            }
    }
    return {grad_input, grad_weight, grad_offset, grad_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &deformable_conv2d_forward, "deformable_conv2d forward (CUDA)");
  m.def("backward", &deformable_conv2d_backward, "deformable_conv2d backward (CUDA)");
}