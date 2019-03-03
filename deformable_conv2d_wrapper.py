import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
# our module
# 这里有个很坑的地方, 这个完全是pytorch的问题,
import deformable_conv2d_gpu


class DeformableConv2DFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 14:
            print("Wrong parameter number, check your input!")
            return
        input = args[0]
        filter = args[1]
        offset = args[2]
        mask = args[3]
        ctx.stride_h = args[4]
        ctx.stride_w = args[5]
        ctx.pad_h = args[6]
        ctx.pad_w = args[7]
        ctx.dilation_h = args[8]
        ctx.dilation_w = args[9]
        ctx.num_groups = args[10]
        ctx.deformable_groups = args[11]
        ctx.im2col_step = args[12]
        ctx.no_bias = args[13]
        output = deformable_conv2d_gpu.forward(
            input,
            filter,
            offset,
            mask,
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.num_groups,
            ctx.deformable_groups,
            ctx.im2col_step,
            ctx.no_bias)
        # print(output)
        ctx.save_for_backward(input, filter, offset, mask)
        return output

# backward 的返回值个数要和forward的输入个数等同
    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, filter, offset, mask = ctx.saved_tensors
        grad_input, grad_weight, grad_offset, grad_mask = deformable_conv2d_gpu.backward(
            input,
            filter,
            offset,
            mask,
            grad_outputs[0],
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.num_groups,
            ctx.deformable_groups,
            ctx.im2col_step,
            ctx.no_bias)
        return grad_input, grad_weight, grad_offset, grad_mask, \
               None, None, None, None, None, None, None, None, None, None


class DeformableConv2DLayer(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel__size,
                 stride_h, stride_w,
                 padding,
                 dilation_h, dilation_w,
                 num_groups,
                 deformable_groups,
                 im2col_step,
                 no_bias
                 ):
        super(DeformableConv2DLayer, self).__init__()
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = padding
        self.pad_w = padding
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.num_groups = num_groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.no_bias = no_bias
        self.weight = nn.Parameter(
            torch.zeros(
                out_channels,
                in_channels,
                kernel__size,
                kernel__size,
                dtype=torch.float32
            )
        )
        nn.init.xavier_uniform_(self.weight, gain=1)


# apply() takes no keyword arguments
    def forward(self,
                inputs,
                offset,
                mask):
        return DeformableConv2DFunction.apply(
            inputs,
            self.weight,
            offset,
            mask,
            self.stride_h, self.stride_w,
            self.pad_h, self.pad_w,
            self.dilation_h, self.dilation_w,
            self.num_groups,
            self.deformable_groups,
            self.im2col_step,
            self.no_bias)
