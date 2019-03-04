# Deformable_Conv2d_Pytorch
deformable_conv2d layer implemented in pytorch. and I wrote several articles on ZHIHU, you can read it for more detailed information <br> 1.[deformable变形卷积pytorch实现(第一节Custom op extension)](https://zhuanlan.zhihu.com/p/58173937) 2.[deformable变形卷积pytorch实现(第二节deformable_conv2d 实现一)](https://zhuanlan.zhihu.com/p/58185157)
<br>besides, I also complete an example net, [here](https://github.com/BIGKnight/ADCrowdNet_tensorflow_implementation)
and I'm very sorry that I did not implement the swapaxis methods, so the im2col_step parameter are only allowed using value one.
## ENVIRONMENT CONFIGURATION
1. OS: ubuntu16.04 <br>
2. GPU: 1 gtx1080Ti <br>
3. LANGUAGE: python3.6.8 & c++11 & cuda c<br>
4. DL FRAMEWORK: Pytorch 1.0.1<br>
5. ANCILLARY LIB: setuptools: 36.4.0, numpy: 1.15.4
6. GPU API: NVIDIA CUDA 9.0 & cuDNN 7.0
7. COMPILE: nvcc & gcc 5.4.0
## INSTALL PROCEDURE
1. cd "current project"
2. run mask.sh<br>tips: you need to modify the path parameters first. and all the -I and -L path in the nvcc and g++ orders need to be checked, make sure they are the correct path in your system<br>
3. import deformable_conv2d_wrapper.py into your python file, and you can call the class BasicDeformableConv2D then.