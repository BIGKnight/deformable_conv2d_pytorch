from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
# 其实这里是先预处理, 编译, 汇编, 分别得到CUDAExtension里的每个文件的obj文件, 然后讲所有的.o文件用g++链接起来. 值得注意的是编译器再前两步都不管内存地址的事情, 所以函数声明不定义在-c条件下是可以编译的, 但是链接会报错.
# 因为不同的ABI对应的应用程序核操作系统之间的协定是不同的, 我的理解诶是API相当于一个上层的对于接口函数的描述, ABI是底层的面向实现的一些协定.
# C++ 11 核C++ 98的ABI接口是不同的, 这里的话, 对于pytorch十一是支持c++ 11的ABI的, 而坑的地方是我们可以看到再setup.py脚本内部的gcc加了条件_GLIBCXX_USE_CXX11_ABI=0, 表示不使用c++ 11的ABI
# 所以这样setup生成的.so是不支持c++11的ABI的, 这就导致.so和实际pytorch脚本是两个不同的c++版本, 就gg了.
# 解决的方式.1 暴力的方式就是直接改回pytorch 0.4.0. 2. 或者修改setup生成的.so, 用自己编译链接后的.so替代掉这个.
setup(name='deformable_conv2d',
      ext_modules=[CUDAExtension('deformable_conv2d_gpu', ['deformable_conv2d.cc', 'deformable_conv2d_cuda.cu']),],
      cmdclass={'build_ext': BuildExtension})

# it is the non-cuda setup script here
# setup(name='deformable_conv2d',
#       ext_modules=[CppExtension('deformable_conv2d', ['deformable_conv2d.cc'])],
#       cmdclass={'build_ext': BuildExtension})
# the equivalent code:
# setuptools.Extension(name='deformable_conv2d', sources=['deformable_conv2d.cc'],
# include_dirs=torch.utils.cpp_extension.include_paths(), language='c++')
